#!/usr/bin/env python3
"""
================================================================================
SCRIPT DI PRE-ADDESTRAMENTO AUTO-SUPERVISIONATO (SELF-SUPERVISED PRETRAINING)
================================================================================
Questo script esegue il pre-addestramento dell'encoder MiniNet sul traffico di rete
utilizzando il paradigma del "Masked Autoencoding" (ricostruzione di pacchetti nascosti).

COME FUNZIONA IN PAROLE SEMPLICI:
1. Prendiamo i flussi di traffico (ciascuno composto da una sequenza dei primi 20 pacchetti).
2. Nascondiamo casualmente circa il 30% dei pacchetti reali (con una maschera).
3. Chiediamo al modello (un piccolo Transformer) di indovinare cosa c'era nei pacchetti nascosti:
   - Dimensione del pacchetto (in byte, trasformata con logaritmo)
   - Tempo di interarrivo (in millisecondi dal pacchetto precedente, con logaritmo)
   - Direzione (in entrata = +1, in uscita = -1)
4. Il modello impara così le regolarità temporali e strutturali del traffico senza
   bisogno di sapere se un flusso è malware o benigno.
5. Alla fine, valutiamo quanto sono buone le rappresentazioni apprese tramite un "Linear Probe"
   sul validation set: questo punteggio (F1-score) serve come metrica obiettivo per l'HPO
   (Hyperparameter Optimization / Ottimizzazione degli Iperparametri).

UTILIZZO DA TERMINALE:
    python scripts/pretrain.py --config scripts/config.json
Oppure sovrascrivendo parametri per test rapidi o HPO:
    python scripts/pretrain.py --config scripts/config.json --learning_rate 0.0003 --d_model 64
================================================================================
"""

import os
import sys
import json
import time
import argparse
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder


# ==============================================================================
# 1. FUNZIONE PER FISSARE I SEED (RIPRODUCIBILITÀ)
# ==============================================================================
def set_seed(seed: int):
    """
    Fissa i semi di tutti i generatori di numeri casuali (Python, NumPy, PyTorch).
    In questo modo, se rilanciamo l'esperimento con lo stesso seed, otterremo
    esattamente gli stessi numeri e gli stessi pesi iniziali.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Su CPU, assicuriamo operazioni deterministiche
    os.environ["PYTHONHASHSEED"] = str(seed)


# ==============================================================================
# 2. DATASET PYTORCH PER LE SEQUENZE SPLT
# ==============================================================================
class FlowSequenceDataset(Dataset):
    """
    Legge le sequenze estratte dal DataFrame Parquet e le converte in tensori PyTorch.
    Ogni elemento restituito è una tupla:
      - x: tensore float (n_pkt, 3) contenente [direzione, dimensione_log, interarrivo_log]
      - pad_mask: tensore booleano (n_pkt) dove True indica 'pacchetto assente/padding'
      - index: indice originale della riga nel DataFrame (utile per tracciare i flussi)
    """
    def __init__(self, df: pd.DataFrame):
        # np.stack converte la serie di array/liste in una matrice 2D continua e veloce in memoria
        self.direction = np.stack(df["splt_direction"].to_numpy()).astype(np.float32)
        self.ps = np.stack(df["splt_ps"].to_numpy()).astype(np.float32)
        self.piat = np.stack(df["splt_piat_ms"].to_numpy()).astype(np.float32)
        self.mask = np.stack(df["splt_mask"].to_numpy()).astype(np.bool_)
        self.indices = df.index.to_numpy()

    def __len__(self):
        # Numero totale di flussi in questa partizione (es. train o val)
        return len(self.indices)

    def __getitem__(self, i):
        # Creiamo la matrice (20 pacchetti x 3 caratteristiche) per il flusso i-esimo
        # Colonna 0 = direzione (-1 o +1)
        # Colonna 1 = dimensione logaritmica standardizzata
        # Colonna 2 = tempo di interarrivo logaritmico standardizzato
        x = np.stack([self.direction[i], self.ps[i], self.piat[i]], axis=1)

        # In PyTorch Transformer:
        # pad_mask = True significa "ignora questa posizione perché è padding (pacchetto fittizio)"
        # self.mask invece valeva True per i pacchetti reali, quindi invertiamo con '~'
        pad_mask = ~self.mask[i]

        return (
            torch.from_numpy(x),
            torch.from_numpy(pad_mask),
            int(self.indices[i])
        )


# ==============================================================================
# 3. ARCHITETTURA DEL MODELLO (MININET ENCODER)
# ==============================================================================
class MiniNetEncoder(nn.Module):
    """
    Encoder basato su architettura Transformer (simile a netFound, ma miniaturizzato).
    Prende in input una sequenza di pacchetti e restituisce una rappresentazione
    vettoriale densa per ciascuna posizione, più un token riassuntivo [CLS].
    """
    def __init__(self, n_pkt=20, d=64, layers=4, heads=4, dim_feedforward_multiplier=4):
        super().__init__()
        self.d = d
        self.n_pkt = n_pkt

        # 1. Proiezione lineare iniziale: trasforma le 3 feature di ciascun pacchetto
        # in un vettore di dimensione 'd' (es. 64)
        self.inp = nn.Linear(3, d)

        # 2. Token speciale [CLS] (Classification Token):
        # È un vettore addestrabile a 64 dimensioni aggiunto in testa alla sequenza (posizione 0).
        # A fine addestramento, l'uscita in questa posizione fungerà da "carta d'identità"
        # complessiva dell'intero flusso.
        self.cls = nn.Parameter(torch.zeros(1, 1, d))

        # 3. Positional Embedding:
        # Poiché il Transformer elabora tutti i pacchetti contemporaneamente (in parallelo)
        # e non in ordine temporale sequenziale, aggiungiamo a ogni pacchetto un vettore
        # che gli insegna in quale posizione si trova (1°, 2°, ..., 20° pacchetto).
        self.pos = nn.Parameter(torch.zeros(1, n_pkt + 1, d))

        # 4. Strati del Transformer Encoder:
        # norm_first=True usa la formulazione moderna Pre-LayerNorm (più stabile da addestrare).
        layer_conf = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=heads,
            dim_feedforward=dim_feedforward_multiplier * d,
            batch_first=True,
            norm_first=True
        )
        self.body = nn.TransformerEncoder(layer_conf, num_layers=layers)

    def forward(self, x, pad_mask):
        """
        x: tensore di dimensione (Batch_size, n_pkt, 3)
        pad_mask: tensore booleano di dimensione (Batch_size, n_pkt)
        """
        batch_size = x.size(0)

        # Proiezione lineare dei pacchetti nello spazio a 'd' dimensioni
        h = self.inp(x)  # (Batch_size, n_pkt, d)

        # Aggiungiamo il token CLS in prima posizione (posizione 0)
        cls_expanded = self.cls.expand(batch_size, -1, -1)
        h = torch.cat([cls_expanded, h], dim=1)  # Ora ha lunghezza (n_pkt + 1)

        # Aggiungiamo le informazioni di posizione
        h = h + self.pos

        # Il token CLS non deve MAI essere mascherato dal padding, quindi creiamo
        # una colonna di False da anteporre alla maschera dei pacchetti
        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=pad_mask.device)
        extended_pad_mask = torch.cat([cls_mask, pad_mask], dim=1)

        # Elaborazione attraverso tutti gli strati di attenzione del Transformer
        out = self.body(h, src_key_padding_mask=extended_pad_mask)
        return out  # (Batch_size, n_pkt + 1, d)


# ==============================================================================
# 4. TESTA DI RICOSTRUZIONE (RECONSTRUCTION HEAD)
# ==============================================================================
class ReconstructionHead(nn.Module):
    """
    Modulo applicato sull'uscita dell'encoder durante il pre-addestramento.
    Prende i vettori a dimensione 'd' e li proietta all'indietro per ricostruire
    i valori originali del pacchetto mascherato:
    - size_head: stima la dimensione continua (MSE)
    - iat_head: stima l'interarrivo continuo (MSE)
    - direction_head: stima la probabilità che il pacchetto sia in ingresso (BCE)
    """
    def __init__(self, d=64):
        super().__init__()
        self.size_head = nn.Linear(d, 1)
        self.iat_head = nn.Linear(d, 1)
        self.direction_head = nn.Linear(d, 1)

    def forward(self, h):
        """
        h: tensore (Batch_size, n_pkt, d) corrispondente alle sole posizioni dei pacchetti
        """
        pred_size = self.size_head(h).squeeze(-1)            # (Batch_size, n_pkt)
        pred_iat = self.iat_head(h).squeeze(-1)              # (Batch_size, n_pkt)
        pred_direction = self.direction_head(h).squeeze(-1)  # (Batch_size, n_pkt)
        return pred_size, pred_iat, pred_direction


# ==============================================================================
# 5. LOGICA DI MASCHERAMENTO E CALCOLO DELLA LOSS
# ==============================================================================
def create_pretraining_mask(pad_mask: torch.Tensor, mask_ratio: float = 0.30) -> torch.Tensor:
    """
    Genera la maschera booleana dei pacchetti da nascondere al modello:
    - Maschera casualmente una frazione pari a 'mask_ratio' (default 30%) dei pacchetti VALIDI.
    - Non maschera mai le posizioni di padding (i pacchetti inesistenti).
    - Garantisce che ogni flusso abbia almeno 1 pacchetto mascherato (altrimenti non ci sarebbe errore da calcolare).
    """
    valid = ~pad_mask
    rand_vals = torch.rand(pad_mask.shape, device=pad_mask.device)
    mask = (rand_vals < mask_ratio) & valid

    # Controllo: se per sfortuna un flusso non ha nessun pacchetto mascherato, ne forziamo uno valido
    flussi_vuoti = torch.where(~mask.any(dim=1) & valid.any(dim=1))[0]
    for idx in flussi_vuoti:
        posizioni_valide = torch.where(valid[idx])[0]
        if len(posizioni_valide) > 0:
            scelta = posizioni_valide[torch.randint(len(posizioni_valide), (1,))]
            mask[idx, scelta] = True

    return mask


def compute_reconstruction_loss(pred_size, pred_iat, pred_dir, target_x, mask):
    """
    Calcola l'errore di predizione (Loss) ESCLUSIVAMENTE sulle posizioni mascherate.
    Se un pacchetto non era nascosto, il modello non viene punito/premiato su di esso.
    """
    # I target reali estratti dal tensore originale
    # target_x ha forma (B, n_pkt, 3) dove:
    # colonna 0 = direzione (-1 per out, +1 per in -> convertiamo in 0/1 per la BCE)
    # colonna 1 = packet size
    # colonna 2 = inter-arrival time
    target_dir = (target_x[:, :, 0] > 0).float()
    target_size = target_x[:, :, 1]
    target_iat = target_x[:, :, 2]

    # Filtriamo prendendo solo le coordinate dove mask == True
    pred_s_masked = pred_size[mask]
    pred_i_masked = pred_iat[mask]
    pred_d_masked = pred_dir[mask]

    target_s_masked = target_size[mask]
    target_i_masked = target_iat[mask]
    target_d_masked = target_dir[mask]

    # 1. Errore quadratico medio (MSE) per la dimensione
    loss_size = nn.functional.mse_loss(pred_s_masked, target_s_masked)

    # 2. Errore quadratico medio (MSE) per l'interarrivo
    loss_iat = nn.functional.mse_loss(pred_i_masked, target_i_masked)

    # 3. Binary Cross-Entropy con Logits per la direzione (in entrata vs uscita)
    loss_dir = nn.functional.binary_cross_entropy_with_logits(pred_d_masked, target_d_masked)

    # Loss totale = somma delle 3 componenti
    total_loss = loss_size + loss_iat + loss_dir
    return total_loss, loss_size.item(), loss_iat.item(), loss_dir.item()


# ==============================================================================
# 6. VALUTAZIONE: LINEAR PROBE SUL VALIDATION SET (METRICA CHIAVE PER HPO)
# ==============================================================================
def evaluate_linear_probe(model, train_loader, val_loader, y_train, y_val, max_iter=200, solver="saga", seed=42):
    """
    Estrae le rappresentazioni 'z_cls' congelate sia per il train che per il val,
    quindi addestra una semplice Regressione Logistica (probe lineare) per predire le 20 classi.
    
    PERCHÉ È LA METRICA MIGLIORE PER L'HPO?
    Durante il pre-training la loss di ricostruzione potrebbe scendere, ma questo non garantisce
    al 100% che l'encoder stia imparando proprietà utili per classificare il traffico.
    Il probe lineare misura direttamente quanta informazione semantica utile è linearmente
    accessibile in 'z_cls'. Un F1-score più alto = pre-training qualitativamente migliore!
    """
    model.eval()
    device = next(model.parameters()).device

    print("\n--> Estrazione rappresentazioni z_cls con encoder congelato per il Linear Probe...")
    def extract_z_cls(loader):
        reps = []
        with torch.inference_mode():
            for x, pad_mask, _ in loader:
                x = x.to(device)
                pad_mask = pad_mask.to(device)
                out = model(x, pad_mask)
                # Il token [CLS] si trova sempre all'indice 0 della sequenza
                z_cls = out[:, 0, :].cpu().numpy()
                reps.append(z_cls)
        return np.concatenate(reps, axis=0)

    start_ext = time.time()
    z_train = extract_z_cls(train_loader)
    z_val = extract_z_cls(val_loader)
    print(f"    Estrazione completata in {time.time() - start_ext:.1f}s. Shape z_train: {z_train.shape}, z_val: {z_val.shape}")

    print(f"--> Addestramento Regressione Logistica (solver={solver}, max_iter={max_iter})...")
    start_fit = time.time()
    probe = LogisticRegression(
        max_iter=max_iter,
        solver=solver,
        random_state=seed,
        n_jobs=-1
    )
    probe.fit(z_train, y_train)
    print(f"    Fit completato in {time.time() - start_fit:.1f}s.")

    # Predizione sul validation set
    val_preds = probe.predict(z_val)
    val_acc = float(accuracy_score(y_val, val_preds))
    val_f1_macro = float(f1_score(y_val, val_preds, average="macro"))
    val_f1_weighted = float(f1_score(y_val, val_preds, average="weighted"))

    return {
        "probe_val_accuracy": val_acc,
        "probe_val_f1_macro": val_f1_macro,
        "probe_val_f1_weighted": val_f1_weighted
    }


# ==============================================================================
# 7. FUNZIONE PRINCIPALE DI ADDESTRAMENTO
# ==============================================================================
def run_pretraining(cfg: dict):
    """
    Funzione principale che orchestra l'intero processo di pre-training:
    caricamento dati -> creazione modello -> ciclo di epoche -> salvataggio -> metriche.
    """
    # 1. Impostazione seed e device (rigorosamente CPU come richiesto)
    set_seed(cfg["seed"])
    device = torch.device("cpu")
    print(f"\n=======================================================")
    print(f"AVVIO PRE-TRAINING SELF-SUPERVISED")
    print(f"Device utilizzato: {device}")
    print(f"Configurazione: {json.dumps({k: v for k, v in cfg.items() if not k.startswith('_')}, indent=2)}")
    print(f"=======================================================\n")

    # 2. Caricamento del dataset
    dataset_path = cfg["dataset_path"]
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset non trovato al percorso: {dataset_path}")

    print(f"Caricamento dati da {dataset_path}...")
    t0 = time.time()
    df = pd.read_parquet(dataset_path)
    print(f"Caricati {len(df)} flussi totali in {time.time() - t0:.2f}s.")

    # 3. Suddivisione split (usiamo solo TRAIN per il pre-training per evitare data-snooping)
    df_train = df[df["split"] == "train"]
    df_val = df[df["split"] == "val"]
    print(f"Split utilizzati: Train={len(df_train)}, Val={len(df_val)}")

    train_dataset = FlowSequenceDataset(df_train)
    val_dataset = FlowSequenceDataset(df_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 0)
    )

    # 4. Inizializzazione dell'Encoder e della Testa di Ricostruzione
    model = MiniNetEncoder(
        n_pkt=cfg["n_pkt"],
        d=cfg["d_model"],
        layers=cfg["n_layers"],
        heads=cfg["n_heads"],
        dim_feedforward_multiplier=cfg.get("dim_feedforward_multiplier", 4)
    ).to(device)

    recon_head = ReconstructionHead(d=cfg["d_model"]).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_recon_params = sum(p.numel() for p in recon_head.parameters() if p.requires_grad)
    print(f"Parametri addestrabili: Encoder={total_params:,}, ReconstructionHead={total_recon_params:,}")

    # 5. Ottimizzatore AdamW:
    # Aggiorna insieme i pesi sia dell'encoder sia della testa di ricostruzione
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(recon_head.parameters()),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"]
    )

    # 6. Ciclo di addestramento sulle epoche
    epochs = cfg["epochs"]
    log_interval = cfg.get("log_interval", 100)
    history = []

    start_training_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        recon_head.train()

        running_loss = 0.0
        running_s = 0.0
        running_i = 0.0
        running_d = 0.0
        n_batches = len(train_loader)

        print(f"\n--- Inizio Epoca {epoch}/{epochs} ---")
        for step, (x, pad_mask, _) in enumerate(train_loader, start=1):
            x = x.to(device)
            pad_mask = pad_mask.to(device)

            # A. Generiamo la maschera per nascondere il 30% dei pacchetti validi
            mask = create_pretraining_mask(pad_mask, mask_ratio=cfg["mask_ratio"])

            # B. Creiamo l'input corrotto (sostituiamo i valori mascherati con zeri per non farli sbirciare)
            x_corrupted = x.clone()
            x_corrupted[mask] = 0.0

            # C. Azzeriamo i gradienti accumulati nei passi precedenti
            optimizer.zero_grad()

            # D. Forward pass attraverso l'encoder
            # out ha dimensione: (Batch, n_pkt + 1, d)
            out = model(x_corrupted, pad_mask)

            # Ignoriamo il token CLS (indice 0) e prendiamo solo le posizioni dei pacchetti (indici 1..20)
            h_packets = out[:, 1:, :]

            # E. Forward pass attraverso la testa di ricostruzione
            pred_size, pred_iat, pred_dir = recon_head(h_packets)

            # F. Calcolo della loss solo sui pacchetti mascherati
            loss, l_s, l_i, l_d = compute_reconstruction_loss(
                pred_size, pred_iat, pred_dir, x, mask
            )

            # G. Backward pass (calcolo delle derivate/gradienti con autograd)
            loss.backward()

            # H. Aggiornamento dei pesi (discesa del gradiente con AdamW)
            optimizer.step()

            # Accumulo statistiche per il log
            running_loss += loss.item()
            running_s += l_s
            running_i += l_i
            running_d += l_d

            if step % log_interval == 0 or step == n_batches:
                avg_l = running_loss / step
                avg_s = running_s / step
                avg_i = running_i / step
                avg_d = running_d / step
                elapsed = time.time() - epoch_start
                print(f"  [Epoca {epoch} | Batch {step:4d}/{n_batches}] "
                      f"Loss: {avg_l:.4f} (Size: {avg_s:.4f}, IAT: {avg_i:.4f}, Dir: {avg_d:.4f}) "
                      f"[{elapsed:.1f}s]")

        train_epoch_loss = running_loss / n_batches

        # 7. Valutazione della loss di ricostruzione sul Validation Set
        print("--> Calcolo loss di ricostruzione su Validation Set...")
        model.eval()
        recon_head.eval()
        val_loss_sum = 0.0
        n_val_batches = len(val_loader)

        with torch.inference_mode():
            for x_val, pad_mask_val, _ in val_loader:
                x_val = x_val.to(device)
                pad_mask_val = pad_mask_val.to(device)

                mask_val = create_pretraining_mask(pad_mask_val, mask_ratio=cfg["mask_ratio"])
                x_val_corrupted = x_val.clone()
                x_val_corrupted[mask_val] = 0.0

                out_val = model(x_val_corrupted, pad_mask_val)
                h_val_packets = out_val[:, 1:, :]
                p_s, p_i, p_d = recon_head(h_val_packets)

                v_loss, _, _, _ = compute_reconstruction_loss(p_s, p_i, p_d, x_val, mask_val)
                val_loss_sum += v_loss.item()

        val_recon_loss = val_loss_sum / n_val_batches
        print(f"✓ Epoca {epoch} completata: Train Loss = {train_epoch_loss:.4f}, Val Loss = {val_recon_loss:.4f}")

        history.append({
            "epoch": epoch,
            "train_recon_loss": float(train_epoch_loss),
            "val_recon_loss": float(val_recon_loss),
            "epoch_duration_seconds": float(time.time() - epoch_start)
        })

    total_training_duration = time.time() - start_training_time
    print(f"\nAddestramento completato in {total_training_duration / 60:.2f} minuti.")

    # 8. Valutazione opzionale con Linear Probe (per guidare l'HPO)
    probe_metrics = {}
    if cfg.get("evaluate_probe_on_val", True):
        label_enc = LabelEncoder()
        y_train_20 = label_enc.fit_transform(df_train["label_20"])
        y_val_20 = label_enc.transform(df_val["label_20"])

        probe_metrics = evaluate_linear_probe(
            model=model,
            train_loader=DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=False),
            val_loader=val_loader,
            y_train=y_train_20,
            y_val=y_val_20,
            max_iter=cfg.get("probe_max_iter", 300),
            solver=cfg.get("probe_solver", "saga"),
            seed=cfg["seed"]
        )
        print(f"\n=======================================================")
        print(f"RISULTATI LINEAR PROBE SU VALIDATION (METRICA HPO):")
        print(f"  Accuracy:    {probe_metrics['probe_val_accuracy'] * 100:.2f}%")
        print(f"  F1 (Macro):   {probe_metrics['probe_val_f1_macro'] * 100:.2f}%")
        print(f"  F1 (Weighted):{probe_metrics['probe_val_f1_weighted'] * 100:.2f}%")
        print(f"=======================================================")

    # 9. Salvataggio del modello pre-addestrato e delle metriche
    out_dir = cfg.get("output_dir", "res/pretraining")
    os.makedirs(out_dir, exist_ok=True)

    save_model_path = os.path.join(out_dir, cfg.get("save_model_filename", "pretrained_encoder.pt"))
    torch.save({
        "model_state_dict": model.state_dict(),
        "reconstruction_head_state_dict": recon_head.state_dict(),
        "config": cfg,
        "history": history,
        "probe_metrics": probe_metrics
    }, save_model_path)
    print(f"\n✓ Modello pre-addestrato salvato in: {save_model_path}")

    # Salvataggio metriche in formato JSON (fondamentale per raccogliere i risultati dell'HPO)
    final_results = {
        "status": "SUCCESS",
        "config": cfg,
        "total_params": total_params,
        "final_train_recon_loss": history[-1]["train_recon_loss"],
        "final_val_recon_loss": history[-1]["val_recon_loss"],
        "training_duration_seconds": total_training_duration,
        "history": history,
        **probe_metrics
    }

    metrics_path = os.path.join(out_dir, cfg.get("metrics_filename", "pretrain_metrics.json"))
    with open(metrics_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"✓ Metriche salvate in: {metrics_path}")

    return final_results


# ==============================================================================
# 8. PARSER DEGLI ARGOMENTI DA LINEA DI COMANDO
# ==============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Lancia il pre-training auto-supervisionato di MiniNet con supporto HPO."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="scripts/config.json",
        help="Percorso del file di configurazione JSON base."
    )
    # Argomenti opzionali per sovrascrivere la configurazione (utilissimi per script HPO/Optuna)
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate per AdamW.")
    parser.add_argument("--d_model", type=int, default=None, help="Dimensione nascosta dell'encoder.")
    parser.add_argument("--n_layers", type=int, default=None, help="Numero di strati Transformer.")
    parser.add_argument("--n_heads", type=int, default=None, help="Numero di teste di attenzione.")
    parser.add_argument("--mask_ratio", type=float, default=None, help="Percentuale di pacchetti mascherati.")
    parser.add_argument("--batch_size", type=int, default=None, help="Dimensione del batch.")
    parser.add_argument("--epochs", type=int, default=None, help="Numero di epoche.")
    parser.add_argument("--seed", type=int, default=None, help="Seed per la riproducibilità.")
    parser.add_argument("--output_dir", type=str, default=None, help="Cartella di output per pesi e metriche.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if not os.path.exists(args.config):
        print(f"ERRORE: File di configurazione non trovato: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = json.load(f)

    # Sovrascrittura dei parametri da CLI se forniti
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.d_model is not None:
        config["d_model"] = args.d_model
    if args.n_layers is not None:
        config["n_layers"] = args.n_layers
    if args.n_heads is not None:
        config["n_heads"] = args.n_heads
    if args.mask_ratio is not None:
        config["mask_ratio"] = args.mask_ratio
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.seed is not None:
        config["seed"] = args.seed
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    # Eseguiamo il pre-training
    results = run_pretraining(config)
    print("\nEsecuzione terminata con successo.")
