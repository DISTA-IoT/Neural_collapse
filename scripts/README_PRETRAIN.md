# Guida al Pre-training Self-Supervised e HPO

Questa cartella contiene tutto il necessario per eseguire il pre-addestramento dell'encoder **MiniNet** sul traffico di rete senza usare notebook, ideale sia per run singoli che per campagne di ottimizzazione degli iperparametri (**HPO**).

---

## 1. File inclusi

- [config.json](file:///home/jesusc/Neural_collapse/scripts/config.json): File di configurazione in formato JSON che racchiude tutti i parametri di dataset, modello Transformer, ottimizzazione, mascheramento e valutazione.
- [pretrain.py](file:///home/jesusc/Neural_collapse/scripts/pretrain.py): Script Python autonomo e commentato riga per riga in italiano.

---

## 2. Come lanciare il pre-training

### Esecuzione standard con i parametri di default:
```bash
python scripts/pretrain.py --config scripts/config.json
```

### Esecuzione con sovrascrittura da riga di comando (comodissimo per test veloci):
Puoi cambiare al volo qualsiasi iperparametro senza modificare il file JSON:
```bash
python scripts/pretrain.py --config scripts/config.json \
    --learning_rate 0.0003 \
    --d_model 64 \
    --n_layers 4 \
    --epochs 2 \
    --batch_size 256
```

---

## 3. Parametri nel `config.json` spiegati semplice

| Parametro | Descrizione | Perché è utile per l'HPO |
| :--- | :--- | :--- |
| `mask_ratio` (es. `0.30`) | Percentuale di pacchetti da nascondere (30%). | Se mascheri troppo poco il modello impara poco; se mascheri troppo il problema diventa impossibile. Si possono testare `0.15`, `0.30`, `0.45`. |
| `d_model` (es. `64`) | Dimensione dei vettori di rappresentazione interna. | Vettori più larghi (es. 128) contengono più informazione ma richiedono più memoria/calcolo. Vettori più stretti (es. 32) sono più veloci. |
| `n_layers` (es. `4`) | Numero di blocchi Transformer in cascata. | Più strati permettono relazioni più complesse. Valori tipici: `2`, `4`, `6`. |
| `n_heads` (es. `4`) | Numero di teste di attenzione multi-head. | Deve essere un divisore di `d_model` (es. per `d=64`: 2, 4 o 8). |
| `learning_rate` (es. `0.001`) | Ampiezza del passo di discesa del gradiente. | È l'iperparametro più importante: valori tipici `1e-4`, `3e-4`, `1e-3`, `3e-3`. |
| `weight_decay` (es. `1e-4`) | Regolarizzazione per evitare che i pesi crescano troppo. | Aiuta a stabilizzare la geometria. |
| `epochs` (es. `1`) | Quante volte il modello vede l'intero training set. | Nel nostro setup 1-2 epoche sono già ricche di aggiornamenti (migliaia di batch). |

---

## 4. Come funziona la valutazione per l'HPO?

Al termine dell'addestramento, lo script:
1. Calcola la **Validation Reconstruction Loss** (quanto bene indovina i pacchetti nascosti su flussi mai visti).
2. Congela l'encoder ed estrae le rappresentazioni `z_cls`.
3. Addestra un **Linear Probe** (Regressione Logistica) sul compito a 20 classi e misura la **Validation Accuracy** e l'**F1-macro**.

Tutti i risultati vengono salvati automaticamente in `res/pretraining/pretrain_metrics.json`.

### Esempio di integrazione con Optuna (HPO automatico):
Se in futuro vorrai usare Optuna per trovare in automatico la migliore combinazione, la funzione obiettivo è semplice come:

```python
import optuna
import subprocess
import json

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    mask_ratio = trial.suggest_categorical("mask_ratio", [0.20, 0.30, 0.40])
    d_model = trial.suggest_categorical("d_model", [32, 64])
    
    cmd = [
        "python", "scripts/pretrain.py",
        "--config", "scripts/config.json",
        "--learning_rate", str(lr),
        "--mask_ratio", str(mask_ratio),
        "--d_model", str(d_model),
        "--output_dir", f"res/hpo_trial_{trial.number}"
    ]
    subprocess.run(cmd, check=True)
    
    with open(f"res/hpo_trial_{trial.number}/pretrain_metrics.json") as f:
        res = json.load(f)
        
    # Obiettivo: massimizzare l'F1-score del probe lineare sul validation set!
    return res["probe_val_f1_macro"]

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=10)
```
