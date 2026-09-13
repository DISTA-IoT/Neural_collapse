import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score


# ============================================================
# CONFIGURAZIONE
# ============================================================

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.001

DATASET_PATH = "data/processed/preprocessed.parquet"
MODEL_PATH = "data/processed/model_best.pt"

print("Device:", DEVICE)


# ============================================================
# DATASET OTTIMIZZATO
# ============================================================

class FlowDataset(Dataset):

    def __init__(self, df, class_to_idx):

        self.x = []
        self.mask = []
        self.y = []

        print("Preparazione dei dati...")

        for _, row in df.iterrows():

            direction = np.asarray(
                row["direction"],
                dtype=np.float32
            )

            packet_size = np.asarray(
                row["packet_size"],
                dtype=np.float32
            )

            piat = np.asarray(
                row["piat"],
                dtype=np.float32
            )

            mask = np.asarray(
                row["mask"],
                dtype=np.float32
            )

            x = np.stack(
                [direction, packet_size, piat],
                axis=1
            )

            self.x.append(x)
            self.mask.append(mask)
            self.y.append(
                class_to_idx[row["label_20"]]
            )

        print("Conversione in tensori...")

        self.x = torch.tensor(
            np.asarray(self.x),
            dtype=torch.float32
        )

        self.mask = torch.tensor(
            np.asarray(self.mask),
            dtype=torch.float32
        )

        self.y = torch.tensor(
            self.y,
            dtype=torch.long
        )

        print("x:", self.x.shape)
        print("mask:", self.mask.shape)
        print("y:", self.y.shape)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        return (
            self.x[idx],
            self.mask[idx],
            self.y[idx]
        )


# ============================================================
# MODELLO
# ============================================================

class NeuralNetwork(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )

        self.classificatore = nn.Linear(
            64,
            num_classes
        )

    def forward(self, x, mask):

        h = self.encoder(x)

        mask = mask.unsqueeze(-1)

        somma = (h * mask).sum(dim=1)

        conteggio = mask.sum(
            dim=1
        ).clamp(min=1)

        h = somma / conteggio

        logits = self.classificatore(h)

        return logits, h


# ============================================================
# CARICAMENTO DATASET
# ============================================================

print("\nCaricamento dataset...")

df = pd.read_parquet(
    DATASET_PATH
)

print(
    "Flussi totali:",
    len(df)
)


# ============================================================
# CLASSI
# ============================================================

class_names = sorted(
    df["label_20"].unique()
)

class_to_idx = {
    classe: i
    for i, classe in enumerate(class_names)
}

idx_to_class = {
    i: classe
    for classe, i in class_to_idx.items()
}

num_classes = len(class_names)

print("\nNumero classi:", num_classes)

for i in range(num_classes):

    print(
        i,
        "->",
        idx_to_class[i]
    )


# ============================================================
# SPLIT
# ============================================================

df_train = df[
    df["split"] == "train"
].copy()

df_val = df[
    df["split"] == "val"
].copy()

print(
    "\nTrain:",
    len(df_train)
)

print(
    "Validation:",
    len(df_val)
)


# ============================================================
# CREAZIONE DATASET
# ============================================================

print("\nCreazione TRAIN dataset...")

dataset_train = FlowDataset(
    df_train,
    class_to_idx
)

print("\nCreazione VALIDATION dataset...")

dataset_val = FlowDataset(
    df_val,
    class_to_idx
)


# ============================================================
# DATALOADER
# ============================================================

loader_train = DataLoader(
    dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)

loader_val = DataLoader(
    dataset_val,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODELLO
# ============================================================

model = NeuralNetwork(
    num_classes
).to(DEVICE)

print("\nModello:")

print(model)


# ============================================================
# LOSS E OTTIMIZZATORE
# ============================================================

criterio = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

miglior_f1 = 0.0

print("\n" + "=" * 60)
print("INIZIO ADDESTRAMENTO")
print("=" * 60)


for epoca in range(EPOCHS):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    loss_totale = 0.0

    predizioni_train = []
    target_train = []

    for x, mask, y in loader_train:

        x = x.to(DEVICE)
        mask = mask.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits, _ = model(
            x,
            mask
        )

        loss = criterio(
            logits,
            y
        )

        loss.backward()

        optimizer.step()

        loss_totale += (
            loss.item() *
            x.size(0)
        )

        pred = torch.argmax(
            logits,
            dim=1
        )

        predizioni_train.extend(
            pred.detach().cpu().numpy()
        )

        target_train.extend(
            y.detach().cpu().numpy()
        )

    loss_train = (
        loss_totale /
        len(dataset_train)
    )

    f1_train = f1_score(
        target_train,
        predizioni_train,
        average="weighted"
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    loss_val_totale = 0.0

    predizioni_val = []
    target_val = []

    with torch.no_grad():

        for x, mask, y in loader_val:

            x = x.to(DEVICE)
            mask = mask.to(DEVICE)
            y = y.to(DEVICE)

            logits, _ = model(
                x,
                mask
            )

            loss = criterio(
                logits,
                y
            )

            loss_val_totale += (
                loss.item() *
                x.size(0)
            )

            pred = torch.argmax(
                logits,
                dim=1
            )

            predizioni_val.extend(
                pred.cpu().numpy()
            )

            target_val.extend(
                y.cpu().numpy()
            )


    loss_val = (
        loss_val_totale /
        len(dataset_val)
    )

    f1_val = f1_score(
        target_val,
        predizioni_val,
        average="weighted"
    )


    # ========================================================
    # RISULTATI
    # ========================================================

    print(
        f"Epoca {epoca + 1:02d}/{EPOCHS} | "
        f"Loss train: {loss_train:.4f} | "
        f"F1 train: {f1_train:.4f} | "
        f"Loss val: {loss_val:.4f} | "
        f"F1 val: {f1_val:.4f}"
    )


    # ========================================================
    # SALVATAGGIO
    # ========================================================

    if f1_val > miglior_f1:

        miglior_f1 = f1_val

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "class_to_idx":
                    class_to_idx,

                "num_classes":
                    num_classes
            },
            MODEL_PATH
        )

        print(
            "  -> Nuovo miglior modello! "
            f"F1 validation = {miglior_f1:.4f}"
        )


# ============================================================
# FINE
# ============================================================

print("\n" + "=" * 60)
print("ADDESTRAMENTO COMPLETATO")
print("=" * 60)

print(
    "Miglior F1 validation:",
    miglior_f1
)

print(
    "Modello salvato in:",
    MODEL_PATH
)
