import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURAZIONE
# ============================================================

DATASET = "data/processed/preprocessed.parquet"

BATCH_SIZE = 64
SEED = 42


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)


# ============================================================
# DATASET
# ============================================================

class FlussiDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(drop=True)

        # Le 20 classi
        self.classi = sorted(self.df["label_20"].unique())

        self.mappa_classi = {
            classe: i
            for i, classe in enumerate(self.classi)
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, indice):

        riga = self.df.iloc[indice]

        # Sequenze di lunghezza 20
        direction = np.asarray(riga["direction"], dtype=np.float32)
        packet_size = np.asarray(riga["packet_size"], dtype=np.float32)
        piat = np.asarray(riga["piat"], dtype=np.float32)
        mask = np.asarray(riga["mask"], dtype=np.float32)

        # Costruiamo una matrice:
        #
        # 20 pacchetti x 3 caratteristiche
        #
        sequenza = np.stack(
            [direction, packet_size, piat],
            axis=1
        )

        x = torch.tensor(sequenza, dtype=torch.float32)

        mask = torch.tensor(mask, dtype=torch.float32)

        y = self.mappa_classi[riga["label_20"]]

        y = torch.tensor(y, dtype=torch.long)

        return x, mask, y


# ============================================================
# CARICAMENTO
# ============================================================

print("\nCaricamento dataset...")

df = pd.read_parquet(DATASET)

print("Flussi totali:", len(df))


# ============================================================
# USIAMO SOLO TRAIN PER IL TEST
# ============================================================

df_train = df[df["split"] == "train"].copy()

print("Flussi train:", len(df_train))


# ============================================================
# DATASET E DATALOADER
# ============================================================

dataset = FlussiDataset(df_train)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# CONTROLLO CLASSI
# ============================================================

print("\nNumero classi:", len(dataset.classi))
print("Classi:")

for i, classe in enumerate(dataset.classi):
    print(i, "->", classe)


# ============================================================
# PRIMO BATCH
# ============================================================

x, mask, y = next(iter(loader))

print("\nDimensioni primo batch:")

print("x:", x.shape)
print("mask:", mask.shape)
print("y:", y.shape)

print("\nTipi:")

print("x:", x.dtype)
print("mask:", mask.dtype)
print("y:", y.dtype)


# ============================================================
# SPOSTAMENTO SU DEVICE
# ============================================================

x = x.to(device)
mask = mask.to(device)
y = y.to(device)

print("\nDati spostati su:", device)

print("x device:", x.device)
print("mask device:", mask.device)
print("y device:", y.device)


# ============================================================
# CONTROLLO VALORI
# ============================================================

print("\nControllo valori:")

print("NaN x:", torch.isnan(x).sum().item())
print("Inf x:", torch.isinf(x).sum().item())

print("NaN mask:", torch.isnan(mask).sum().item())
print("Inf mask:", torch.isinf(mask).sum().item())

print("\nTest completato!")
