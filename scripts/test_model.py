import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DATASET = "data/processed/preprocessed.parquet"

BATCH_SIZE = 64
EMBEDDING_DIM = 64
NUM_CLASSES = 20


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)


class FlussiDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(drop=True)

        self.classi = sorted(self.df["label_20"].unique())

        self.mappa_classi = {
            classe: i
            for i, classe in enumerate(self.classi)
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, indice):

        riga = self.df.iloc[indice]

        direction = np.asarray(
            riga["direction"],
            dtype=np.float32
        )

        packet_size = np.asarray(
            riga["packet_size"],
            dtype=np.float32
        )

        piat = np.asarray(
            riga["piat"],
            dtype=np.float32
        )

        mask = np.asarray(
            riga["mask"],
            dtype=np.float32
        )

        sequenza = np.stack(
            [direction, packet_size, piat],
            axis=1
        )

        x = torch.tensor(
            sequenza,
            dtype=torch.float32
        )

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        )

        y = torch.tensor(
            self.mappa_classi[riga["label_20"]],
            dtype=torch.long
        )

        return x, mask, y


class NeuralNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, EMBEDDING_DIM)
        )

        self.classificatore = nn.Linear(
            EMBEDDING_DIM,
            NUM_CLASSES
        )

    def forward(self, x, mask):

        h_pacchetti = self.encoder(x)

        mask = mask.unsqueeze(-1)

        h_pacchetti = h_pacchetti * mask

        somma = h_pacchetti.sum(dim=1)

        numero_pacchetti = mask.sum(dim=1).clamp(min=1)

        h = somma / numero_pacchetti

        logits = self.classificatore(h)

        return logits, h


print("\nCaricamento dataset...")

df = pd.read_parquet(DATASET)

df_train = df[df["split"] == "train"].copy()

dataset = FlussiDataset(df_train)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


model = NeuralNetwork().to(device)

print("\nModello creato:")
print(model)


loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)


x, mask, y = next(iter(loader))

x = x.to(device)
mask = mask.to(device)
y = y.to(device)


print("\nForward...")

logits, h = model(x, mask)

print("x:", x.shape)
print("h:", h.shape)
print("logits:", logits.shape)
print("y:", y.shape)


loss = loss_fn(logits, y)

print("\nLoss iniziale:", loss.item())


optimizer.zero_grad()

loss.backward()

print("\nBackward completato!")


print("\nControllo gradienti:")

for nome, parametro in model.named_parameters():

    if parametro.grad is None:
        print(nome, "-> gradiente NONE")
    else:
        print(
            nome,
            "->",
            parametro.grad.shape,
            "norma:",
            parametro.grad.norm().item()
        )


optimizer.step()

print("\nAggiornamento dei pesi completato!")

print("\nTEST MODELLO COMPLETATO!")
