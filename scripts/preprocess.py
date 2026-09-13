import pandas as pd
import numpy as np
import pickle

INPUT = "data/processed/flows.parquet"
OUTPUT = "data/processed/preprocessed.parquet"
STATS_OUTPUT = "data/processed/normalization_stats.pkl"

print("Caricamento dataset...")
df = pd.read_parquet(INPUT)

print(f"Flussi caricati: {len(df)}")


# ============================================================
# 1. CONVERSIONE DELLE SEQUENZE SPLT
# ============================================================

def preprocess_sequence(row):
    import ast

    directions = np.array(
        ast.literal_eval(row["splt_direction"])
        if isinstance(row["splt_direction"], str)
        else row["splt_direction"],
        dtype=float
    )

    sizes = np.array(
        ast.literal_eval(row["splt_ps"])
        if isinstance(row["splt_ps"], str)
        else row["splt_ps"],
        dtype=float
    )

    piats = np.array(
        ast.literal_eval(row["splt_piat_ms"])
        if isinstance(row["splt_piat_ms"], str)
        else row["splt_piat_ms"],
        dtype=float
    )

    # Nei dati NFStream -1 indica elemento assente/padding
    mask = directions != -1

    # Direzione: -1 / +1
    directions = np.where(mask, directions, 0)

    # Dimensioni e inter-arrivi:
    # i valori di padding vengono temporaneamente messi a 0
    sizes = np.where(mask, sizes, 0)
    piats = np.where(mask, piats, 0)

    # log1p sui valori reali
    sizes = np.log1p(np.maximum(sizes, 0))
    piats = np.log1p(np.maximum(piats, 0))

    return directions, sizes, piats, mask


print("Preprocessing delle sequenze...")

risultati = df.apply(preprocess_sequence, axis=1)

df["direction"] = risultati.apply(lambda x: x[0])
df["packet_size"] = risultati.apply(lambda x: x[1])
df["piat"] = risultati.apply(lambda x: x[2])
df["mask"] = risultati.apply(lambda x: x[3])


# ============================================================
# 2. CALCOLO DELLE STATISTICHE SOLO SUL TRAIN
# ============================================================

print("Calcolo media e deviazione standard sul TRAIN...")

train_mask = df["split"] == "train"

train_sizes = np.concatenate(
    df.loc[train_mask, "packet_size"].values
)

train_piats = np.concatenate(
    df.loc[train_mask, "piat"].values
)

train_valid_sizes = train_sizes[
    np.isfinite(train_sizes)
]

train_valid_piats = train_piats[
    np.isfinite(train_piats)
]

media_size = train_valid_sizes.mean()
std_size = train_valid_sizes.std()

media_piat = train_valid_piats.mean()
std_piat = train_valid_piats.std()

print(f"Media packet size: {media_size}")
print(f"Std packet size: {std_size}")

print(f"Media PIAT: {media_piat}")
print(f"Std PIAT: {std_piat}")


# ============================================================
# 3. STANDARDIZZAZIONE
# ============================================================

def standardize(row):
    directions = row["direction"].copy()
    sizes = row["packet_size"].copy()
    piats = row["piat"].copy()
    mask = row["mask"].copy()

    sizes = (sizes - media_size) / std_size
    piats = (piats - media_piat) / std_piat

    # Il padding non deve diventare un valore reale
    sizes[~mask] = 0
    piats[~mask] = 0

    return directions, sizes, piats


print("Standardizzazione...")

standardizzati = df.apply(standardize, axis=1)

df["direction"] = standardizzati.apply(lambda x: x[0])
df["packet_size"] = standardizzati.apply(lambda x: x[1])
df["piat"] = standardizzati.apply(lambda x: x[2])


# ============================================================
# 4. SALVATAGGIO DELLE STATISTICHE
# ============================================================

stats = {
    "media_size": media_size,
    "std_size": std_size,
    "media_piat": media_piat,
    "std_piat": std_piat,
}

with open(STATS_OUTPUT, "wb") as f:
    pickle.dump(stats, f)


# ============================================================
# 5. SALVATAGGIO DATASET
# ============================================================

print("Salvataggio dataset preprocessato...")

df.to_parquet(
    OUTPUT,
    index=False
)

print()
print("Preprocessing completato!")
print(f"Dataset salvato in: {OUTPUT}")
print(f"Statistiche salvate in: {STATS_OUTPUT}")

print()
print("Colonne create:")
print("- direction")
print("- packet_size")
print("- piat")
print("- mask")
