import pandas as pd
from sklearn.model_selection import train_test_split

# Seed fisso per rendere la divisione riproducibile
SEED = 42

# Leggi il dataset
df = pd.read_parquet("data/processed/flows.parquet")

print(f"Flussi totali: {len(df)}")

# --------------------------------------------------
# 1. Creazione del PROBE SET
# --------------------------------------------------

# 5.000 flussi stratificati sulle 20 classi
probe, restante = train_test_split(
    df,
    test_size=len(df) - 5000,
    stratify=df["label_20"],
    random_state=SEED
)

# --------------------------------------------------
# 2. Divisione TRAIN / VALIDATION / TEST
# --------------------------------------------------

train, temporaneo = train_test_split(
    restante,
    test_size=0.30,
    stratify=restante["label_20"],
    random_state=SEED
)

val, test = train_test_split(
    temporaneo,
    test_size=0.50,
    stratify=temporaneo["label_20"],
    random_state=SEED
)

# --------------------------------------------------
# 3. Aggiungiamo la colonna SPLIT
# --------------------------------------------------

train = train.copy()
val = val.copy()
test = test.copy()
probe = probe.copy()

train["split"] = "train"
val["split"] = "val"
test["split"] = "test"
probe["split"] = "probe"

# --------------------------------------------------
# 4. Ricomponiamo il dataset
# --------------------------------------------------

df_finale = pd.concat(
    [train, val, test, probe],
    ignore_index=True
)

# --------------------------------------------------
# 5. Salviamo il nuovo dataset
# --------------------------------------------------

output = "data/processed/flows.parquet"

df_finale.to_parquet(output, index=False)

# --------------------------------------------------
# 6. Controlli
# --------------------------------------------------

print("\nPartizioni:")
print(df_finale["split"].value_counts())

print("\nDimensione finale:")
print(df_finale.shape)

print("\nConteggio per classe e partizione:")
print(
    pd.crosstab(
        df_finale["label_20"],
        df_finale["split"]
    )
)

print("\nDataset salvato in:")
print(output)
