import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# 1. CARICAMENTO DATASET
# ============================================================

print("Caricamento dataset...")

df = pd.read_parquet("data/processed/preprocessed.parquet")

print(f"Flussi caricati: {len(df)}")


# ============================================================
# 2. DEFINIZIONE DELLE 48 FEATURE AGGREGATE
# ============================================================

feature_cols = [
    "bidirectional_min_ps",
    "bidirectional_mean_ps",
    "bidirectional_stddev_ps",
    "bidirectional_max_ps",

    "src2dst_min_ps",
    "src2dst_mean_ps",
    "src2dst_stddev_ps",
    "src2dst_max_ps",

    "dst2src_min_ps",
    "dst2src_mean_ps",
    "dst2src_stddev_ps",
    "dst2src_max_ps",

    "bidirectional_min_piat_ms",
    "bidirectional_mean_piat_ms",
    "bidirectional_stddev_piat_ms",
    "bidirectional_max_piat_ms",

    "src2dst_min_piat_ms",
    "src2dst_mean_piat_ms",
    "src2dst_stddev_piat_ms",
    "src2dst_max_piat_ms",

    "dst2src_min_piat_ms",
    "dst2src_mean_piat_ms",
    "dst2src_stddev_piat_ms",
    "dst2src_max_piat_ms",

    "bidirectional_syn_packets",
    "bidirectional_cwr_packets",
    "bidirectional_ece_packets",
    "bidirectional_urg_packets",
    "bidirectional_ack_packets",
    "bidirectional_psh_packets",
    "bidirectional_rst_packets",
    "bidirectional_fin_packets",

    "src2dst_syn_packets",
    "src2dst_cwr_packets",
    "src2dst_ece_packets",
    "src2dst_urg_packets",
    "src2dst_ack_packets",
    "src2dst_psh_packets",
    "src2dst_rst_packets",
    "src2dst_fin_packets",

    "dst2src_syn_packets",
    "dst2src_cwr_packets",
    "dst2src_ece_packets",
    "dst2src_urg_packets",
    "dst2src_ack_packets",
    "dst2src_psh_packets",
    "dst2src_rst_packets",
    "dst2src_fin_packets",
]


print(f"Numero di feature: {len(feature_cols)}")


# ============================================================
# 3. DIVISIONE TRAIN / VALIDATION / TEST
# ============================================================

train = df[df["split"] == "train"]
val = df[df["split"] == "val"]
test = df[df["split"] == "test"]


X_train = train[feature_cols]
X_val = val[feature_cols]
X_test = test[feature_cols]


# ============================================================
# 4. FUNZIONE PER ESEGUIRE LE BASELINE
# ============================================================

def esegui_baseline(nome, modello, X_train, y_train, X_val, y_val,
                    X_test, y_test):

    print("\n" + "=" * 60)
    print(nome)
    print("=" * 60)

    print("Addestramento...")

    modello.fit(X_train, y_train)

    # Validation
    pred_val = modello.predict(X_val)

    f1_val = f1_score(
        y_val,
        pred_val,
        average="weighted"
    )

    print(f"F1 weighted validation: {f1_val:.4f}")

    # Test
    pred_test = modello.predict(X_test)

    f1_test = f1_score(
        y_test,
        pred_test,
        average="weighted"
    )

    print(f"F1 weighted test: {f1_test:.4f}")

    print("\nMatrice di confusione TEST:")

    print(confusion_matrix(y_test, pred_test))

    return f1_val, f1_test


# ============================================================
# 5. K = 2
# ============================================================

print("\n")
print("#" * 60)
print("# CLASSIFICAZIONE K = 2")
print("#" * 60)


y_train_2 = train["label_2"]
y_val_2 = val["label_2"]
y_test_2 = test["label_2"]


# ------------------------------------------------------------
# Regressione logistica
# ------------------------------------------------------------

logistic_2 = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])


esegui_baseline(
    "Logistic Regression - K=2",
    logistic_2,
    X_train,
    y_train_2,
    X_val,
    y_val_2,
    X_test,
    y_test_2
)


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

forest_2 = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


esegui_baseline(
    "Random Forest - K=2",
    forest_2,
    X_train,
    y_train_2,
    X_val,
    y_val_2,
    X_test,
    y_test_2
)


# ============================================================
# 6. K = 20
# ============================================================

print("\n")
print("#" * 60)
print("# CLASSIFICAZIONE K = 20")
print("#" * 60)


y_train_20 = train["label_20"]
y_val_20 = val["label_20"]
y_test_20 = test["label_20"]


# ------------------------------------------------------------
# Regressione logistica
# ------------------------------------------------------------

logistic_20 = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        solver="saga",
        C=1.0,
        random_state=42
    ))
])
esegui_baseline(
    "Logistic Regression - K=20",
    logistic_20,
    X_train,
    y_train_20,
    X_val,
    y_val_20,
    X_test,
    y_test_20
)


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

forest_20 = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


esegui_baseline(
    "Random Forest - K=20",
    forest_20,
    X_train,
    y_train_20,
    X_val,
    y_val_20,
    X_test,
    y_test_20
)


print("\n")
print("Baseline completate!")
