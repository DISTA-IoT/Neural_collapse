# E-7bis — Un compito troppo facile: comportamento o artefatto di sessione?

Addendum a §4.5 della guida. Da fare subito dopo l'E-7, prima di lanciare la griglia del §7.4.

## Il problema

La Random Forest sulle 48 statistiche, per *K* = 2 (malware/benigno), ottiene F1 = 1.0 sul test — non su una manciata di esempi, ma sui 56255 flussi del test set. È esattamente il caso descritto al §4.1: *"se il modello arriva a F1 = 0,999 in cinquanta passi, non c'è nessuna curva da studiare"*. Un compito che nessun classificatore sbaglia mai non lascia niente da misurare più avanti: niente dinamica di apprendimento, niente probe che si differenziano, niente da confrontare fra regimi.

Per *K* = 20, invece, RF si ferma a F1 = 0,855 e la regressione logistica a 0,466: lì il compito ha ancora attrito. Il problema è specifico di *K* = 2, non del disegno nel suo complesso.

## Perché potrebbe essere un artefatto, non apprendimento

USTC-TFC2016 è organizzato come 20 file pcap, uno per classe: ogni famiglia di malware e ogni applicazione benigna è registrata in un'unica sessione di cattura. Lo si vede nella pivot table già prodotta in E-6 (`label_20` × `label_2`): ogni riga ha zero flussi nell'altra colonna — ovvio per costruzione, non una scoperta.

Il punto è un altro: se ogni classe coincide con un'unica sessione di cattura, allora anche statistiche del tutto innocenti — durata totale del flusso, byte totali, numero di pacchetti — possono correlare fortissimamente con *quale sessione* ha generato quel flusso, indipendentemente da cosa quel traffico stia effettivamente facendo. È la stessa famiglia di problema del difetto MAC/timestamp che la guida segnala esplicitamente per questo dataset (§4.1): un classificatore che sembra aver imparato "malware" potrebbe aver imparato "file pcap numero 7".

## Come si distingue una spiegazione dall'altra

Non lo si vede a occhio dal solo F1: serve chiedere alla Random Forest *quali* feature sta usando per decidere, e conviene farlo in due modi diversi perché hanno trappole diverse.

**Importanza da impurità.** Quanto ogni feature riduce l'impurità nei nodi degli alberi — il numero che si ottiene "gratis" dal modello già addestrato. Ha un difetto noto: sovrastima sistematicamente le feature continue ad alta cardinalità, che qui sono esattamente le feature "di scala grezza" (durata, byte totali, conteggio pacchetti) — le più sospette di essere impronte di sessione. Va letta con un certo scetticismo se sono proprio quelle a dominare.

**Permutation importance.** Si mescola a caso una feature alla volta, sul validation set, e si misura di quanto scende l'F1. Risponde a una domanda più onesta — "il modello dipende davvero da questa feature per decidere?" — senza lo stesso bias verso le feature continue.

Se entrambe le classifiche mettono in cima feature di scala grezza, la spiegazione più probabile è l'artefatto di sessione. Se invece dominano feature di "forma" — rapporti fra flag TCP, statistiche di dimensione o interarrivo, che non dipendono dalla scala assoluta del flusso — la separabilità è più plausibilmente comportamentale, anche se resta comunque sospettosamente alta.

## E-7bis

Subito dopo la cella della Random Forest per *K* = 2 (quella con `rf.fit(X_train, y_train)`), aggiunga una nuova cella. Le do l'intestazione e i nomi delle feature già pronti — sbagliare l'ordine qui è un errore silenzioso, non darebbe nessun errore ma le etichette non corrisponderebbero alle colonne giuste — il resto lo completa Lei.

```python
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

# Stesso ordine ESATTO con cui ha impilato le colonne in stats_48 (E-7).
# Se non coincide con il Suo codice, lo corregga qui prima di procedere.
feature_names = [
    "bidirectional_duration_ms", "bidirectional_packets", "bidirectional_bytes",
    "src2dst_duration_ms", "src2dst_packets", "src2dst_bytes",
    "dst2src_duration_ms", "dst2src_packets", "dst2src_bytes",
    "bidirectional_min_ps", "bidirectional_mean_ps", "bidirectional_stddev_ps", "bidirectional_max_ps",
    "src2dst_min_ps", "src2dst_mean_ps", "src2dst_stddev_ps", "src2dst_max_ps",
    "dst2src_min_ps", "dst2src_mean_ps", "dst2src_stddev_ps", "dst2src_max_ps",
    "bidirectional_min_piat_ms", "bidirectional_mean_piat_ms", "bidirectional_stddev_piat_ms", "bidirectional_max_piat_ms",
    "src2dst_min_piat_ms", "src2dst_mean_piat_ms", "src2dst_stddev_piat_ms", "src2dst_max_piat_ms",
    "dst2src_min_piat_ms", "dst2src_mean_piat_ms", "dst2src_stddev_piat_ms", "dst2src_max_piat_ms",
    "bidirectional_syn_packets", "bidirectional_cwr_packets", "bidirectional_ece_packets", "bidirectional_urg_packets",
    "bidirectional_ack_packets", "bidirectional_psh_packets", "bidirectional_rst_packets", "bidirectional_fin_packets",
    "src2dst_syn_packets", "src2dst_ack_packets", "src2dst_psh_packets", "src2dst_rst_packets",
    "dst2src_syn_packets", "dst2src_ack_packets", "dst2src_psh_packets",
]
assert len(feature_names) == 48 == X_train.shape[1]

# 1) Importanza da impurità: la ottiene direttamente da `rf`, non richiede
#    ripetizioni. La ordini in un pandas.Series indicizzato da feature_names,
#    decrescente.
imp_impurity = ...  # TODO

# 2) Permutation importance su (X_val, y_val): guardi la firma di
#    sklearn.inspection.permutation_importance. Le serve scoring="f1_weighted",
#    un n_repeats ragionevole (8-10) e un random_state fissato. Il risultato ha
#    un campo .importances_mean: lo metta anche questo in un Series ordinato.
imp_perm = ...  # TODO

print("=== TOP 15 — importanza da impurità (K=2) ===")
print(imp_impurity.head(15))
print()
print("=== TOP 15 — permutation importance su validation (K=2) ===")
print(imp_perm.head(15))

# 3) Due gruppi di feature, definiti da Lei: "scala grezza" (durata, byte
#    totali, conteggi di pacchetti — dipendono da quanto è durata/quanto ha
#    prodotto quella sessione di cattura) contro "forma" (rapporti fra flag
#    TCP, statistiche di dimensione e interarrivo — non dipendono dalla scala
#    assoluta del flusso). Ogni nome in feature_names deve finire in uno dei
#    due gruppi, e in uno solo.
raw_scale = [...]   # TODO
shape_like = [...]  # TODO
assert set(raw_scale) | set(shape_like) == set(feature_names)
assert set(raw_scale) & set(shape_like) == set()

print()
print("Somma importanza (impurità) — feature di scala grezza:", imp_impurity[raw_scale].sum())
print("Somma importanza (impurità) — feature di forma:       ", imp_impurity[shape_like].sum())
```

**Controllo.** La somma di tutte le importanze da impurità deve fare 1 (sono normalizzate); se dopo aver diviso in due gruppi le due somme non tornano a 1, ha sbagliato la partizione delle feature o ne ha dimenticata qualcuna — è per questo che i due `assert` sopra ci sono già: non li tolga.

> **Q-7bis.** Se la permutation importance e l'importanza da impurità danno classifiche diverse, quale delle due preferisce come evidenza, e perché? (Non è una domanda retorica: la risposta dipende da cosa ciascuna delle due misure sta davvero misurando — lo riveda sopra.)

## Cosa fare con il risultato

- Se dominano le feature di scala grezza in entrambe le classifiche: niente, me lo dica, poi vediamo :)
- Se dominano le feature di forma: tenga *K* = 2 così com'è e proceda.
- In entrambi i casi, questo controllo — e la sua giustificazione — entra nel capitolo di metodologia. È precisamente il tipo di verifica che Arp et al. (§12) chiedono di fare prima di fidarsi di un risultato "troppo bello".

**Criterio di uscita:** le due classifiche di importanza (impurità e permutation) per *K* = 2, la somma per gruppo, e una riga scritta — non solo pensata — su quale delle due spiegazioni (comportamento vs. artefatto di sessione) i numeri sostengono.
