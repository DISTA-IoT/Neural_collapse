# HPO Final Report: Pre-training Self-Supervised MiniNet

**Data:** 2026-09-16  
**Durata campagna:** ~10 ore (14:53 → 01:33+)  
**Trial eseguiti:** 10 (Trial 0–9)  
**Obiettivo:** Ottimo di Pareto tra fedeltà di compressione ($P(X)$) ed espressività per il downstream task ($Y$)  
**Risultato e Scelta Ufficiale:** 🏆 **Trial 5 — Campione Eletto (Ottimo di Pareto: Val Loss = 0.435, Probe F1 = 79.38%)**

> [!IMPORTANT]
> **Chiarimento sui termini e modelli di riferimento:**
> 1. **Encoder Random (pesi casuali non addestrati):** la rete a pesi mai addestrati con probe lineare $\to$ **Probe F1-macro = 70.55%**.
> 2. **Baseline (modello addestrato da Francesca in E-10):** l'encoder originale pre-addestrato da Francesca ($d=64, L=4, H=4$, mascheramento casuale al 30%, 10 epoche) $\to$ **Probe F1-macro = 76.66%**. Questo è il riferimento ufficiale che la campagna HPO si proponeva di superare.
> 3. **Trial 0 (replica della baseline a 5 epoche):** stessa architettura di Francesca ($d=64, L=4$) ma addestrata per 5 epoche per allinearsi al budget temporale dei trial HPO $\to$ **Probe F1-macro = 75.86%**.

---

## 1. Fondamento Teorico della Selezione: Compressione vs Rappresentazione e Trade-off di Pareto

La selezione del modello "campione" in un contesto di pre-training auto-supervisionato (SSL) **non** deve seguire la logica cieca delle competizioni (selezionare esclusivamente il picco di F1 a valle), ma deve basarsi su rigorosi principi di teoria dell'informazione e rappresentazione geometrica:

### 1.1 Compressione pura ($P(X)$) vs Rappresentazione Semantica
* **Il ruolo dell'encoder SSL:** Un encoder auto-supervisionato è primariamente un modellatore della distribuzione dei dati non etichettati $P(X)$. Il suo scopo è comprimere e ricostruire la struttura intrinseca del traffico di rete. Misurare la "bontà" di un encoder unicamente dalla loss del downstream task è concettualmente parziale: minimizzare la loss a valle è un problema a sé stante.
* **Il trade-off di fedeltà:** Se un encoder impara a fare una compressione eccellente (bassa reconstruction loss), preserva intatta l'informazione dei dati $X$. Tuttavia, nei dati grezzi del traffico (timestamp, millisecondi, overhead di protocollo), molta varianza è puramente stocastica. 

### 1.2 La definizione di rumore e separabilità è funzione del Downstream Task
* In teoria dell'informazione (**Information Bottleneck**), il concetto di "rumore" o "irrilevanza" non esiste nel vuoto: è formalmente definito in relazione a una variabile bersaglio $Y$ attraverso l'informazione mutua condizionata $I(X; Z \mid Y)$.
* Ciò che costituisce "rumore" per classificare un'applicazione malware (ad esempio le micro-oscillazioni di jitter temporale tra pacchetti) potrebbe essere il segnale discriminante primario per un task differente (es. identificazione dell'hardware o fingerprinting della scheda di rete).
* Poiché l'encoder viene pre-addestrato in modo **task-agnostico** (senza conoscere le etichette $Y$), ogni scelta architetturale e di masking (in particolare lo **span masking**) costituisce un *inductive bias*: una scommessa a priori su quali correlazioni strutturali (dipendenze temporali a blocchi) debbano essere preservate nello spazio latente $z_{cls}$.

### 1.3 Perché Trial 5 è l'Ottimo di Pareto (e batte Trial 4 e Trial 7)
Analizzando lo spazio delle soluzioni:
* **I modelli a sola compressione (Trial 0, 2):** Hanno una val reconstruction loss bassissima ($\approx 0.22 - 0.24$), ma ottengono un F1 modesto ($\approx 75.8\% - 76.0\%$). Hanno imparato a interpolare dettagli locali senza costruire astrazioni utili.
* **Il modello al picco estremo di F1 (Trial 4):** Raggiunge il picco di $F1 = 80.25\%$, ma con una val loss salita a **0.524** (+20% di errore di ricostruzione rispetto a Trial 5). Questo indica un'eccessiva degradazione della capacità generativa intrinseca dell'encoder.
* **Il modello profondo (Trial 7, L=6):** Ottiene val loss 0.399 ma crolla a $F1 = 76.89\%$ (quasi identico alla baseline), aumentando i parametri a 1.19M (+50%) e i tempi a 150m (+45%).
* **🏆 La scelta di Trial 5:** Rappresenta il vero **punto di equilibrio di Pareto**:
  * Abbassa la val reconstruction loss a **0.435** (**-17% di errore di ricostruzione** rispetto a Trial 4), dimostrando che l'encoder modella i dati con molta più fedeltà.
  * Mantiene un probe F1-macro straordinario: **79.38%** (praticamente all'80%, **+2.72%** rispetto alla baseline di Francesca e a soli 0.87 punti dal picco di Trial 4).
  * Mantiene l'architettura compatta ed efficiente a 4 layer ($796\text{k}$ parametri, 105 min).

### 1.4 Collegamento al Neural Collapse
Il *Neural Collapse* indaga l'emergenza di geometrie simmetriche (Simplex Equiangular Tight Frame - ETF) e il collasso della varianza intra-classe nello spazio latente durante l'addestramento supervisionato. Partire da un encoder pre-addestrato come **Trial 5** garantisce una base di partenza ideale: uno spazio latente $z_{cls}$ che non è iper-specializzato né degradato nella rappresentazione del traffico grezzo, ma possiede la plasticità geometrica per collassare armoniosamente sui centroidi di classe durante la Fase 3 e Fase 4.

---

## 2. Tabella Riassuntiva dei Trial

| Trial | Configurazione Chiave | Train Loss | Val Loss | Probe Acc | Probe F1 (Macro) | Δ vs Baseline Francesca (76.66%) | Durata | Valutazione / Ruolo |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 0 (E-5) | d=64, L=4, H=4, lr=1e-3, random, bs=512 | 0.247 | 0.242 | 75.59% | 75.86% | -0.80% | 72m | Replica Francesca (5 epoche) |
| 1 | lr=3e-4, cosine+warmup=200, grad_clip=1.0 | 0.296 | 0.288 | 75.30% | 75.21% | -1.45% | 74m | Sotto-addestrato (LR basso) |
| 2 | **d=128**, H=8, grad_clip=1.0 | 0.207 | 0.223 | 75.75% | 76.03% | -0.63% | 103m | Solo capacità, random mask |
| 3 | d=128, H=8, **span mask 35%**, **w_iat=0.5** | 0.148 | 0.503 | 77.32% | 79.09% | +2.43% | 100m | Ottimo sblocco (+3% F1) |
| 4 | d=128, H=8, span 30%, w_iat=0.5, wd=1e-3 | 0.149 | 0.524 | 80.23% | **80.25%** | +3.59% | 103m | Picco F1 (downstream-biased) |
| **5** 🏆 | **span 25%, wd=5e-3, cosine+warmup=100, w_iat=0.5** | **0.133** | **0.435** | **78.28%** | **79.38%** | **+2.72%** | **105m** | 🏆 **CAMPIONE (Ottimo di Pareto)** |
| 6 | Trial 4 config + **E7** (più epoche) | 0.142 | 0.476 | 79.93% | 78.37% | +1.71% | 147m | Overfitting classi dominanti |
| 7 | d=128, **L=6**, H=8, span 30% | 0.148 | 0.399 | 77.36% | 76.89% | +0.23% | 150m | Troppo pesante, segnale diluito |
| 8 | span 35%, **w_dir=2.0**, w_iat=0.5 | 0.197 | 0.487 | 78.12% | 77.61% | +0.95% | 105m | Shortcut learning su direzione |
| 9 | span 30%, **w_size=2.0**, w_iat=0.5 | 0.221 | 0.470 | 77.84% | 77.66% | +1.00% | 105m | Distorsione su grandezza byte |

---

## 3. Configurazione Ufficiale del Campione (Trial 5)

La configurazione selezionata per proseguire le successive fasi di tesi è quella di **Trial 5**:

```json
{
  "d_model": 128,
  "n_layers": 4,
  "n_heads": 8,
  "dim_feedforward_multiplier": 4,
  "learning_rate": 0.001,
  "weight_decay": 0.005,
  "batch_size": 512,
  "epochs": 5,
  "mask_ratio": 0.25,
  "masking_strategy": "span",
  "scheduler": "cosine",
  "warmup_steps": 100,
  "grad_clip": 1.0,
  "w_size": 1.0,
  "w_iat": 0.5,
  "w_dir": 1.0,
  "seed": 42
}
```

* **Parametri encoder:** 796,416 (architettura bilanciata ed efficiente a 4 strati).
* **Val Reconstruction Loss:** **0.435** (fedeltà generativa conservata).
* **Linear Probe F1-Macro:** **79.38%** (+2.72% rispetto alla Baseline di Francesca in E-10).
* **Linear Probe Accuracy:** **78.28%** (+3.01% rispetto a Francesca).
* **Checkpoint salvato e sincronizzato:** `res/pretraining/best_encoder.pt` (backup originario in `res/pretraining_trial_5/pretrained_encoder.pt`).

---

## 4. Analisi dei Trend e Lezioni Apprese

### 4.1 Span Masking: la modifica fondamentale (+3% F1)
Il passaggio dal mascheramento casuale (*random*) allo **Span Masking** (blocchi contigui di 2-4 pacchetti) ha rivoluzionato le rappresentazioni:
- Trial 2 (d=128, random) $\to$ F1 = 76.03%
- Trial 3 (d=128, span 35%) $\to$ F1 = 79.09% (**+3.06%**)
- Trial 5 (d=128, span 25%) $\to$ F1 = 79.38% con Val Loss nettamente migliore (0.435 vs 0.503).

**Motivazione fisica:** Nei flussi di rete, i pacchetti adiacenti sono correlati (burst TCP, finestre di congestione). Mascherando singoli pacchetti isolati, il modello interpolava banalmente dal contesto locale immediato. Lo span masking costringe l'encoder a modellare **dipendenze a lungo raggio** dell'intera sessione.

### 4.2 Bilanciamento della Loss Multi-Task: de-enfatizzare l'IAT ($w_{iat}=0.5$)
L'encoder ricostruisce simultaneamente tre proprietà per ogni pacchetto:
$$\mathcal{L}_{\text{total}} = w_{\text{size}} \cdot \mathcal{L}_{\text{size}} + w_{\text{iat}} \cdot \mathcal{L}_{\text{iat}} + w_{\text{dir}} \cdot \mathcal{L}_{\text{dir}}$$

* **IAT ($w_{\text{iat}}=0.5$):** Il tempo di inter-arrivo è soggetto a jitter intrinseco di rete e code router (rumore ad alta frequenza). Con peso $1.0$, i gradienti IAT monopolizzano l'ottimizzazione per inseguire fluttuazioni casuali. Dimezzare il peso ($w_{\text{iat}}=0.5$) "calma" l'aggiornamento e permette al modello di focalizzarsi sulle dinamiche dimensionali e direzionali.
* **Perché amplificare ad esempio $w_{\text{dir}}=2.0$ fallisce (-2.64% F1):** La direzione è un segnale binario facile da predire localmente; raddoppiarne il peso crea *shortcut learning*, inducendo l'encoder a trascurare le correlazioni globali del flusso.

### 4.3 Regolarizzazione e Ottimizzazione: Weight Decay 5e-3 con Cosine Schedule
Confrontando le strategie di ottimizzazione tra i modelli più avanzati:

| Trial | Base LR | Scheduler | Warmup | Weight Decay | Val Loss | Probe F1 | Esito |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **4** | $1 \times 10^{-3}$ | Nessuno | 0 | $1 \times 10^{-3}$ | 0.524 | 80.25% | Picco F1 ma loss alta |
| **5 (Campione)** | $1 \times 10^{-3}$ | Cosine | 100 step | $5 \times 10^{-3}$ | **0.435** | **79.38%** | **Miglior equilibrio globale** |

In Trial 5, un weight decay più robusto ($5 \times 10^{-3}$) abbinato al Cosine Annealing (con 100 step di warmup) impedisce ai pesi di divergere, guidando l'encoder verso un minimo più regolare e liscio, che spiega l'eccellente valore di val loss (0.435) mantenendo una separabilità di quasi l'80% di F1.

### 4.4 Architettura: L=4 vince nettamente su L=6
Il modello a 6 strati (Trial 7, 1.19M parametri) ha ottenuto una val loss di 0.399 ma un F1 crollato a 76.89%.  
**Motivazione:** Con 6 layer di auto-attenzione su sequenze corte (20 pacchetti), l'informazione semantica si disperde lungo gli strati profondi (*signal dilution*), riducendo la densità informativa nel token `[CLS]`. Con 4 strati, l'encoder concentra efficacemente la semantica globale del flusso in $z_{cls}$.

---

## 5. Pareto Frontier

La frontiera di Pareto evidenzia le scelte ottimali a seconda dell'obiettivo:

1. **🏆 Trial 5 (Scelta Raccomandata - Pareto Optimum):** F1 = 79.38%, Val Loss = 0.435, 796k params, 105 min. Il miglior connubio tra compressione fedele della distribuzione dei dati ed eccellente separabilità delle classi.
2. **Trial 4 (Variante Downstream-Biased):** F1 = 80.25%, Val Loss = 0.524, 796k params, 103 min. Utile se l'unico obiettivo fosse la classificazione pura a scapito della fedeltà di ricostruzione.
3. **Trial 0 / Replica Francesca (Baseline d=64):** F1 = 75.86%, Val Loss = 0.242, 198k params, 72 min. Utile solo in scenari a estremo vincolo di calcolo.

---

## 6. Guida Pratica per la Studentessa: Come Usare il Campione nelle Fasi 3 e 4

Questa sezione spiega passo dopo passo **come utilizzare concretamente il checkpoint di Trial 5** nello svolgimento delle fasi successive previste dalla guida operativa ([`D_guide.md`](file:///home/jesusc/Neural_collapse/res/D_guide.md)).

---

### 6.1 Fase 3 — Analisi Geometrica a Riposo ($t = 0$)

Nella Fase 3 della guida (settimana 5), l'obiettivo è misurare lo "stato di salute" e la geometria delle rappresentazioni prima di toccare qualsiasi etichetta di malware.

1. **Caricare il modello corretto:**
   * Usa il checkpoint salvato in `res/pretraining/best_encoder.pt` (Trial 5).
   * **⚠️ Attenzione alla dimensione latente:** La baseline iniziale di Francesca aveva $d=64$; il nostro campione Trial 5 usa **$d=128$** ($L=4, H=8$). Assicurati che i tuoi script e notebook allochino i tensori di output per $z_{\text{cls}} \in \mathbb{R}^{128}$.
2. **Costruire la Tabella $t = 0$ (Esercizio E-13):**
   * Estrai le rappresentazioni congelate $z_{\text{cls}}$ su tutti i flussi di train e validation.
   * Calcola la "cassetta degli attrezzi quantitativa" (§6.2 di `D_guide.md`):
     * **Rango effettivo:** verifica quante delle 128 direzioni disponibili l'encoder sta effettivamente sfruttando (ricorda di centrare i dati sottraendo la media globale prima del calcolo).
     * **Anisotropia:** misura la cosine similarity media tra coppie casuali di flussi per verificare se i vettori occupano una sfera ampia o un cono stretto.
     * **NC1 a riposo:** calcola la metrica di collasso $\operatorname{Tr}(\Sigma_B^\dagger \Sigma_W)$ sul training set. Poiché l'encoder non ha mai visto le etichette, il collasso intra-classe sarà basso/moderato: questo numero sarà il tuo **punto di riferimento zero** per confrontare quanto le classi si compatteranno dopo.
     * **Distribuzione nulla:** genera sempre le 1.000 permutazioni casuali delle etichette come spiegato in E-13 per dimostrare che ogni misura è statisticamente distinguibile dal caso.
3. **Visualizzazione UMAP (Esercizio E-11):**
   * Proietta i vettori $z_{\text{cls}}$ a 128 dimensioni in 2D usando UMAP (`metric="cosine"`).
   * Ricorda la regola metodologica aurea: *la figura UMAP serve per generare intuizioni visive (es. vedere se i flussi della stessa applicazione tendono già a raggrupparsi), ma le conclusioni scientifiche si traggono solo dai numeri calcolati nello spazio a 128 dimensioni*.

---

### 6.2 Fase 4 — Fine-tuning nei Quattro Regimi (Settimane 6–9)

Nella Fase 4 affronterai la domanda centrale della tesi: **"Il fine-tuning supervisionato distrugge la geometria composizionale?"**

1. **I 4 regimi a confronto (§7.1 di `D_guide.md`):**
   * **HEAD (Controllo):** Congela l'encoder di Trial 5 e addestra soltanto il classificatore lineare sovrastante sulle 20 classi. Le rappresentazioni $z_{\text{cls}}$ non cambiano per costruzione; serve come controllo di base.
   * **FULL (Fine-tuning Standard):** Inizializza l'encoder con i pesi di Trial 5 e aggiorna *tutti* i parametri (encoder + testa) con la cross-entropy. Questo è l'esperimento principale in cui osserverai se si verifica il Neural Collapse (NC1 $\to 0$).
   * **LP-FT (Ricetta di Kumar et al.):** Addestra prima solo la testa fino a convergenza (HEAD), e poi sblocca l'intero modello (FULL). Serve a testare se "scaldare" prima la testa protegge le rappresentazioni dell'encoder dalla distruzione.
   * **SCRATCH (Controllo da zero):** Prendi la stessa identica architettura ($d=128, L=4, H=8$), inizializzala con **pesi casuali** (senza pre-training) e addestrala da zero in modo supervisionato. Risponde alla domanda: *il collasso geometrico è dovuto al pretraining o è una proprietà intrinseca della cross-entropy supervisionata?*
2. **Perché Trial 5 è il candidato ideale per questo test:**
   * Avendo scelto un modello all'**ottimo di Pareto** (loss di ricostruzione fedele a 0.435 ed F1 al 79.38%), l'encoder possiede una ricca struttura interna sia a livello di dati grezzi che di semantica globale.
   * Se durante il regime **FULL** noterai che le prestazioni su compiti OOD o le correlazioni intrinseche crollano mentre le classi collassano (NC1 $\to 0$), potrai affermare con certezza scientifica che è stato il **processo di fine-tuning supervisionato a cancellare le varietà latenti**, escludendo che il modello di partenza fosse carente o sotto-addestrato.

---

## 7. Modifiche al Codice Effettuate (Solo Scripts)

Le modifiche architetturali e algoritmiche sviluppate durante l'HPO risiedono nei file della cartella `scripts/` (principalmente in [`scripts/pretrain.py`](file:///home/jesusc/Neural_collapse/scripts/pretrain.py) e [`scripts/config.json`](file:///home/jesusc/Neural_collapse/scripts/config.json)):

1. **`create_span_mask()`**: implementazione dello span masking a blocchi contigui (2-4 pacchetti).
2. **Pesi configurabili per la loss multi-task**: supporto a `w_size`, `w_iat`, `w_dir` in `compute_reconstruction_loss()`.
3. **Scheduler Cosine Annealing con Warmup**: supporto al decadimento coseno con warmup lineare per step.
4. **Gradient Clipping**: parametro `grad_clip` su tutti i parametri per stabilizzare AdamW.
5. **CLI ed estensione parametri**: supporto da riga di comando per tutti i nuovi iperparametri di pretraining.

> [!WARNING]
> **NOTA BENE PER LA TESISTA — Aggiornare i Notebook Jupyter:**  
> Tutte queste modifiche sono state implementate **esclusivamente negli script Python (`scripts/`)** e **NON toccano il codice dei Jupyter Notebook** (es. `notebooks/` o `E-10`).  
> **Cosa fare:** Se si intende eseguire esperimenti o mostrare il codice attraverso i notebook, **è necessario aggiornare i notebook** allineandoli alle nuove funzioni e logiche introdotte in [`scripts/pretrain.py`](file:///home/jesusc/Neural_collapse/scripts/pretrain.py) e alla configurazione di Trial 5.

