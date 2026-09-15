# Come sta andando il progetto? (Stato di salute e risultati spiegati semplice)

Questo documento spiega punto per punto **cosa abbiamo fatto finora, come sono andati gli esperimenti e perché quei numeri contano**, senza dare nulla per scontato e traducendo il gergo tecnico in concetti concreti.

---

## 1. La situazione in due parole

Fino a questo momento abbiamo completato le **Fasi 0, 1 e 2** della guida operativa ([D_guide.md](file:///home/jesusc/Neural_collapse/res/D_guide.md)), coprendo gli esercizi da **E-1 a E-10**. 

In estrema sintesi:
1. **La parte preparatoria e i dati funzionano:** Abbiamo ripulito e preparato oltre mezzo milione di flussi di rete (562.549) senza toccare i dati di test e senza imbrogliare con statistiche globali.
2. **Il pericolo del "compito troppo facile" è stato scongiurato (E-7bis):** Avevamo il timore che il modello classificasse il traffico distinguendo solo la durata delle registrazioni pcap (un trucco da quattro soldi); abbiamo dimostrato che invece riconosce il *comportamento reale* del traffico (forme dei pacchetti, flag TCP).
3. **L'architettura del modello è solida (E-8, E-9):** Il nostro mini-Transformer gestisce correttamente il padding (i pacchetti fittizi non ingannano il modello) ed estrae in modo deterministico le rappresentazioni interne.
4. **Il pre-addestramento ha funzionato davvero (E-10):** Il modello ha imparato da solo a capire la struttura del traffico prima ancora di vedere le etichette di malware.

Vediamo ora nel dettaglio i punti chiave che hanno portato a queste conclusioni.

---

## 2. Il "giallo" del 100% di accuratezza: E-7 ed E-7bis

### Il problema (E-7)
Nell'esercizio E-7 abbiamo addestrato due modelli tradizionali (Random Forest e Regressione Logistica) sulle 48 statistiche aggregate dei flussi (durata, byte medi, interarrivi, ecc.) per stabilire una base di confronto (*baseline*).

I risultati sono stati:
- Sul compito a **20 classi** (riconoscere la specifica applicazione/malware): la Random Forest ottiene un buon **F1 = 85.3%**, la Regressione Logistica il **46.6%**. Qui c'è "attrito", il compito è stimolante.
- Sul compito a **2 classi** ($K=2$, Malware vs Benigno): la Random Forest ha ottenuto **F1 = 0.99997 (praticamente 100%)** su tutti i 56.255 flussi del test set!

In qualsiasi esame universitario prendere 100% sembra un trionfo. In questa tesi, invece, **era un campanello d'allarme gravissimo**. Perché?
Perché per studiare il *Neural Collapse* dobbiamo osservare l'evoluzione delle curve nel tempo: come cambiano le rappresentazioni man mano che il modello impara. Se un modello fa 100% in 2 secondi, **non c'è alcuna dinamica da studiare**. Inoltre, c'era il forte sospetto che il dataset (USTC-TFC2016) avesse un trucco: ogni classe proviene da un unico file `.pcap`. Se il modello avesse semplicemente imparato che *"le catture di malware duravano 10 minuti e quelle benigne 2 ore"*, avrebbe fatto 100% barando sulle caratteristiche della registrazione e non sul malware!

### L'indagine e il responso (E-7bis)
Per capire se la Random Forest stava barando, abbiamo analizzato **quali feature stava guardando per decidere**, separandole in due gruppi:
1. **Feature di "scala grezza" (le sospette):** durata totale, byte complessivi scambiati, numero totale di pacchetti. Queste dipendono da quanto a lungo l'operatore ha lasciato acceso Wireshark durante la registrazione.
2. **Feature di "forma" (il vero comportamento):** dimensioni minime/medie dei pacchetti, tempi di interarrivo tra un pacchetto e l'altro, rapporti tra flag TCP (SYN, ACK, PSH, RST). Queste descrivono come "parla" il software, a prescindere da quanto dura la telefonata.

**Il verdetto numerico:**
- Importanza delle feature di **forma**: **79.8%**
- Importanza delle feature di **scala grezza**: **20.2%**

La feature singola più importante in assoluto è risultata `dst2src_min_ps` (la dimensione minima dei pacchetti di risposta), seguita dalla media dei tempi di interarrivo e dai flag di handshake TCP.
**Cosa significa in pratica?** Il compito $K=2$ è molto facile non perché il dataset sia rotto, ma perché le famiglie di malware analizzate hanno una firma di rete intrinsecamente molto diversa da applicazioni ordinarie come Gmail o FaceTime. Possiamo quindi tenere $K=2$ nei nostri esperimenti a testa alta.

---

## 3. L'architettura e i test di sanità: E-8 ed E-9

Prima di addestrare una rete complessa, bisogna accertarsi che il codice non abbia bug silenziosi.

### E-8: Il test della maschera (Padding)
Nel traffico reale, i flussi hanno lunghezze diverse: alcuni scambiano 5 pacchetti, altri 20 o più. Noi analizziamo i primi 20 pacchetti. Se un flusso ne ha solo 5, gli altri 15 vengono riempiti con zeri (*padding*).
Se la rete neurale legge quegli zeri come se fossero veri pacchetti di dimensione zero, impara sciocchezze. Abbiamo quindi implementato una maschera booleana esplicita che dice al Transformer: *"ignora completamente le posizioni da 6 a 20"*.

Abbiamo fatto due verifiche matematiche rigide:
1. **Determinismo:** passando lo stesso flusso due volte in modalità valutazione (`eval`), l'output è identico bit per bit.
2. **Verifica della maschera:** modificando a caso i numeri nelle posizioni mascherate (es. mettendo numeri assurdi nei pacchetti fittizi), l'output per il token riassuntivo `CLS` e per i pacchetti validi **non cambia di una virgola**. Test superato al 100%.

### E-9: Le tre rappresentazioni
Il fine-tuning collega all'encoder una testa di classificazione (un piccolo MLP). Nella guida c'è un avvertimento fondamentale: **non esiste una sola rappresentazione**, ce ne sono tre:
- `z_cls`: il vettore a 64 dimensioni prodotto dall'uscita dell'encoder in corrispondenza del token `CLS`. È la carta d'identità del flusso prodotta dall'encoder. È **questo** che vogliamo riutilizzare per altri compiti futuri.
- `z_mean`: la media dei vettori dei pacchetti validi (un'alternativa a `z_cls` senza parametri).
- `h_pen`: il vettore prodotto dal penultimo strato, cioè *dentro* la testa, subito prima del classificatore finale. **È qui che la teoria dice che avviene il Neural Collapse.**

Se misurassimo solo `z_cls` e non trovassimo il collasso, potremmo pensare che il collasso non esista, quando invece è avvenuto dentro `h_pen`. Viceversa, se il collasso avviene in `h_pen` ma `z_cls` resta ricco di informazioni, significa che l'encoder non si è degradato ed è ancora riutilizzabile! In E-9 abbiamo costruito ed estratto in modo deterministico tutti e tre i vettori.

---

## 4. Il successo del Pre-training: E-10

L'esercizio E-10 era il banco di prova decisivo di tutta la prima metà della tesi.

### Cos'è il pre-training auto-supervisionato?
Immagina di voler insegnare a qualcuno la lingua italiana senza dargli un vocabolario: prendi migliaia di articoli di giornale, cancelli il 30% delle parole a caso con un pennarello nero, e gli chiedi di indovinare le parole mancanti basandosi sul contesto.
Noi abbiamo fatto esattamente questo: abbiamo preso i flussi del training set (senza toccare etichette o classi malware), abbiamo mascherato il 30% dei pacchetti validi, e abbiamo chiesto al `MiniNetEncoder` di ricostruire:
1. La dimensione del pacchetto nascosto (regressione).
2. L'intervallo di tempo dal pacchetto precedente (regressione).
3. La direzione del pacchetto: in entrata o in uscita? (classificazione binaria).

La guida imponeva **due controlli obbligatori** per dichiarare il pre-training riuscito:

### Controllo (i): Battere il predittore banale
Se uno studente non sa rispondere a una domanda a crocette, tira a indovinare rispondendo sempre con la risposta più frequente. Nel nostro caso, la baseline banale è un algoritmo che restituisce sempre la media esatta di dimensioni e tempi calcolata sull'intero dataset.
- Errore (loss) della baseline banale: **1.6415**
- Errore (loss) del nostro modello pre-addestrato: **0.3030**

L'errore è crollato da 1.64 a 0.30. La rete ha imparato la struttura temporale e dimensionale del traffico di rete molto meglio del semplice caso o della media.

### Controllo (ii): Il probe lineare batte l'encoder casuale
Questo è il controllo più importante. Se congeliamo i pesi dell'encoder appena pre-addestrato e ci attacchiamo sopra un semplice classificatore lineare (un *probe*) per predire le 20 classi applicative, questo probe deve funzionare **sensibilmente meglio** dello stesso probe attaccato a una rete inizializzata con pesi casuali.

I risultati ottenuti sul validation set sono stati:
- **Encoder a pesi casuali:** Accuracy = **70.68%** | F1-macro = **70.55%**
- **Encoder pre-addestrato:** Accuracy = **75.27%** | F1-macro = **76.66%**

Un incremento di circa **+5-6 punti percentuali di F1** con una singola trasformazione lineare a pesi congelati dimostra in modo inequivocabile che:
1. Il pre-addestramento ha estratto informazione geometrica semanticamente utile.
2. L'encoder non è una scatola vuota o casuale: ha costruito una mappa coerente dello spazio del traffico.

Entrambi i controlli di E-10 sono stati superati e l'encoder pre-addestrato è stato salvato con successo.

---

## 5. Tabellone di riepilogo: da E-1 a E-10

| Esercizio | Obiettivo | Risultato ottenuto | Significato in parole povere |
| :--- | :--- | :--- | :--- |
| **E-1** | Discesa del gradiente in NumPy da zero | Retta fittata e loss a plateau | Abbiamo capito la matematica fondamentale dei gradienti. |
| **E-2** | Scan del learning rate (0.001 … 10) | Identificata la soglia di divergenza | Sappiamo empiricamente cosa succede se i passi sono troppo lunghi. |
| **E-3** | Gradient checking | Coincidenza analitico/numerico a $10^{-6}$ | Sappiamo verificare matematicamente che non ci siano bug nei gradienti. |
| **E-4** | Implementazione con autograd PyTorch | `param.grad` coincide con E-1 | PyTorch fa la stessa cosa che abbiamo fatto a mano, ma veloce. |
| **E-5** | Bias implicito su 2 Gaussiane (cross-entropy) | Pesi crescono a norma infinita oltre il 100% | Abbiamo visto l'antenato del Neural Collapse in 2D. |
| **E-6** | Preprocessing USTC-TFC2016 e split | 562.549 flussi puliti salvati in Parquet | Dati pronti, standardizzati solo su train, split intatti. |
| **E-7** | Baseline non neurali sulle 48 statistiche | RF 20 classi: F1=0.85; RF 2 classi: F1=1.0 | Abbiamo i punti di riferimento con cui confrontare la rete neurale. |
| **E-7bis** | Indagine sul 100% di $K=2$ | 80% importanza su feature di forma | Il modello guarda il comportamento reale, non artefatti di sessione. |
| **E-8** | Transformer `MiniNetEncoder` e maschera | Determinismo ok, padding non altera l'output | L'architettura è solida e priva di bug di mascheramento. |
| **E-9** | Head e funzione `extract` per 3 vettori | Estrazione deterministica di `z_cls`, `z_mean`, `h_pen` | Siamo pronti a misurare la geometria separando encoder e testa. |
| **E-10** | Pre-training mascherato e controlli | Loss 0.30 vs 1.64; Probe batte random di +6% F1 | **L'encoder è ufficialmente pre-addestrato con successo.** |

---

## 6. Cosa manca adesso e qual è la prossima mossa?

Siamo al giro di boa. Ora che abbiamo l'encoder pre-addestrato, dobbiamo misurare cosa gli succede quando lo sottoponiamo al fine-tuning.

### Passo 1: Fase 3 — Gli strumenti di misura (Settimana 5)
Prima di lanciare il fine-tuning, dobbiamo preparare il "righello" per misurare lo spazio:
- **E-11 (UMAP):** Visualizzare lo spazio bidimensionale delle rappresentazioni, tenendo a mente che UMAP distorce le distanze e serve solo a generare intuizioni visive.
- **E-12 (`effective_rank`):** Scrivere e testare la funzione del rango effettivo: deve restituire $d=64$ per rumore puro, $1$ se i punti sono su una retta, e un valore intermedio per cluster separati.
- **E-13 (La riga $t=0$):** Calcolare tutte le metriche geometriche (rango effettivo, NC1, anisotropia, probe ausiliari su durata/protocollo) sull'encoder pre-addestrato *prima* di toccarlo con il fine-tuning. Questa tabella sarà il nostro termine di paragone assoluto.

### Passo 2: Fase 4 — Il Fine-tuning e la Griglia (Settimane 6–9)
Addestreremo il modello sul compito di classificazione confrontando **4 regimi**:
1. `HEAD`: congeliamo l'encoder e alleniamo solo la testa (controllo di sanità).
2. `FULL`: fine-tuning standard (tutto insieme).
3. `LP-FT`: prima alleni la testa, poi sblocchi tutto (ricetta per evitare distorsioni).
4. `SCRATCH`: alleni da zero con pesi casuali (per verificare se il collasso dipende dal fine-tuning o dall'addestramento in sé).

Campioneremo le rappresentazioni a intervalli logaritmici (passo 0, 1, 2, 5, 10, 20, 50, 100...) e produrremo il grafico definitivo della tesi: **il grafico del Plateau del Rango Effettivo contro $K-1$**.

Se il rango effettivo scende seguendo la retta $K-1$, avremo dimostrato quantitativamente il collasso della rappresentazione nel traffico di rete. Se non scende o se resta confinato in `h_pen` lasciando intatto `z_cls`, avremo dimostrato che i modelli per il traffico possono essere riutilizzati in sicurezza. In entrambi i casi, sarà un contributo scientifico originale e solido.
