# Tesi D — Guida operativa

**Il fine-tuning supervisionato distrugge la geometria composizionale?**

Da leggere insieme a *Rappresentazioni lineari, superposizione e cybersecurity* (in particolare §4.4), che resta il riferimento per il *perché*. Qui si parla del *come*, dentro vincoli precisi.

La mole di lavoro dovrebbe corrispondere a 12 settimane, scrittura tesi compresa. Tutto ciò che segue è progettato dentro questi vincoli. Nessun esperimento dovrebbe richiedere molto più di mezz'ora di calcolo su un portatile.

---
In questa guida ci sono sei fasi. Ognuna ha un obiettivo, degli **esercizi** (`E-n`), delle **domande di controllo** (`Q-n`) sulle quali Le conviene ragionare prima di passare al prossimo punto, e un **criterio di uscita** verificabile.

Sulle IA generative: ottime per spiegarle un concetto in tre modi diversi finché uno non fa presa, e per sbloccarci su errori con il codice. Pessime come sostituto del ragionamento, con un difetto specifico e pericoloso per questa tesi: producono codice che gira, che sembra ragionevole, però che potrebbe star misurando la cosa sbagliata. Quindi avanti con l'AI, ma massima prudenza :)

---

## 0. Vocabolario essenziale, prima di cominciare

Questa guida presuppone alcuni termini che altrimenti si incontrerebbero via via, senza una definizione tutta in un posto. Li raccogliamo qui perché il §7.1 in particolare — dove si parla di regimi HEAD, FULL, LP-FT, SCRATCH — non si legge bene senza di essi. Se qualcosa Le è già chiaro, salti pure avanti.

### 0.1 Modello, encoder, testa, pre-addestramento, fine-tuning

Un **modello**, in questo contesto, è una funzione con dei numeri regolabili (i **parametri**, o **pesi**) che si aggiustano guardando dati, in modo da minimizzare un errore. **Addestrare** un modello vuol dire eseguire questo aggiustamento iterativo dei pesi (si veda anche il glossario finale, §13).

Il modello che costruirà si divide concettualmente in due parti:

- l'**encoder**: la parte che trasforma l'input grezzo (la sequenza di pacchetti) in una **rappresentazione**, cioè un vettore di numeri che ne è un riassunto. È la parte "generica", in linea di principio riusabile per compiti diversi da quello per cui è stata addestrata.
- la **testa** (*head*): un piccolo modulo che prende la rappresentazione prodotta dall'encoder e la trasforma nell'output specifico del compito corrente (per esempio un vettore di *K* punteggi, uno per classe).

Ci sono due modi distinti di addestrare, e la tesi vive esattamente nella differenza fra i due:

- il **pre-addestramento** (*pre-training*): si addestra l'encoder da solo, senza etichette scelte da una persona, su un compito surrogato ricavato dai dati stessi — qui, mascherare parte della sequenza e chiedere al modello di ricostruirla (§5.3). Si chiama anche apprendimento **auto-supervisionato**.
- il **fine-tuning**: si prende un encoder già pre-addestrato, gli si attacca una testa (di solito inizializzata a caso), e si continua l'addestramento — encoder e testa insieme, o solo la testa, a seconda del regime — su un compito supervisionato specifico, con etichette vere assegnate da una persona (qui: le classi di traffico).

Un **modello fondazionale** è un modello pre-addestrato su una grande quantità di dati non etichettati, pensato per essere poi adattato a molti compiti diversi, di solito con fine-tuning. netFound (§2) è un modello fondazionale per il traffico di rete; il piccolo encoder che costruirà è una sua miniatura.

### 0.2 Compito chiuso e regime fuori distribuzione (OOD)

Un **compito chiuso** (*closed-set task*) è un compito di classificazione in cui l'insieme delle classi possibili è fissato e noto in anticipo: ogni esempio, in addestramento come in valutazione, appartiene sicuramente a una delle *K* classi previste. È il caso "normale" con cui si addestra e si valuta un classificatore, ed è quello di cui parla la "versione ottimista" al §1.1.

Il **regime fuori distribuzione** (*out-of-distribution*, **OOD**) è l'opposto: valutare un modello, o una sua rappresentazione, su dati che non assomigliano a quelli visti in addestramento — un compito diverso, classi mai viste, condizioni di rete alterate. La domanda "quanto sopravvive, in una rappresentazione, l'informazione utile per un compito diverso da quello per cui è stata affinata" — che è l'ipotesi H2 di questa tesi — è per definizione una domanda sul comportamento OOD.

### 0.3 config.yaml: dove vivono i parametri

Da qui in avanti la guida userà spesso l'espressione "lo scriva nel `config.yaml`". Lo anticipiamo perché è una regola di progetto, non un dettaglio implementativo: **ogni valore che potrebbe cambiare da un run all'altro — learning rate, numero di epoche, seed, *K*, dimensione nascosta, percentuale di mascheramento, percorsi dei dati, eccetera — deve avere un unico posto in cui è definito**: un file `config.yaml`, che Lei stessa creerà all'inizio del progetto e che il codice **legge**, invece di avere quei valori scritti a mano dentro le funzioni. Due vantaggi concreti: (1) per riprodurre un run basta il file di configurazione più l'hash del commit (§4.4) — non c'è da ricordare quali valori sono stati usati; (2) cambiare *K* o il seed per lanciare la griglia del §7.4 diventa una modifica di una riga, non una caccia nel codice.

### 0.4 Il probe lineare, con calma

Un **probe lineare** è il classificatore più semplice che esista: una singola trasformazione lineare applicata a una rappresentazione, seguita da una soglia o da una softmax. Si usa così: si prende un encoder — pre-addestrato, oppure a un certo checkpoint del fine-tuning — lo si **congela** (non se ne aggiornano più i pesi) e si addestra *soltanto* la matrice del probe sopra le rappresentazioni che quell'encoder produce.

Il probe **viene addestrato** — ma è l'unica cosa che si addestra in quell'esperimento; l'encoder resta fermo. È proprio questo che rende il probe uno strumento di *misura* e non un modello da massimizzare (§3.2): la sua accuratezza dice quanta informazione la rappresentazione *già contiene* in forma linearmente accessibile, non quanto in generale sia possibile estrarne (con abbastanza strati non lineari si estrae informazione anche da rappresentazioni pessime, ed è per questo che il probe va tenuto rigorosamente lineare).

Questo chiarisce anche l'ipotesi H2: *"l'accuratezza di probe lineari per attributi non presenti fra le etichette del fine-tuning decresce"* significa che, a ogni checkpoint *t*, si congela l'encoder di quel checkpoint e si addestra **da zero** un nuovo probe, con lo **stesso budget** per tutti i checkpoint (stesse epoche, stesso learning rate, stessa procedura di ottimizzazione). Il confronto è così leale: cambia solo la rappresentazione su cui il probe lavora, non le risorse con cui viene addestrato. Se i probe di checkpoint diversi avessero budget diversi, un calo di accuratezza potrebbe dipendere dal budget invece che dalla rappresentazione, e la conclusione non varrebbe nulla.

**La matematica del probe, con calma.** Vale la pena vederla scritta per esteso almeno una volta, perché è più semplice di quanto il nome "probe lineare" lasci immaginare — e perché aiuta a rispondere con precisione alla domanda "ma cosa sta misurando, esattamente?".

Il probe classifica (o fa regressione su) l'**attributo che Lei ha scelto di testare**, non le classi del compito di fine-tuning. Vediamolo passo per passo.

*Il punto di partenza.* Per ogni flusso *i* Lei ha due cose, e sono cose diverse fra loro:

- la rappresentazione congelata *z_i* ∈ ℝ^*d* — l'uscita dell'encoder per quel flusso, per esempio con *d* = 64;
- un'etichetta ausiliaria *y_i* per l'attributo che vuole testare — per esempio *y_i* = "TCP" oppure "UDP".

Il punto da non perdere di vista: *y_i* **non viene dalla rappresentazione**. Viene da fuori — nel Suo caso, dalle 48 statistiche calcolate direttamente dal pcap con nfstream (§4.2). Lei sa già, indipendentemente da qualunque cosa l'encoder abbia imparato, qual è il protocollo di ogni flusso: quell'informazione era nei dati fin dall'inizio, prima ancora che l'encoder esistesse. Il probe serve solo a chiederci se quell'informazione, che noi già conosciamo per altra via, è anche *leggibile linearmente* dalla rappresentazione.

*Il probe stesso.* È deliberatamente poca cosa:

p_i = σ(wᵀ z_i + b)

dove *w* ∈ ℝ^*d* e *b* ∈ ℝ sono gli **unici** parametri che si addestrano — non un'architettura nuova, ma esattamente la regressione logistica dell'E-5 — e *p_i* è la probabilità stimata che il flusso *i* sia, diciamo, TCP. La funzione di errore è la cross-entropy fra *p_i* e *y_i*, minimizzata con discesa del gradiente (§3.1) tenendo *z_i* **fissi**: l'encoder non si tocca, si muovono solo *w* e *b*.

*Cosa vuol dire "AUC alta per protocollo".* Se dopo l'addestramento la quantità wᵀz separa bene i flussi TCP da quelli UDP, vuol dire che esiste una **direzione** nello spazio a *d* dimensioni — proprio quella indicata da *w* — lungo cui, proiettando *z*, si legge il protocollo. Non serve che una singola coordinata di *z* "sia" il protocollo: l'informazione può essere spalmata su tutte le *d* dimensioni contemporaneamente — è la superposizione di cui parla il documento principale. Ma se una combinazione lineare la recupera, allora quell'informazione è lì, ed è accessibile in forma lineare: è esattamente questo, non altro, che il probe misura.

*Come sa che sta testando "protocollo" e non qualcos'altro.* Perché è Lei a scegliere *y_i*. La procedura è sempre identica — stesso *z_i*, stesso encoder congelato — e cambia solo l'etichetta esterna: una volta *y* = protocollo, un'altra *y* = classe malware/benigno, un'altra ancora *y* = bin di durata del flusso. Ogni scelta di *y* addestra un *w* diverso e produce un numero diverso. **Un probe = un *w* = un attributo**: quando in tesi scrive "il probe per la durata ottiene AUC 0,7", si riferisce esattamente a questa procedura ripetuta con *y* = durata.

### 0.5 Rango delle rappresentazioni, e rango effettivo

Se si mettono in una matrice tutte le rappresentazioni prodotte dal modello per un insieme di flussi (una riga per flusso), il **rango** di quella matrice, nel senso classico dell'algebra lineare, dice quante di quelle dimensioni sono davvero indipendenti — cioè quante direzioni servono per ricostruire esattamente tutti i punti. È una nozione fragile in pratica: basta un filo di rumore numerico perché il rango classico salti al valore massimo possibile, anche quando quasi tutta la variabilità sta in poche direzioni.

Per questo si usa il **rango effettivo** (formula al §6.2): invece di contare le direzioni con varianza esattamente zero, pesa ciascuna direzione in proporzione a quanta varianza spiega. Il risultato è un numero continuo, non necessariamente intero, che si legge in modo intuitivo: vicino a 1 se tutta la varianza è concentrata su un'unica direzione, vicino a *d* se è distribuita uniformemente su *d* direzioni. È questo il numero che H1 predice debba scendere verso *K* − 1 durante il fine-tuning — non che sopravvivano letteralmente solo *K* − 1 direzioni, ma che la variabilità si concentri sempre di più in quel numero di direzioni.

### 0.6 CKA, subito

Nel §7.4 e altrove comparirà la sigla **CKA** (*Centered Kernel Alignment*): una misura di quanto due insiemi di rappresentazioni — anche di dimensione diversa, anche prodotte da modelli diversi — si somiglino *nella struttura*, cioè nelle distanze relative fra i punti, indipendentemente da rotazioni o riscalamenti. Vale 1 se le due geometrie sono identiche a meno di questi fattori, 0 se non c'è nessuna relazione. La useremo in due modi: per confrontare la rappresentazione a un checkpoint con quella iniziale, cioè misurare quanto si è "spostata" (§7.4, c3), e per confrontare la rappresentazione appresa dalla rete con le feature statistiche fatte a mano (§7.4, c2).

### 0.7 Seed: cos'è, e come lo usiamo qui

Un **seed** è il numero intero che inizializza il generatore di numeri pseudo-casuali. Fissandolo, ogni sorgente di casualità del run — inizializzazione dei pesi, ordine di mescolamento dei dati, mascheramento casuale del §5.3 — diventa riproducibile: stesso seed, stesso codice, stessi dati → risultato identico bit per bit.

Nella tesi il seed serve anche a un secondo scopo, altrettanto importante: distinguere un effetto reale da una fluttuazione casuale. Un singolo run con un singolo seed potrebbe mostrare un calo del rango effettivo solo per un'inizializzazione fortunata (o sfortunata). Per questo **ogni risultato riportato va ripetuto su almeno tre seed diversi** (§4.4), con barre d'errore calcolate su quelle ripetizioni: se le tre raccontano storie diverse, non si ha ancora un effetto — si ha rumore.

---

## 1. La domanda, resa operativa

Prima di formalizzare nulla, mettiamo la domanda in parole povere. Un modello si addestra in due passate distinte, che ha appena visto al §0.1: prima il **pre-addestramento**, in cui l'encoder impara una rappresentazione generale del traffico senza che nessuno gli dica "questo è malware"; poi il **fine-tuning**, in cui gli si insegna il compito specifico, con le etichette vere. La domanda di questa tesi è: durante il fine-tuning, cosa succede a tutta l'informazione che l'encoder si era costruito nel pre-addestramento, ma che non serve strettamente al compito nuovo? Sopravvive intatta? O il modello, specializzandosi, "dimentica" tutto il resto?

Non è una domanda oziosa. Se vale il secondo scenario, un encoder sottoposto a fine-tuning è un vicolo cieco: non lo si può riusare per nient'altro senza ripartire da zero. Se vale il primo, il fine-tuning è "gratis": si specializza il modello e si tiene comunque la possibilità di riusarlo. In letteratura esistono due risultati pubblicati che suggeriscono risposte opposte, e nessuno li ha ancora messi a confronto sul traffico di rete. Lo farà lei.

### 1.1 I due risultati in conflitto

**Versione ottimista (Vaze et al., ICLR 2022).** Su problemi di visione, migliorare l'accuratezza di un classificatore sul compito chiuso (§0.2) — cioè allenarlo meglio sulle classi note — rende quello stesso modello anche migliore nel riconoscere che un input *non* appartiene a nessuna delle classi note (il cosiddetto *riconoscimento a insieme aperto*, si veda il glossario del documento principale). In altre parole: allenare bene non sembra costare nulla altrove. Corollario implicito: il fine-tuning affina la rappresentazione, non la distrugge.

**Versione pessimista.** Due risultati indipendenti dicono il contrario:

- **Neural collapse** (Papyan, Han, Donoho, PNAS 2020). Il nome descrive bene il fenomeno: nella fase finale dell'addestramento di un classificatore a *K* classi, le rappresentazioni di tutti gli esempi di una stessa classe collassano, cioè diventano sempre più simili fra loro fino quasi a coincidere, mentre le classi diverse si dispongono nello spazio nel modo più simmetrico possibile. La conseguenza pratica (si veda anche il rango effettivo, §0.5): alla fine la rappresentazione può usare al più *K* − 1 direzioni indipendenti. Con *K* = 2, una sola. Tutto il resto dell'informazione che l'encoder conteneva prima — dettagli irrilevanti per "malware sì/no" ma magari utili per un altro compito — evapora.
- **Feature distortion** (Kumar et al., ICLR 2022). Nel fine-tuning "standard", in cui encoder e testa si addestrano insieme fin dall'inizio, i primi passi — quando la testa è ancora casuale e quindi sbaglia molto — generano gradienti grandi e disordinati che si propagano anche all'encoder e ne stravolgono le rappresentazioni pre-addestrate. Il danno si vede soprattutto in regime OOD (§0.2). Da qui la ricetta **LP-FT** (*linear probing then fine-tuning*): prima si addestra solo la testa, a encoder congelato, e solo dopo si sblocca tutto — così quei gradienti iniziali non raggiungono mai l'encoder.

Le due storie non possono essere entrambe vere senza qualificazioni: se il fine-tuning "affina soltanto", da dove viene un collasso a *K* − 1 direzioni? E se il collasso è generale, perché migliorare l'accuratezza chiusa dovrebbe aiutare anche il riconoscimento dell'ignoto? Nel dominio del traffico di rete nessuno ha ancora messo le due cose a confronto sugli stessi dati. Questa è la tesi.

### 1.2 Le ipotesi, in forma falsificabile

Un po' di notazione, che userà per il resto della guida. Chiami *f_t* il modello al passo *t* del fine-tuning: *t* = 0 è il momento subito dopo aver attaccato la testa (ancora casuale) all'encoder pre-addestrato, prima di qualunque passo di ottimizzazione; *t* cresce mano a mano che l'addestramento procede. *K* resta il numero di classi del compito.

Con questa notazione, le tre ipotesi da mettere alla prova:

- **H1 (collasso).** All'aumentare di *t*, il rango effettivo della rappresentazione (§0.5) decresce e si stabilizza — raggiunge cioè un **plateau**, si veda il glossario — a un valore che **dipende da *K*** e non dalla quantità o ricchezza dei dati. Nella sua forma più precisa, e più rischiosa da difendere: il plateau vale circa *K* − 1.
- **H2 (perdita di informazione riutilizzabile).** All'aumentare di *t*, l'accuratezza di un probe lineare (§0.4) per attributi **non presenti fra le etichette del fine-tuning** — per esempio il protocollo di trasporto, se il compito è "malware sì/no" — decresce. E decresce *dopo* che la prestazione sul compito chiuso ha già smesso di migliorare: l'informazione in più si perde anche quando, per il compito in corso, non ce n'è più bisogno.
- **H3 (esiste un regime utile).** Esiste un intervallo di *t* in cui la prestazione sul compito è già vicina al massimo, ma H1 e H2 non si sono ancora manifestate del tutto. Se esiste, la sua ampiezza è già un risultato pratico: dice a chi costruisce un modello quando fermarsi se vuole un risultato riusabile.

Osservi la struttura, perché conta: H1 e H2 sono **direzionali** — dicono che qualcosa scende — e quindi un esperimento reale può smentirle (il rango potrebbe non calare, i probe potrebbero perfino migliorare). H3 è condizionale all'esistenza di quell'intervallo. Nessuna delle tre può restituire "nulla di informativo" per definizione — ma questo è vero solo **a condizione che lei abbia dimostrato di essere in grado di rilevare l'effetto, se ci fosse**. Ci torniamo al §7.5, ed è il punto più importante di tutto il documento: un esperimento che non vedrebbe l'effetto nemmeno se ci fosse non dimostra nulla, dice solo che non si è guardato abbastanza da vicino.

### 1.3 Il disegno che rende H1 falsificabile a costo quasi nullo

Se H1 vale nella forma forte, il plateau del rango effettivo dipende da *K*. Quindi: **stesso esperimento, stessi dati, stesso modello, facendo variare solo *K***. Se il plateau segue *K* − 1, ha una conferma numerica; se il plateau è identico per *K* = 2 e *K* = 20, la forma forte è falsa e lei ha un risultato negativo pulito e quantificato.

Il dataset USTC-TFC2016 le regala questo disegno gratis: le stesse catture sono etichettabili come 20 classi (applicazione) oppure come 2 (malware/benigno), e con raggruppamenti intermedi ottiene anche *K* = 5 e *K* = 10. **Quattro punti su un grafico, un solo dataset.** Quella figura — plateau del rango effettivo contro *K* − 1, con barre d'errore su tre seed — è la figura principale della sua tesi.

> **Q-1.** Perché una predizione numerica (*K* − 1) vale molto più di una direzionale ("decresce")?

> **Q-2.** Se il rango effettivo cala come previsto, questo dimostra che è il *fine-tuning* a causarlo? Quale controllo serve per escludere che sia il semplice addestramento supervisionato, a prescindere dal pre-addestramento?

---

## 2. Modello

Lei costruirà un modello fondazionale (§0.1) in miniatura: un piccolo encoder che replica la struttura essenziale di netFound — un modello fondazionale reale per il traffico di rete, pubblicato in letteratura — ma su scala ridotta di circa tre ordini di grandezza (mille volte più piccolo). Il motivo non è pigrizia: usare netFound così com'è richiederebbe GPU, molto più tempo e molto più spazio su disco, e non potremmo permetterci il numero di esperimenti — molti *K*, molti seed — di cui questa tesi ha bisogno per essere convincente.

La tabella qui sotto confronta riga per riga netFound e il suo modello: la legga come un elenco di "cosa teniamo" e "cosa semplifichiamo", non come una specifica tecnica da imparare a memoria.

|                         | netFound                                                              | il suo modello                                                       |
| ----------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------- |
| input                   | header tokenizzati per campo di protocollo, metadati di burst         | sequenza dei primi *n* pacchetti: direzione, dimensione, interarrivo |
| privacy per costruzione | no payload, no IP                                                     | idem: la sequenza SPLT non contiene né payload né indirizzi          |
| architettura            | transformer gerarchico burst→flusso                                   | transformer piatto, 2–4 strati                                       |
| dimensione nascosta     | 512                                                                   | 64                                                                   |
| parametri               | 53 M                                                                  | ~0,2 M                                                               |
| pre-addestramento       | mascheramento di token + 3 obiettivi ausiliari, ~10 miliardi di token | mascheramento di posizioni, ~10⁵ flussi                              |
| costo di addestramento  | migliaia di ore GPU                                                   | minuti su CPU                                                        |

Non si preoccupi se "transformer" non Le dice ancora nulla: è il tipo di architettura di rete neurale che useremo (dettagli al §5.1), lo stesso alla base dei modelli linguistici moderni. Per ora basta sapere che elabora una sequenza di elementi — qui, i pacchetti di un flusso — tenendo conto delle relazioni fra tutte le posizioni contemporaneamente, invece di leggerle una alla volta in ordine. "Parametri" sono i numeri regolabili del modello (§0.1); "dimensione nascosta" (*hidden dimension*, si veda il glossario) è la lunghezza del vettore di rappresentazione che l'encoder produce internamente.

La rappresentazione SPLT (*sequence of packet lengths and times*) non è una semplificazione arbitraria: è la rappresentazione standard nella letteratura sulla classificazione del traffico cifrato da vent'anni, ed è esattamente il tipo di segnale — temporale e dimensionale, non di contenuto — su cui netFound costruisce il proprio *operational context embedding*.

**Cosa non facciamo, e va scritto in tesi.** Non affrontiamo la scala del pre-addestramento (e quindi la possibilità di affermare qualcosa sui modelli fondazionali *reali*), non vediamo la gerarchia burst/flusso, né la ricchezza dei campi di protocollo. Le nostre conclusioni valgono per un modello di questa scala e vanno formulate così: *«in una miniatura controllata di un encoder per il traffico, il compromesso fra prestazione e riusabilità si comporta così»*. È un'affermazione più modesta e molto più difendibile di quella che avrebbe ottenuto con un solo run su un modello grande.

**Cosa guadagna, ed è molto.** Controllo totale sul pre-addestramento — sa esattamente cosa il modello ha visto, quindi nessuna contaminazione fra dati di pre-addestramento e di test, che è il difetto che affligge buona parte della letteratura sul tema. Costo per run trascurabile, quindi molti seed, molti valori di *K*, veri intervalli di confidenza. E la possibilità di eseguire l'analisi di potenza del §7.5, che senza run economici è semplicemente impossibile.

> **Q-3.** Un revisore obietta: «il neural collapse è un fenomeno dei modelli sovraparametrizzati, il suo modello da 200k parametri non è nel regime giusto». Prepari la risposta. Suggerimento: ciò che conta non è il numero assoluto di parametri ma la capacità di interpolare il training set. Come lo verifica empiricamente?

---

## 3. Fase 0 — Prerequisiti (settimana 1)

**Obiettivo:** poter leggere e scrivere un ciclo di addestramento senza che si sembri magia nera.

### 3.1 La discesa del gradiente

Cominciamo dall'idea, senza formule, perché tutto il resto della tesi poggia su questo singolo meccanismo ripetuto migliaia di volte.

Immagini di essere bendata in mezzo a una collina e di voler arrivare al punto più basso, senza poter vedere il paesaggio. L'unica cosa che può fare è sentire, sotto i piedi, da che parte il terreno scende di più in quel punto preciso, fare un piccolo passo in quella direzione, e ripetere. Non è il modo più furbo possibile di scendere — non vede il resto della collina, potrebbe fermarsi in un avvallamento che non è il punto più basso in assoluto — ma funziona sorprendentemente bene, ed è praticabile anche quando la collina ha migliaia o milioni di dimensioni, dove "vedere il paesaggio" non è un'opzione per nessuno.

Ora traduciamo. La "collina" è la **funzione di errore** (o *loss*): un numero che dice quanto il modello sta sbagliando, calcolato a partire dai suoi parametri attuali (i pesi, si veda §0.1) e dai dati di esempio. La "posizione" in cui si trova sono proprio i valori correnti dei parametri. Il "sentire da che parte scende di più" è il **gradiente**: una formula che, per ciascun parametro, dice in che direzione e di quanto cambierebbe l'errore se si spostasse quel parametro di un pochino. Il "piccolo passo" è l'aggiornamento: si sposta ogni parametro nella direzione opposta al gradiente (opposta, perché il gradiente punta dove l'errore *cresce* di più, e noi vogliamo che *scenda*), di una quantità proporzionale a un numero che si chiama **learning rate** — quanto è lungo ogni passo. Ripetere questo passo migliaia di volte è tutto ciò che "addestrare" un modello, alla fine, vuol dire: **discesa del gradiente** (*gradient descent*) è il nome di questa procedura.

Due cose da tenere a mente prima degli esercizi:

- Il passo può essere troppo lungo. Se il learning rate è troppo grande, invece di scendere verso il minimo il passo lo scavalca, e da lì il successivo scavalca ancora di più: l'errore, invece di scendere, comincia a crescere senza controllo. Si chiama **divergenza**, e la vedrà con i suoi occhi nell'E-2.
- Il gradiente non è magia: è solo la derivata dell'errore rispetto a ciascun parametro, la stessa nozione di analisi che si studia al liceo o al primo anno, applicata a una funzione con molti argomenti invece che uno solo. Per questo l'esercizio successivo Le chiede di calcolarlo a mano prima di lasciarlo fare a una libreria: se non sa cosa dovrebbe uscire, non saprà mai se il codice ha un bug.

Con questo in mente, gli esercizi:

**E-1.** In NumPy, senza librerie di machine learning: generi 200 punti secondo *y* = 3*x* + 1 + rumore (una retta con un po' di disordine sopra), poi trovi la retta che meglio approssima quei punti — cioè trovi i due parametri (pendenza e intercetta) che minimizzano l'errore — usando la discesa del gradiente appena descritta, non una formula chiusa. Cerchi su internet le formule del gradiente per la regressione lineare, non le librerie che lo calcolano al posto suo. **Calcoli il gradiente a mano su carta prima** di scriverlo in codice. Alla fine, faccia un grafico dell'errore contro il numero di iterazioni: deve scendere e appiattirsi.

**E-2.** Rifaccia lo stesso esperimento con learning rate 0,001 / 0,01 / 0,1 / 1 / 10. Descriva a parole cosa succede in ciascun caso — scende lentamente? scende veloce e si stabilizza? oscilla? esplode? — e trovi sperimentalmente la soglia oltre la quale si osserva la divergenza descritta sopra.

**E-3.** *Gradient checking:* un modo per verificare che il gradiente scritto a mano sia corretto, confrontandolo con un'approssimazione numerica che non richiede di aver capito la formula, solo di saper valutare la funzione di errore: (*L*(θ+ε) − *L*(θ−ε))/(2ε), con ε = 10⁻⁵. Il suo gradiente analitico e questa approssimazione devono coincidere a 6–7 cifre decimali. Se non coincidono, il bug è nel gradiente scritto a mano, non nell'approssimazione. Questa tecnica Le farà risparmiare giorni interi più avanti: la impari adesso, su un caso in cui può verificare tutto a occhio.

**E-4.** Rifaccia l'E-1, questa volta in PyTorch, lasciando che sia la libreria a calcolare il gradiente per lei con `loss.backward()` e ad aggiornare i parametri con `torch.optim.SGD`. Stampi `param.grad` dopo la chiamata e verifichi che assomigli al gradiente calcolato a mano in E-1: è la stessa identica cosa, solo automatizzata.

**E-5.** Un task leggermente diverso: regressione logistica su due gaussiane ben separabili (due nuvole di punti distinte, facili da separare con una retta), addestrata con cross-entropy invece dell'errore quadratico. Continui ad addestrare **molto oltre** il punto in cui l'accuratezza è già al 100%, e osservi la norma dei pesi (cioè quanto sono "grandi" i parametri nel loro insieme): invece di stabilizzarsi, cresce indefinitamente. Questo fenomeno — chiamato *implicit bias* verso il margine massimo — è un parente stretto del neural collapse che incontrerà al §3.3. Averlo visto qui, su un problema in due dimensioni disegnabile su un foglio, prima di cercarlo in uno spazio a 64 dimensioni, cambia tutto: saprà riconoscerlo.

Materiale: `micrograd` di Karpathy, primo video di *Neural Networks: Zero to Hero* — due ore che valgono un semestre di slide, e costruisce la discesa del gradiente esattamente come l'abbiamo descritta sopra, un pezzo alla volta. In aggiunta i capitoli 1–4 di [3Blue1Brown](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) per l'intuizione geometrica del gradiente.

> **Q-4.** Perché nella cross-entropy compare il logaritmo? Cosa va storto usando l'errore quadratico sulle probabilità? Guardi la forma del gradiente nei due casi.

### 3.2 Le tre cose che si confondono

|                              | cosa si addestra                                          | a cosa serve                                             |
| ---------------------------- | --------------------------------------------------------- | -------------------------------------------------------- |
| **probe lineare**            | solo una mappa lineare su rappresentazioni congelate      | *misurare* quanta informazione è linearmente accessibile |
| **testa di classificazione** | un piccolo modulo, di solito non lineare, sopra l'encoder | *risolvere* un compito                                   |
| **fine-tuning**              | encoder + testa insieme                                   | adattare il modello                                      |

Il probe è uno strumento di misura, non un modello che deve funzionare bene. Se lo rendiamo non lineare o troppo capace smette di misurare l'accessibilità lineare e comincia a misurare sé stesso. Teniamolo rigorosamente lineare, con regolarizzazione scelta su validazione, e riportiamo **sempre** la baseline con etichette permutate.

> **Q-5.** Supponiamo che un probe ottiene AUC 0,95 per «protocollo» e 0,55 per «durata del flusso». Sono lecite entrambe queste conclusioni: (a) la rappresentazione contiene il protocollo; (b) la rappresentazione non contiene la durata? Una delle due è illecita: quale e perché?

### 3.3 Il neural collapse in dieci righe, e tre avvertenze che decidono tutto

L'idea, in una frase: se si continua ad addestrare un classificatore anche dopo che ha già smesso di sbagliare sui dati di addestramento, le sue rappresentazioni interne collassano in una configurazione geometrica estremamente semplice e rigida — e questo, per motivi che vedremo, limita drasticamente quanta altra informazione quella rappresentazione può ancora contenere.

In dettaglio: Papyan, Han e Donoho osservano che, continuando ad addestrare *oltre* l'azzeramento dell'errore di training (la cosiddetta «fase terminale»), succedono quattro cose: la variabilità dentro ciascuna classe collassa, cioè tutti gli esempi di una classe finiscono per avere quasi la stessa rappresentazione (NC1); le medie delle diverse classi si dispongono nello spazio nel modo più simmetrico possibile, un «simplesso equiangolare» — pensi a un tetraedro perfetto se le classi sono quattro (NC2); i pesi dell'ultimo strato si allineano a quelle medie (NC3); classificare si riduce a guardare quale media di classe è più vicina (NC4). La conseguenza geometrica, quella che ci interessa: *K* punti (le *K* medie di classe) in posizione generica occupano al più uno spazio di dimensione *K* − 1 — due punti stanno su una retta, tre su un piano, e così via — quindi la rappresentazione finisce per vivere, di fatto, in *K* − 1 dimensioni.

Tre avvertenze, e sono le tre cose che possono affondare l'esperimento:

1. È un fenomeno del **training set**. Le metriche NC vanno calcolate sui dati di addestramento. Calcolate sul test set e non vedendo nulla, non abbiamo smentito niente.
2. È un fenomeno della **fase terminale**. Richiede di addestrare oltre l'interpolazione del training set. Un fine-tuning di quattro epoche con early stopping non ci arriva mai. Dobbiamo **progettare l'esperimento perché il fenomeno possa manifestarsi**: training set abbastanza piccolo da poterlo interpolare, nessun early stopping, molte epoche. È il vantaggio concreto di avere run che costano minuti.
3. Riguarda il **penultimo strato**, cioè l'ingresso del classificatore lineare finale — non l'uscita dell'encoder. Sono cose diverse, e nel §5.2 vedremo che confonderle è l'errore più facile da commettere.

Legga l'articolo del 2020 almeno nelle figure: sono didattiche.

> **Q-6.** Se il collasso è un fenomeno del training set, perché dovrebbe preoccupare qualcuno? Costruisca l'argomento per cui è un problema *pratico* e non una curiosità geometrica.

**Criterio di uscita:** sa scrivere in venti righe di NumPy una discesa del gradiente e verificarla con differenze finite; sa dire su quali dati va calcolato NC1 e perché.

---

## 4. Fase 1 — I dati (settimana 2)

**Obiettivo:** un file tabellare, piccolo, pulito, con le partizioni fissate per sempre.

### 4.1 Il dataset

**USTC-TFC2016**: 20 file pcap, 3,71 GB, dieci applicazioni benigne e dieci famiglie di malware. Reperibile da `github.com/davidyslu/USTC-TFC2016` (alcuni file sono compressi in `.7z`).

**Avvertenza da mettere per iscritto in tesi.** È un dataset del 2016 con difetti noti: alberi decisionali addestrati su di esso finiscono per usare indirizzi MAC e byte di timestamp. Lei è parzialmente protetta perché userà solo direzione, dimensioni e tempi *relativi* — niente indirizzi, niente payload, niente timestamp assoluti. Ma c'è una conseguenza sperimentale seria: **un compito troppo facile non ha dinamica**. Se il modello arriva a F1 = 0,999 in cinquanta passi, non c'è nessuna curva da studiare. Se le succede, riduca il training set o aggiunga rumore controllato alle etichette. Non è barare: è costruire un esperimento che abbia qualcosa da mostrare, e va dichiarato apertamente.

### 4.2 Da pcap a tabella, in un colpo solo

```python
from nfstream import NFStreamer

df = NFStreamer(
    source="data/raw/Cridex.pcap",
    statistical_analysis=True,   # 48 feature statistiche per flusso
    splt_analysis=20,            # sequenza dei primi 20 pacchetti
    n_dissections=0,
).to_pandas()
```

Ottiene, per ogni flusso: `splt_direction`, `splt_ps` (dimensioni), `splt_piat_ms` (interarrivi) come liste di 20 elementi, più 48 statistiche aggregate — minimo, media, massimo e deviazione standard di dimensioni e interarrivi per direzione, analisi dei flag TCP, durata, conteggi.

Le due cose servono a scopi diversi:

- le **sequenze SPLT** sono l'input del suo modello;
- le **48 statistiche** sono la baseline non-neurale e il termine di confronto per la CKA del §7.4.

**Dopo l'estrazione può cancellare i pcap.** La tabella risultante è di poche centinaia di MB. Conservi lo script: la riproducibilità sta lì, non nei 3,7 GB.

**E-6.** Estragga tutti i 20 pcap, unisca le corrispondenti sequenze in un unico dataframe con una colonna `label_20` e una `label_2`. Riporti la tabella dei conteggi per classe. Ci aspettiamo  forte sbilanciamento: per il pre-training, faremo un  sottocampionamento a classi bilanciate, la scelta più semplice e la più difendibile — scriviamolo nel `config.yaml`.

### 4.3 Preprocessing dei valori

Quattro decisioni, tutte da giustificare in tesi:

- le dimensioni dei pacchetti sono conteggi a coda lunga → scaliamo usando `log1p`;
- gli interarrivi idem, con molti zeri → `log1p` sui millisecondi;
- la direzione è categorica {in, out} →mappiamo a  {−1, +1};
- flussi con meno di 20 pacchetti → padding con **maschera esplicita**, non con zeri silenziosi. Gli zeri sono valori legittimi, e un modello che non distingue «pacchetto assente» da «pacchetto di dimensione zero» imparerà sciocchezze.

Standardizziamo con media e deviazione standard **calcolate solo sul training set**. Se le calcoliamo su tutto il dataset abbiamo già fatto data snooping, che è la trappola più comune in questo ambito di ricerca e la prima che ci contesteranno.

### 4.4 Le partizioni, e le regole non negoziabili

1. **Divisione train / validation / test per flusso, decisa prima di tutto il resto, mai più toccata.** Mai per pacchetto.
2. **Il test set si guarda una volta sola, alla fine.** Ogni scelta — learning rate, epoche, architettura — si fa sul validation set.
3. **Un `probe set`** separato di 2000–5000 flussi stratificati, mai usato per addestrare, su cui estrarremo le rappresentazioni a ogni checkpoint.
4. **Ogni run ha un seed** registrato nel `config.yaml`, e ogni risultato riportato è replicato su almeno 3 seed. Un effetto che si vede con un seed e non con gli altri due non è un effetto.
5. **`git commit` prima di ogni run**, con l'hash del commit nel `config.yaml`. A ottobre potremmo non ricordare quale versione ha prodotto la figura 7.

### 4.5 Le baseline non-neurali, subito

**E-7.** Regressione logistica e random forest sulle 48 statistiche, con il protocollo del §4.4, per *K* = 2 e *K* = 20. Riportiamo F1 pesato e matrice di confusione.

Serve a tre cose: ci dà un termine di paragone onesto (se il nostro transformer non batte una random forest su feature aggregate, ---e probabilmente non lo farà, ma va bene così, un giorno glielo spiego a voce perché--- lo dobbiamo scrivere in tesi); ci fa attraversare l'intero protocollo sperimentale prima di aggiungere la complicazione delle reti neurali; e ci dice quanto è difficile il compito, che è l'informazione di cui abbiam bisogno per il §4.1.

**Criterio di uscita:** un `.parquet` con tutti i flussi, colonne SPLT, 48 statistiche, `label_20`, `label_2` e una colonna `split ∈ {train, val, test, probe}`. La tabella dei conteggi. La tabella delle baseline.

---

## 5. Fase 2 — Il modello (settimane 3–4)

**Obiettivo:** un encoder che addestra, e la consapevolezza di *quale* vettore sta guardando.

### 5.1 Architettura

Piccola e senza fronzoli:

```python
import torch, torch.nn as nn

class MiniNetEncoder(nn.Module):
    def __init__(self, n_pkt=20, d=64, layers=4, heads=4):
        super().__init__()
        self.inp = nn.Linear(3, d)                     # (direzione, log size, log iat)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, n_pkt + 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d,
                                         batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(enc, layers)

    def forward(self, x, pad_mask):                    # x: (B, n_pkt, 3)
        h = self.inp(x)
        h = torch.cat([self.cls.expand(h.size(0), -1, -1), h], dim=1) + self.pos
        # TODO: estendere pad_mask di una posizione per il CLS
        return self.body(h, src_key_padding_mask=pad_mask)   # (B, n_pkt+1, d)
```

Circa 200 000 parametri. Un passo su un batch da 256 richiede frazioni di secondo su CPU.

**E-8.** Completi la gestione della maschera e verifichi con due test: (i) passi due volte lo stesso batch con `model.eval()`, le uscite devono essere identiche bit per bit; (ii) cambi i valori nelle posizioni mascherate e verifichi che l'uscita **non** cambi. Se cambia, la maschera non funziona, e potremmo passare tre settimane a inseguire risultati privi di senso.

### 5.2 Le tre rappresentazioni

Questo è forse il punto tecnico più importante della guida.

La testa di classificazione è un piccolo MLP:

```python
class Head(nn.Module):
    def __init__(self, d=64, K=20):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.classifier = nn.Linear(d, K)

    def forward(self, z):
        h = self.mlp(z)
        return self.classifier(h), h
```

Esistono quindi almeno tre vettori diversi che si possono chiamare «la rappresentazione»:

| nome     | definizione                                           | a cosa serve                                       |
| -------- | ----------------------------------------------------- | -------------------------------------------------- |
| `z_cls`  | uscita dell'encoder in posizione CLS                  | *ciò che si riuserebbe*: dipende solo dall'encoder |
| `z_mean` | media dell'uscita dell'encoder sulle posizioni valide | aggregazione alternativa, senza parametri          |
| `h_pen`  | uscita di `mlp`, cioè l'ingresso di `classifier`      | è **questa** che il neural collapse riguarda       |

Il codice «ovvio» che ci suggerirebbe qualunque AI — prendi l'uscita dell'encoder e misuraci sopra il collasso — misura la cosa sbagliata. `z_cls` e `h_pen` sono separati da due strati densi addestrabili. Se misuriamo il collasso su `z_cls` e non lo troviamo, non abbiamo smentito il neural collapse: abbiam misurato un'altra cosa. Se lo troviamo su `h_pen`, non abbiam ancora mostrato che l'*encoder* si sia degradato: potrebbe essersi degradata solo la testa.

E la differenza fra i due è un risultato, non un fastidio: dice se il danno resta confinato nella testa o si propaga all'encoder. Che è poi l'unica domanda che interessi davvero a chi vuole riusare un modello.

Una nota su `z_mean`: usi una media **non pesata**, senza parametri. Un pooling con parametri addestrabili renderebbe impossibile attribuire un cambiamento all'encoder anziché al pooling. (Nota di secondo ordine, per quando avrà tempo: la media è essa stessa un'aggregazione, e il §3.1 del documento principale spiega perché le medie sono infide. Se ne ricordi quando interpreterà i risultati.)

**E-9.** Scriva `extract(model, loader, which)` con `which ∈ {"z_cls", "z_mean", "h_pen"}`, che restituisce un array (*n*, *d*) e gli indici **nell'ordine originale**. Con `model.eval()`, dentro `torch.inference_mode()`. Verifichi il determinismo.

### 5.3 Il pre-addestramento auto-supervisionato

Obiettivo: mascheramento e ricostruzione. Si maschera il 30% delle posizioni di pacchetto e si chiede al modello di ricostruirle.

- perdita su dimensione e interarrivo (valori standardizzati): errore quadratico o L1, **sulle sole posizioni mascherate**;
- perdita sulla direzione: cross-entropy binaria, sulle sole posizioni mascherate.

Usi **soltanto i flussi del training set**. Altrimenti ha contaminato il test attraverso il pre-addestramento, che è esattamente il difetto che l'articolo di netFound rimprovera a buona parte della letteratura esistente. Poterlo garantire è uno dei vantaggi concreti di avere un modello proprio: lo scriveremo in tesi.

**E-10.** Implementi il pre-addestramento e verifichi che funzioni con due controlli: (i) la loss di ricostruzione scende sotto quella di un predittore banale che restituisce sempre la media; (ii) un probe lineare per `label_20` sopra `z_cls` **congelato** fa meglio dello stesso probe sopra un encoder a pesi casuali. Se (ii) fallisce, il pre-addestramento non ha imparato nulla di utile e non ha senso proseguire: torniamo indietro e capisciamo perché.

Il controllo (ii) non è burocrazia: è la definizione operativa di «questo modello è pre-addestrato in modo utile», ed è la premessa di tutta la tesi.

**Criterio di uscita:** un encoder pre-addestrato salvato, i due controlli di E-10 superati, la funzione `extract` verificata.

---

## 6. Fase 3 — Guardare i dati senza farsi ingannare (settimana 5)

**Obiettivo:** saper generare ipotesi da una figura senza trarne conclusioni.

### 6.1 UMAP, e perché non deve fidarsene

```python
import umap
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0)
emb2d = reducer.fit_transform(X)
```

È utilissimo per accorgersi di cose: due classi che si sovrappongono, un cluster spurio che corrisponde a un singolo file pcap, punti che si compattano dopo il fine-tuning.

C'è una controversia scientifica documentata su quanto ci si possa fidare, ed è un ottimo esercizio di igiene epistemica:

- Chari & Pachter, *The specious art of single-cell genomics* (PLOS Comp. Biol. 2023): t-SNE e UMAP distorcono le distanze e non preservano né la struttura locale né quella globale. Gli autori costruiscono `Picasso`, un metodo che proietta *qualunque* dato nella forma di un elefante con metriche di fedeltà comparabili a quelle di UMAP.
- La replica di Kobak e colleghi, *The art of seeing the elephant in the room*: le metriche usate erano inadeguate, UMAP preserva vicinati e classi meglio di quanto sostenuto.
- Il punto su cui **entrambe le parti concordano**: le proiezioni 2D distorcono le distanze e non vanno usate per analisi quantitativa a valle.

**E-11.** Legga entrambi e scriva mezza pagina: chi ha ragione su cosa, e quale conclusione pratica ne trae. Riscritta, entrerà nel capitolo di metodologia.

Regola operativa: **UMAP genera ipotesi, i numeri nello spazio originale le confermano.** Ogni figura UMAP in tesi va accompagnata da una misura quantitativa calcolata in ℝ^d che dice la stessa cosa. Se la misura contraddice la figura, ha ragione la misura.

Dettaglio pratico: fissi `random_state` e i parametri una volta sola, e per i checkpoint successivi usi `reducer.transform`, non `fit_transform`. Altrimenti confronta due proiezioni diverse, e le differenze che vede potrebbero essere soltanto l'inizializzazione.

### 6.2 La cassetta degli attrezzi quantitativa

Raccogliamo qui le misure quantitative che userà per tutta la tesi. Non sono una collezione arbitraria: ciascuna risponde a una domanda diversa, e ciascuna ha una trappola specifica in cui è facile cadere. Le presentiamo con calma, una alla volta.

**Anisotropia.** Si calcola come il coseno medio dell'angolo fra coppie di rappresentazioni prese a caso. Dice quanto lo spazio delle rappresentazioni è «stretto» — quanto, cioè, i vettori tendono ad ammucchiarsi tutti nella stessa direzione invece di sparpagliarsi in ogni verso. La trappola: in questi spazi l'anisotropia è **alta quasi sempre**, indipendentemente da cosa il modello abbia imparato. Un coseno di 0,9 fra due gruppi di rappresentazioni non significa per forza "collasso" se il «fondo» — il coseno fra coppie prese a caso, senza nessuna relazione fra loro — è già 0,7. Va sempre riportata insieme a questa baseline, mai da sola.

**Rango effettivo.** Introdotto al §0.5, si calcola come exp(−Σ*p_i* log *p_i*), dove *p_i* = λ_i / Σλ_j e λ_i sono gli autovalori della matrice di covarianza delle rappresentazioni. Dice quante direzioni la rappresentazione sta effettivamente usando. La trappola: dipende dalla scala dei dati, quindi va calcolato dopo aver centrato le rappresentazioni (cioè sottratto la loro media), e Lei deve decidere esplicitamente — e dichiararlo in tesi — se le sta anche normalizzando (dividendo per la deviazione standard).

**NC1.** Si calcola come la traccia di Σ_B^† Σ_W, dove Σ_W è la covarianza intra-classe, Σ_B quella inter-classe, e † indica la pseudoinversa di Moore-Penrose (non l'inversa ordinaria, che qui in generale non esiste). Dice quanto le rappresentazioni della stessa classe si sono ravvicinate fra loro rispetto a quanto le classi si sono allontanate l'una dall'altra: è la misura diretta del collasso descritto al §3.3. La trappola, già vista ma che vale ripetere: va calcolato **sul training set**, mai sul test set.

**AUC di un probe lineare.** Si addestra un probe lineare (§0.4) su un attributo di interesse e si misura la sua area sotto la curva ROC. Dice quanta informazione su quell'attributo è accessibile con una singola trasformazione lineare. La trappola: da sola l'AUC non basta, perché anche un probe che non ha imparato nulla di reale può ottenere un valore diverso da 0,5 per puro rumore statistico, specie su un dataset piccolo. Serve sempre una distribuzione nulla ottenuta permutando le etichette.

**CKA lineare centrata.** Introdotta al §0.6, misura quanto due insiemi di rappresentazioni — anche di dimensione diversa — si somiglino nella loro struttura interna. La trappola: va calcolata su un insieme **fisso** di campioni, confrontati nello stesso ordine nei due insiemi; confrontare rappresentazioni estratte da campioni diversi, o messi in ordine diverso, produce un numero senza senso.

**Silhouette e accuratezza kNN.** Due misure più tradizionali di separabilità fra classi. La silhouette confronta la distanza media di un punto dai punti della propria classe con quella dai punti della classe più vicina. L'accuratezza kNN classifica ogni punto guardando la classe maggioritaria dei suoi vicini più prossimi. Entrambe dicono quanto le classi risultano ben separate nello spazio delle rappresentazioni. La trappola: sono sensibili alla metrica di distanza scelta e allo sbilanciamento fra classi — con classi sbilanciate, un'accuratezza kNN alta può nascondere una classe minoritaria completamente ignorata.

Un'ultima nota, già anticipata al §4.1 del documento principale ma che vale la pena ripetere qui: la cosine similarity, su cui si basano sia l'anisotropia sia in parte la CKA, **scarta la norma dei vettori** — e la norma porta informazione. In particolare, qui l'implicit bias della cross-entropy la fa crescere durante l'addestramento, come ha visto in E-5. Riporti sempre le norme separatamente, mai il solo coseno.

**E-12.** Implementi `effective_rank(X)` e la testi su tre casi di cui conosce la risposta: colonne i.i.d. gaussiane indipendenti (→ *d*); matrice di rango 1 (→ 1); *K* cluster puntiformi in posizione generica (→ ?). **Prediciamo il terzo caso prima di misurarlo**: la discrepanza fra predizione e risultato è la parte istruttiva dell'esercizio.

**E-13.** Calcoli tutte le misure sulle rappresentazioni pre-addestrate congelate, ciascuna con distribuzione nulla per permutazione (1000 ripetizioni). Una sola tabella: misura, valore, mediana nulla, intervallo al 95%, p-value. È la riga *t* = 0 di tutte le tabelle della tesi.

**Criterio di uscita:** la tabella di E-13 e una figura UMAP delle rappresentazioni pre-addestrate colorata per classe. Sa dire, per ogni numero, se è distinguibile dal caso.

---

## 7. Fase 4 — Fine-tuning e misure (settimane 6–9)

**Obiettivo:** le curve che rispondono a H1, H2, H3.

### 7.1 I quattro regimi

Prima della tabella, in parole semplici, riprendendo il vocabolario del §0.1 (encoder, testa, pre-addestramento, fine-tuning): abbiamo un encoder già pre-addestrato in modo auto-supervisionato (§5.3) e una testa da attaccarci per il compito di classificazione. I quattro regimi sono quattro modi diversi di combinare le due cose durante il fine-tuning, scelti apposta perché ciascuno isola una possibile causa del fenomeno che si vuole misurare. Non sono quattro esperimenti indipendenti: sono quattro condizioni di controllo l'una per l'altra, e vanno lette come tali.

| regime      | cosa si addestra                                           | cosa isola                                                                                                            |
| ----------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **HEAD**    | solo la testa, encoder congelato                           | controllo: l'encoder non cambia per costruzione. Tutto ciò che si muove su `z_cls` in questo regime è artefatto o bug |
| **FULL**    | tutto, dall'inizio                                         | il fine-tuning standard, l'oggetto di studio                                                                          |
| **LP-FT**   | prima HEAD fino a convergenza, poi FULL da quel checkpoint | la ricetta di Kumar et al.: preserva le rappresentazioni?                                                             |
| **SCRATCH** | tutto, encoder inizializzato a caso                        | risponde a Q-2: il collasso è dovuto al fine-tuning o al semplice addestramento supervisionato?                       |

In dettaglio, uno per uno:

- **HEAD** — l'encoder pre-addestrato resta **congelato** (i suoi pesi non vengono più toccati); si addestra solo la testa sopra le rappresentazioni che produce. Poiché per costruzione `z_cls` non può cambiare, questo regime è il **controllo di sanità**: se in HEAD qualche misura sull'encoder si muove, non abbiamo scoperto nulla sul fine-tuning, abbiamo trovato un bug nel codice.
- **FULL** — il regime di fine-tuning "normale": encoder e testa si addestrano insieme, dall'inizio. È l'oggetto di studio vero e proprio: è ciò che fa chiunque faccia fine-tuning senza precauzioni particolari.
- **LP-FT** (*linear probing then fine-tuning*, la ricetta di Kumar et al., §1.1) — prima si fa un regime HEAD fino a convergenza, poi si "sblocca" l'encoder e si continua in regime FULL da quel checkpoint. L'idea è che allenare prima solo la testa eviti i gradienti grandi e disordinati che una testa casuale produrrebbe se applicati fin da subito anche all'encoder.
- **SCRATCH** — identico a FULL, ma l'encoder parte da pesi **casuali** invece che pre-addestrati. Serve a rispondere alla Q-2: se il collasso del rango si osserva anche qui, non è una conseguenza specifica del *fine-tuning* di un modello pre-addestrato, ma del semplice addestramento supervisionato in sé — e questo cambia tutta l'interpretazione dei risultati.

HEAD è il controllo più prezioso che ha ed è anche il suo test di sanità: se in regime HEAD la CKA (§0.6) fra `z_cls`(*t*) e `z_cls`(0) non è esattamente 1, abbiamo un bug. Lo verifichi prima di qualunque altra cosa.

SCRATCH è il controllo che chiuderà la bocca al revisore più severo, e ci costa un run in più.

### 7.2 Iperparametri, senza barare

Il learning rate è l'unico da curare davvero. Scansione su {1e-4, 3e-4, 1e-3, 3e-3} **sul validation set**, poi lo fissiamo una volta per tutte e lo usiamo identico per tutti i regimi, tutti i *K*, tutti i seed. Se scegliessimo il migliore per ciascun regime confrontemmo regimi *e* iperparametri insieme, e non potremmo attribuire le differenze.

Ricordiamo il §3.3, punto 2: **niente early stopping, molte epoche, training set piccolo abbastanza da poter essere interpolato.** Verifichi esplicitamente di aver raggiunto l'accuratezza di training ≈ 100% e **riporti l'epoca in cui accade**: è un dato, non un dettaglio.

### 7.3 Checkpoint: salvi le rappresentazioni, non i pesi

Errore che fanno quasi tutti: salvare quaranta checkpoint di pesi e poi doverli ricaricare tutti. Faccia invece così: a intervalli prefissati estragga e salvi le rappresentazioni del `probe set`. Sono poche centinaia di kB per checkpoint, e sono già nel formato che le serve per le analisi.

**La spaziatura dei checkpoint è logaritmica, non uniforme.** Le cose interessanti succedono nei primi passi: una testa inizializzata a caso produce gradienti grandi che distorcono l'encoder proprio all'inizio, ed è esattamente il meccanismo descritto da Kumar et al. Campioni ai passi 0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, … Se campiona ogni 100 passi si perde il fenomeno più interessante della tesi.

**E-14.** Scriva il ciclo di addestramento con lo snapshot e verifichi che (i) non altera il risultato — stessa loss finale con e senza snapshot, stesso seed; (ii) ripristina correttamente lo stato `train`/`eval` del modello. Perché (i) potrebbe fallire, se lo snapshot è scritto male?

### 7.4 La griglia e le misure

Regimi × *K* × seed = 4 × 4 × 3 = 48 run. A pochi minuti l'uno, è mezza giornata di calcolo. **Questo è l'intero motivo per cui abbiamo scelto un modello piccolo.**

Per ogni checkpoint calcoli:

**(a) Prestazione.** F1 pesato su train e test. È l'asse dei tempi semantico: tutte le altre curve si leggono rispetto al momento in cui questa raggiunge il plateau.

**(b) Geometria** (sul training set): rango effettivo di `h_pen`, `z_cls`, `z_mean`; NC1; anisotropia; norma media.

*La verifica di H1 nella forma forte:* grafico del plateau del rango effettivo di `h_pen` contro *K*, con i quattro punti *K* ∈ {2, 5, 10, 20}, barre d'errore su tre seed, sovrapposto alla retta *y* = *K* − 1. Se i punti ci cadono sopra abbiamo una conferma non banale; se non lo fanno, abbiamo un risultato negativo pulito e quantificato. In entrambi i casi abbiamo la figura principale della tesi.

**(c) Informazione riutilizzabile.** Tre famiglie:

- *(c1) Probe per attributi ausiliari*, cioè attributi **non** presenti fra le etichette del fine-tuning. Con *K* = 2 sono gratis: le 20 classi applicative. Ne aggiunga di ricavabili dalle 48 statistiche, discretizzate in bin: protocollo di trasporto, durata del flusso, byte totali, numero di pacchetti, interarrivo medio. Etichette che non costano nulla e generano un'intera tabella.

- *(c2) Allineamento con le feature esperte:* CKA(`z_cls`(*t*), *F*) dove *F* sono le 48 statistiche. È la metrica di *metric alignment* usata nella letteratura sulla valutazione intrinseca dei modelli fondazionali per il traffico, dove sui modelli reali vale circa 0,09–0,15. La domanda: il fine-tuning la aumenta — il modello impara feature più simili a quelle di un esperto — o la riduce, perché butta via tutto ciò che non serve al compito?

- *(c3) Deriva dall'inizio:* CKA(`z_cls`(*t*), `z_cls`(0)), misura diretta della *feature distortion*. Predizione: FULL deriva molto e presto, LP-FT molto meno, HEAD per niente.

**(d) H3.** Sovrapponga (a) e (c1) normalizzate. Se esiste una finestra in cui (a) è ≥ 95% del massimo e (c1) è ancora ≥ 95% del valore iniziale, quella finestra è il regime utile. Ne riporti l'ampiezza in passi e in frazione dell'addestramento totale, per ciascun *K*.

### 7.5 L'analisi di potenza — la sezione da non saltare

Se il risultato è «non succede niente», quel risultato vale **solo se ha dimostrato di poter rilevare l'effetto qualora ci fosse**.

Costruisca un caso in cui il collasso c'è per costruzione: training set minuscolo, poche centinaia di flussi, *K* = 2, migliaia di epoche, nessuna regolarizzazione. In quel regime il collasso *deve* comparire. Verifichi che le sue metriche lo vedono, e con quale ampiezza.

Poi faccia la cosa più utile: riduca progressivamente la forza del fenomeno — aumentando i dati, aggiungendo weight decay — finché le metriche non lo distinguono più dal rumore. Quel punto è la **sensibilità** del suo strumento di misura, e va riportato accanto a ogni risultato negativo.

Senza questa sezione, «non ho trovato l'effetto» e «non sono in grado di trovare l'effetto» sono indistinguibili. È la differenza fra una tesi e un esercizio.

### 7.6 Come si legge il risultato

Compili questa tabella **prima** di guardare i dati, e la riempia dopo. Serve a impedirsi di raccontare a posteriori la storia che i dati sembrano suggerire.

| (b) rango effettivo di `h_pen` | (c1) probe ausiliari | lettura |
|---|---|---|
| cala verso *K*−1 | calano dopo il plateau di (a) | H1 + H2 confermate: esiste un regime utile, lo quantifichi |
| cala verso *K*−1 | non calano | collasso confinato nella testa, encoder intatto: buona notizia per il riuso, e risultato interessante |
| non cala | calano | qualcosa distrugge informazione senza ridurre il rango: guardi *quali* direzioni, non solo quante |
| non cala | non calano | in questo regime il fine-tuning non danneggia. Verifichi la fase terminale, poi lo riporti: contraddice un'aspettativa teorica, ed è un risultato |

**Criterio di uscita:** le curve (a)–(d) per almeno FULL e HEAD, *K* ∈ {2, 20}, 3 seed, e il grafico plateau-contro-*K*.

---

## 8. Fase 5 — Robustezza, ed estensione opzionale (settimana 10)

Nell'ordine, e si fermi quando finisce il tempo:

1. **Completare la griglia**: LP-FT, SCRATCH, *K* = 5 e 10.
2. **Analisi di potenza** (§7.5), se non l'ha già fatta. Questa non è opzionale.
3. **Sensibilità alle scelte progettuali:** ripeta l'esperimento principale con *n* = 10 e *n* = 30 pacchetti, e con *d* = 32 e *d* = 128. Se le conclusioni cambiano con la dimensione nascosta, lo deve sapere lei prima del revisore.

---

## 9. Checklist delle trappole

Da rileggere ogni due settimane.

- [ ] Divisione per flusso, non per pacchetto, decisa prima di tutto.
- [ ] Test set toccato una volta sola.
- [ ] Media e deviazione standard della standardizzazione calcolate **solo sul train**.
- [ ] Pre-addestramento auto-supervisionato eseguito **solo sui flussi di train**.
- [ ] `model.eval()` prima di ogni estrazione.
- [ ] NC1 e rango effettivo calcolati sul **training set**.
- [ ] Interpolazione del training set verificata, epoca riportata.
- [ ] Almeno 3 seed per ogni risultato riportato.
- [ ] Baseline a pesi casuali per ogni misura geometrica.
- [ ] Etichette permutate per ogni probe.
- [ ] Checkpoint spaziati logaritmicamente.
- [ ] Stesso learning rate fra i regimi confrontati.
- [ ] `reducer.transform`, non `fit_transform`, per i checkpoint successivi al primo.
- [ ] Nessuna conclusione quantitativa tratta da una figura UMAP.
- [ ] La rappresentazione di cui parla è **sempre** specificata: `z_cls`, `z_mean` o `h_pen`.
- [ ] Norme riportate accanto alle cosine similarity.
- [ ] Padding gestito con maschera esplicita, non con zeri.
- [ ] Commit hash e seed nel `config.yaml` di ogni run.
- [ ] Se cita «sparse autoencoder», disambigua quale dei due significati (§3.6 del documento principale).

---

## 10. Calendario, 12 settimane

| sett. | cosa | consegna verificabile |
|---|---|---|
| 1 | Fase 0: E-1…E-5 | notebook con i cinque esercizi |
| 2 | Fase 1: dati, partizioni, baseline non-neurali (E-6, E-7) | `.parquet` + tabella conteggi + tabella baseline |
| 3 | Modello e ciclo di addestramento supervisionato (E-8, E-9) | il modello addestra e batte il caso |
| 4 | Pre-addestramento auto-supervisionato (E-10) | i due controlli di E-10 superati |
| 5 | Fase 3: strumenti di misura (E-11, E-12, E-13) | tabella *t* = 0 + figura UMAP |
| 6 | Snapshot e primi run: FULL e HEAD, *K* = 20 (E-14) | prime curve (a) e (b) |
| 7 | Griglia: 4 regimi × *K* ∈ {2, 20} × 3 seed | curve complete, controllo HEAD superato |
| 8 | *K* ∈ {5, 10}; curva (c) | **grafico plateau-contro-*K*** |
| 9 | Curva (d), regime utile; analisi di potenza | §7.5 completa |
| 10 | Robustezza; eventuale netem | tabella di sensibilità |
| 11–12 | Scrittura | — |

La scrittura occupa solo due settimane perché i capitoli 3 e 4 vanno scritti **durante** le fasi corrispondenti, non alla fine. Alla settimana 11 devono già esistere in bozza.

**Tesi minima accettabile**, se qualcosa va storto: fasi 0–3 complete, regimi FULL e HEAD, *K* ∈ {2, 20}, 3 seed, curve (a) (b) (c1), analisi di potenza. È già una tesi. Tutto il resto è miglioramento.

**Due punti di controllo espliciti.** Se alla fine della settimana 4 il pre-addestramento non supera il controllo (ii) di E-10, ne parliamo e cambiamo strategia: non insista da sola per settimane. Se alla fine della settimana 7 non ha le curve di base, tagliamo *K* = 5 e 10 e la parte opzionale, e ci concentriamo sulla qualità di ciò che c'è.

---

## 11. Struttura della tesi

Le anticipo l'indice, perché sapere dove va a finire ogni esperimento aiuta a decidere quali fare.

1. **Introduzione** — il problema del riuso delle rappresentazioni; perché il traffico di rete è un caso interessante.
2. **Background** — rappresentazioni distribuite; modelli fondazionali per il traffico; neural collapse; feature distortion. Corto e preciso, non un riassunto del deep learning.
3. **Il conflitto e le ipotesi** — Vaze et al. contro neural collapse; H1, H2, H3. È il cuore concettuale e va scritto **per primo**, non per ultimo.
4. **Metodologia** — il modello in miniatura e la giustificazione della scala (§2.2: si prenda spazio, è la scelta che dovrà difendere); le tre rappresentazioni (§5.2: è materiale originale); metriche, baseline, protocollo.
5. **Risultati** — le curve, il grafico plateau-contro-*K*, la tabella dei regimi, l'analisi di potenza.
6. **Discussione** — cosa significa per chi vuole riusare un encoder per il traffico; limiti, a partire dalla scala; cosa cambierebbe su un modello reale e come lo si verificherebbe.
7. **Conclusioni e lavori futuri** — cosa servirebbe per rispondere alle Tesi A e C del documento principale, che sono i seguiti naturali.

---

## 12. Letture, in ordine, con una domanda ciascuna

Cinque, distribuite su tre mesi. Nessuna opzionale.

1. **Papyan, Han, Donoho**, *Prevalence of neural collapse during the terminal phase of deep learning training*, PNAS 2020. *Domanda:* su quale insieme di dati sono calcolate le figure, e perché è cruciale per lei?
2. **Kumar et al.**, *Fine-tuning can distort pretrained features and underperform out-of-distribution*, ICLR 2022. *Domanda:* il meccanismo che propongono predice che il danno avvenga presto o tardi nell'addestramento? Come lo verifica con i suoi dati?
3. **Vaze et al.**, *Open-set recognition: a good closed-set classifier is all you need?*, ICLR 2022. *Domanda:* è davvero in contraddizione con il neural collapse, o i due lavori parlano di regimi diversi?
4. **Arp et al.**, *Dos and Don'ts of Machine Learning in Computer Security*, USENIX Security 2022. *Domanda:* quali delle dieci trappole si applicano al suo disegno? Ne elenchi almeno tre e dica come le evita.
5. **Chari & Pachter** e la replica di **Kobak et al.** (§6.1).

Se avanza tempo e le è piaciuta la parte geometrica: **Roy & Vetterli**, *The effective rank* (2007), e il capitolo sul lemma di Johnson–Lindenstrauss in un testo di algoritmi randomizzati. Da lì si capisce perché la superposizione del §1.2 del documento principale sia possibile — in alta dimensione esistono esponenzialmente più direzioni quasi-ortogonali che ortogonali — e quindi perché un collasso a rango *K* − 1 sia una catastrofe: in dimensione 4 di direzioni quasi-ortogonali ce ne sono pochissime.

---

## 13. Glossario dei termini nuovi

Solo quelli assenti dal glossario del documento principale.

**SPLT** (*sequence of packet lengths and times*) — rappresentazione di un flusso come sequenza dei primi *n* pacchetti, ciascuno descritto da direzione, dimensione e tempo di interarrivo. Non contiene payload né indirizzi.

**Fase terminale dell'addestramento (TPT)** — la porzione di addestramento successiva all'azzeramento dell'errore sul training set. È il regime in cui si manifesta il neural collapse.

**NC1** — indice di collasso della variabilità intra-classe, Tr[Σ_B^† Σ_W], con Σ_W covarianza intra-classe, Σ_B inter-classe, † pseudoinversa di Moore-Penrose.

**Rango effettivo (erank)** — esponenziale dell'entropia dello spettro normalizzato di una matrice: exp(−Σ*p_i* log *p_i*) con *p_i* = λ_i / Σλ_j. Misura continua di «quante direzioni sono effettivamente usate». Vale *d* se la varianza è uniforme su tutte le direzioni, 1 se è concentrata su una sola.

**LP-FT** (*linear probing then fine-tuning*) — prima si addestra solo la testa a encoder congelato, poi si sblocca tutto. Riduce la distorsione delle rappresentazioni pre-addestrate.

**Feature distortion** — deformazione delle rappresentazioni pre-addestrate causata dai gradienti grandi che una testa inizializzata a caso produce nei primi passi di fine-tuning.

**Probe set** — sottoinsieme fisso di campioni, mai usato per addestrare, su cui si estraggono le rappresentazioni a ogni checkpoint per poterle confrontare nel tempo.

**CLS token** — posizione speciale aggiunta alla sequenza, la cui rappresentazione in uscita funge da riassunto dell'intera sequenza.

**Analisi di potenza** — verifica che il proprio strumento di misura sarebbe in grado di rilevare l'effetto cercato, se l'effetto ci fosse. Senza di essa un risultato negativo non è interpretabile.

**Addestramento (training)** — il processo iterativo di aggiustamento dei pesi di un modello per minimizzare una funzione di errore su dati di esempio. In questa guida "addestrare" indica sempre questo processo, sia nella forma auto-supervisionata (pre-addestramento) sia in quella supervisionata (fine-tuning).

**Mascheramento (masking)** — tecnica di pre-addestramento auto-supervisionato: si nasconde una parte dell'input (qui, alcune posizioni della sequenza di pacchetti) e si addestra il modello a ricostruirla dal contesto rimanente. È il compito surrogato del §5.3.

**Tokenizzazione** — conversione di un input grezzo in una sequenza di unità discrete (*token*) che un modello può elaborare. netFound tokenizza i campi degli header di protocollo; il nostro modello, più semplice, evita questo passo e lavora direttamente sui valori numerici — direzione, dimensione, interarrivo — del §2.

**Burst** — un gruppo di pacchetti consecutivi vicini nel tempo, tipicamente appartenenti allo stesso scambio applicativo. netFound modella esplicitamente la gerarchia burst → flusso; il nostro modello, più piatto, non lo fa (§2).

**Payload** — il contenuto applicativo di un pacchetto, cioè i dati trasportati al netto delle intestazioni di protocollo. Il nostro modello, come netFound, non lo usa mai: è una scelta di privacy per costruzione (§2).

**Dimensione nascosta (hidden dimension / hidden size)** — la lunghezza del vettore di rappresentazione interno al modello, indicata con *d* nel §5.1. Più è grande, più informazione il modello può in linea di principio codificare per ogni posizione — e più costa calcolarla.

**Plateau** — un tratto di una curva (di accuratezza, di rango effettivo, o di qualunque altra metrica) in cui il valore smette di cambiare in modo apprezzabile all'aumentare del tempo di addestramento. È la parola chiave di questa tesi: si cerca il plateau dell'accuratezza (per stabilire che "il compito è appreso") e il plateau del rango effettivo (per verificare H1).

**Divergenza** — nell'addestramento, il fenomeno per cui la funzione di errore, invece di scendere, cresce senza limite, tipicamente per un learning rate troppo alto (si veda E-2). Da non confondere con la divergenza statistica (per esempio la KL-divergenza), che qui non compare.

**Early stopping** — tecnica che interrompe l'addestramento quando una metrica di validazione smette di migliorare, per evitare overfitting. In questa tesi va **evitata di proposito** negli esperimenti sul neural collapse (§3.3): il fenomeno si manifesta solo continuando ad addestrare oltre la convergenza, cosa che l'early stopping impedirebbe.

**Data snooping** — l'uso, anche indiretto o involontario, di informazione dal validation o test set in scelte che dovrebbero dipendere solo dal training set (standardizzazione, selezione di iperparametri, pre-addestramento). È la trappola metodologica più comune nella letteratura su ML e sicurezza (Arp et al., §3.6 del documento principale) ed è la ragione delle regole del §4.4.