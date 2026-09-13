from pathlib import Path
import pandas as pd
from nfstream import NFStreamer


# Cartelle
CARTELLA_RAW = Path("data/raw")
CARTELLA_PROCESSED = Path("data/processed")

# File di output
FILE_OUTPUT = CARTELLA_PROCESSED / "flows.parquet"


def assegna_label(nome_file):
    """
    Assegna:
    - label_20 = classe/applicazione
    - label_2 = benign oppure malware
    """

    nome = nome_file.stem.lower()

    # Classi benigne
    classi_benigne = [
        "bittorrent",
        "facetime",
        "ftp",
        "gmail",
        "mysql",
        "outlook",
        "skype",
        "smb",
        "weibo",
        "worldofwarcraft"
    ]

    # Classi malware
    classi_malware = [
        "cridex",
        "geodo",
        "htbot",
        "miuref",
        "neris",
        "nsis-ay",
        "shifu",
        "tinba",
        "virut",
        "zeus"
    ]

    # Controlliamo prima i nomi delle classi benigne
    for classe in classi_benigne:
        if nome.startswith(classe):
            return classe, "benign"

    # Poi le classi malware
    for classe in classi_malware:
        if nome.startswith(classe):
            return classe, "malware"

    raise ValueError(f"Classe non riconosciuta: {nome_file}")


def estrai_file(file_pcap):
    """
    Estrae i flussi da un singolo PCAP.
    """

    print(f"\nElaborazione: {file_pcap.name}")

    label_20, label_2 = assegna_label(file_pcap)

    df = NFStreamer(
        source=str(file_pcap),
        statistical_analysis=True,
        splt_analysis=20,
        n_dissections=0,
    ).to_pandas()

    # Aggiungiamo le etichette
    df["label_20"] = label_20
    df["label_2"] = label_2

    # Aggiungiamo il nome del file di origine
    df["source_file"] = file_pcap.name

    print(f"  Classe: {label_20}")
    print(f"  Tipo: {label_2}")
    print(f"  Flussi: {len(df)}")

    return df


def main():

    CARTELLA_PROCESSED.mkdir(parents=True, exist_ok=True)

    file_pcap = sorted(CARTELLA_RAW.glob("*.pcap"))

    print(f"Trovati {len(file_pcap)} file PCAP.")

    if len(file_pcap) == 0:
        raise RuntimeError("Nessun file PCAP trovato in data/raw/")

    tutti_i_flussi = []

    for file in file_pcap:
        df = estrai_file(file)
        tutti_i_flussi.append(df)

    print("\nUnione dei dataset...")

    df_finale = pd.concat(
        tutti_i_flussi,
        ignore_index=True
    )

    print(f"Numero totale di flussi: {len(df_finale)}")
    print(f"Numero totale di colonne: {len(df_finale.columns)}")

    # Salvataggio
    print("\nSalvataggio del file Parquet...")

    df_finale.to_parquet(
        FILE_OUTPUT,
        index=False
    )

    print("\nCompletato!")
    print(f"File salvato in: {FILE_OUTPUT}")

    print("\nConteggio per classe:")
    print(df_finale["label_20"].value_counts())

    print("\nConteggio benign/malware:")
    print(df_finale["label_2"].value_counts())


if __name__ == "__main__":
    main()
