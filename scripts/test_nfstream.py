from nfstream import NFStreamer

file_pcap = "data/raw/Cridex.pcap"

df = NFStreamer(
    source=file_pcap,
    statistical_analysis=True,
    splt_analysis=20,
    n_dissections=0,
).to_pandas()

print("Estrazione completata!")
print("Numero di flussi:", len(df))
print("Numero di colonne:", len(df.columns))

print("\nColonne:")
print(df.columns.tolist())

print("\nPrime righe:")
print(df.head())
