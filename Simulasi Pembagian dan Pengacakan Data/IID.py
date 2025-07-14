import numpy as np
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

data = pd.DataFrame({
    "Fitur": np.arange(1, 37)
})
data["label"] = ((data["Fitur"]- 1) % 9)  # label: 1–9

print("\nData Awal:\n", data.to_string(index=False))

# Inisialisasi untuk 2 client
num_clients = 2
client_data = [[] for _ in range(num_clients)]

# Ambil data untuk setiap label dari 1–9
for label in range(0, 9):
    df_label = data[data["label"] == label]
    df_label = df_label.sample(frac=1).reset_index(drop=True) #Pengacakan Index Setiap Round

    # Split ke client secara berurutan
    splits = np.array_split(df_label, num_clients)

    for i in range(num_clients):
        client_data[i].append(splits[i])


print ("\n\n====Hasil Pembagian Data====")
# Menampilkan hasil pembagian data tiap client
for i in range(num_clients):
    df_client = pd.concat(client_data[i]).reset_index(drop=True)
    print(f"\nClient {i+1} - Data:\n{df_client.to_string(index=False)}")