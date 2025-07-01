import numpy as np
import pandas as pd

# Buat data angka 1–20 dan beri label: 1 = ganjil, 0 = genap
data = pd.DataFrame({
    "angka": np.arange(1, 21)
})
data["label"] = data["angka"] % 2

print("Data Awal:")
print(data)

# Inisialisasi untuk 2 client
num_clients = 2
client_data = [[] for _ in range(num_clients)]

# Ambil data label 1 (ganjil) dan label 0 (genap)
for label in [1, 0]:
    df_label = data[data["label"] == label]
    df_label = df_label.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split ke 2 client sesuai urutan (bukan random)
    splits = np.array_split(df_label, num_clients)

    for i in range(num_clients):
        client_data[i].append(splits[i])

# Gabungkan dan reset index
for i in range(num_clients):
    # df_client = pd.concat(client_data[i]).reset_index(drop=True)
    df_client = pd.concat(client_data[i]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nClient {i} - Data:\n{df_client}")
