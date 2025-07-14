import pandas as pd
import numpy as np

# Buat DataFrame awal
df = pd.DataFrame({
    "Data": ['A', 'B', 'C', 'D', 'E']
})

print("Sebelum Diacak:")
print(df)

# Acak dengan random_state acak
df = df.sample(frac=1).reset_index(drop=True)

print("\nSetelah Diacak:")
print(df)
