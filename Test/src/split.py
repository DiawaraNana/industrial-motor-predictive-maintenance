import pandas as pd
from sklearn.model_selection import train_test_split

print("Loading raw data...")
df = pd.read_csv('deep_motor_training_data.csv') ## Chaque ligne = un état du moteur à un instant donné. Chaque colonne = une feature mesurée (vibration, température, bins FFT...).

print("Splitting data (60% Train, 20% Validation, 20% Test)...")

# Step 1: Split off 20% for the Test set
df_train_val, df_test = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df['label']
)

# Step 2: Split the remaining 80% into Train (60% of total) and Validation (20% of total)
# (0.25 * 80% = 20%)
df_train, df_val = train_test_split(
    df_train_val, test_size=0.25, random_state=42, stratify=df_train_val['label']
)

print("Saving split datasets to CSV files...")
df_train.to_csv('train.csv', index=False)
df_val.to_csv('val.csv', index=False)
df_test.to_csv('test.csv', index=False)

print(f"Success! Saved:")
print(f" - train.csv ({len(df_train)} rows)")
print(f" - val.csv ({len(df_val)} rows)")
print(f" - test.csv ({len(df_test)} rows)")