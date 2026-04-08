# preprocess.py

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_data(csv_path):
    """
    Preprocess the dataset:
    - Load CSV
    - Convert labels to binary
    - Handle missing/infinite values
    - Standardize features
    - Split into train/test
    - Convert to PyTorch tensors
    Returns a dictionary with all outputs
    """
    # 1. Load dataset
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # 2. Detect label column
    label_col = [c for c in df.columns if "label" in c.lower()][0]

    # 3. Convert to binary labels
    df["BinaryLabel"] = df[label_col].apply(lambda x: 0 if str(x).lower() in ["benign", "normal"] else 1)

    # 4. Select numeric features
    X = df.drop(columns=[label_col, "BinaryLabel"], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    y = df["BinaryLabel"]

    # 5. Clean data
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 6. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 7. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # 8. Convert to PyTorch tensors
    X_train_torch = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    # 9. Return dictionary
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "X_train_torch": X_train_torch,
        "X_test_torch": X_test_torch
    }

# ===== MAIN =====
# preprocess.py

# ... preprocess_data function ...

csv_path = "CICIDS_Merged_80K.csv"  # Replace with your file path
data = preprocess_data(csv_path)

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]
X_train_torch = data["X_train_torch"]
X_test_torch = data["X_test_torch"]

print("\n===== MODULE 1: PREPROCESSING OUTPUT =====")
print("\n1. Train Shape:", X_train.shape)
print("2. Test Shape:", X_test.shape)

preview_df = pd.DataFrame(X_train[:5], columns=[f"F{i}" for i in range(X_train.shape[1])])
preview_df["Label"] = y_train[:5]
print("\n3. Sample Features + Labels (First 5 rows):")
print(preview_df)

print("\n4. Sample Labels (First 10):", y_train[:10])

print("\n5. Torch Tensor Sample (first 2 rows):")
print(X_train_torch[:2])

print("\n6. Data Type Check:")
print("X_train type:", type(X_train))
print("Tensor type:", type(X_train_torch))  