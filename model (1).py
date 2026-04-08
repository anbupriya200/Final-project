import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from xgboost import XGBClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Autoencoder ----------
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# ---------- Training ----------
def build_models(data, epochs=40, lr=0.001):
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_train_torch = data["X_train_torch"]

    input_dim = X_train.shape[1]

    # Train Autoencoder (Normal only)
    ae = Autoencoder(input_dim).to(DEVICE)
    optimizer = optim.Adam(ae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    normal_mask = (y_train == 0)
    X_normal = X_train_torch[normal_mask]

    print(f"Training Autoencoder on {len(X_normal)} normal samples...")

    for epoch in range(epochs):
        recon = ae(X_normal)
        loss = loss_fn(recon, X_normal)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    # Train XGBoost
    print("\nTraining XGBoost...")

    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )

    xgb.fit(X_train, y_train)

    # Save models
    torch.save(ae.state_dict(), "autoencoder.pth")
    xgb.save_model("xgboost.json")

    print("Models saved!")

    return {"ae": ae, "xgb": xgb}