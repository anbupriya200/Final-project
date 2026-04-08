import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_models(data, models):
    X_test = data["X_test"]
    y_test = data["y_test"]
    X_test_torch = data["X_test_torch"]

    ae = models["ae"]
    xgb = models["xgb"]

    # 🔷 XGBoost probabilities
    xgb_probs = xgb.predict_proba(X_test)[:, 1]

    # 🔷 Autoencoder error
    ae.eval()
    with torch.no_grad():
        recon = ae(X_test_torch)
        ae_error = torch.mean((X_test_torch - recon) ** 2, dim=1).cpu().numpy()

    # Normalize AE error
    min_err, max_err = np.min(ae_error), np.max(ae_error)

    # 🔥 WADE Dynamic Scoring
    r_prev = 0
    sigma = 1e-8
    eta = 2
    lambda_val = 0

    final_scores = []

    for i in range(len(X_test)):
        rit = ae_error[i]
        rit_norm = (rit - min_err) / (max_err - min_err + 1e-8)

        confidence = abs(xgb_probs[i] - 0.5) * 2
        alpha = confidence
        beta = 1 - confidence

        ret = xgb_probs[i] if y_test[i] == 1 else (1 - xgb_probs[i])

        rt = alpha * xgb_probs[i] + beta * rit_norm + ret

        rho = (rt - r_prev) / min(abs(rt + sigma), abs(r_prev + sigma))
        x = rho + np.sign(rt - r_prev)
        kx = np.arctan((x * np.pi) / (2 * eta))
        r_star = rt + (kx - lambda_val) * abs(rt)

        final_scores.append(r_star)
        r_prev = rt

    final_scores = np.array(final_scores)

    # 🔥 Dynamic Threshold
    precision, recall, thresholds = precision_recall_curve(y_test, final_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print(f"\nDynamic Threshold: {best_threshold:.6f}")

    y_pred = (final_scores >= best_threshold).astype(int)

    # ================= METRICS =================
    print("\n===== RESULTS =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, final_scores))

    # ================= SAVE CSV =================
    output = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred,
        "XGB_Prob": xgb_probs,
        "AE_Error": ae_error,
        "Final_Score": final_scores
    })
    output.to_csv("Dynamic_WADE_Output.csv", index=False)

    print("\n✅ Saved: Dynamic_WADE_Output.csv")

    # ================= CONFUSION MATRIX =================
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    print("✅ Saved: confusion_matrix.png")

    # ================= ROC CURVE =================
    fpr, tpr, _ = roc_curve(y_test, final_scores)

    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.close()

    print("✅ Saved: roc_curve.png")

    # ================= PRECISION-RECALL CURVE =================
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig("precision_recall_curve.png")
    plt.close()

    print("✅ Saved: precision_recall_curve.png")

    # ================= FEATURE IMPORTANCE =================
    importances = xgb.feature_importances_

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importances)), importances)
    plt.title("Feature Importance (XGBoost)")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.savefig("feature_importance.png")
    plt.close()

    print("✅ Saved: feature_importance.png")

    # 🔥 Save feature importance as CSV
    fi_df = pd.DataFrame({
        "Feature_Index": list(range(len(importances))),
        "Importance": importances
    })
    fi_df.to_csv("feature_importance.csv", index=False)

    print("✅ Saved: feature_importance.csv")