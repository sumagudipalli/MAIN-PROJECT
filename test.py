import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support
)

# -------------------- CLEAN GPU --------------------
torch.cuda.empty_cache()
gc.collect()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# -------------------- RECREATE MODEL (OOM-SAFE) --------------------
model = ABSA_RoBERTa_LLaMA32()


# Move non-LLaMA modules to GPU
model.encoder.to(device)
model.llama_proj.to(device)
model.classifier.to(device)
model.cross_attn.to(device)
model.self_attn.to(device)
model.gate_mlp.to(device)
model.norm.to(device)

# Move Parameter correctly
model.llama_gate.data = model.llama_gate.data.to(device)


# -------------------- LOAD CHECKPOINT (SKIP LLaMA) --------------------
ckpt = torch.load("best_hybrid_absa.pt", map_location="cpu")

filtered_ckpt = {
    k: v for k, v in ckpt.items()
    if not k.startswith("llama.")
}

model.load_state_dict(filtered_ckpt, strict=False)
model.eval()

print("Model loaded successfully (OOM-safe)")

# -------------------- TEST INFERENCE --------------------
test_preds, test_trues = [], []

with torch.no_grad():
    for b in test_loader:
        logits = model(
            b["input_ids"].to(device),
            b["attention_mask"].to(device),
            b["dep_distances"].to(device),
            b["aspect_mask"].to(device)
        )

        preds = logits.argmax(dim=1).cpu().numpy()
        test_preds.extend(preds)
        test_trues.extend(b["label"].numpy())

# -------------------- METRICS --------------------
class_names = ["positive", "negative", "neutral"]

precision, recall, f1, support = precision_recall_fscore_support(
    test_trues, test_preds, labels=[0,1,2], average=None
)

accuracy = accuracy_score(test_trues, test_preds)
macro_f1 = f1_score(test_trues, test_preds, average="macro")

def fbeta_score(p, r, beta=2):
    return (1 + beta**2) * (p * r) / (beta**2 * p + r + 1e-8)

f2_scores = [fbeta_score(p, r, beta=2) for p, r in zip(precision, recall)]

# -------------------- PRINT RESULTS --------------------
print("\n" + "="*65)
print("FINAL TEST RESULTS")
print("="*65)

print(classification_report(
    test_trues,
    test_preds,
    target_names=class_names,
    digits=4
))

print(f"Overall Accuracy : {accuracy:.4f}")
print(f"Macro F1-score   : {macro_f1:.4f}")

# -------------------- CONFUSION MATRIX --------------------
cm = confusion_matrix(test_trues, test_preds, labels=[0,1,2])

plt.figure(figsize=(3,3))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Test Set)")
plt.show()

# -------------------- FINAL METRICS TABLE --------------------
metrics_df = pd.DataFrame({
    "Class": class_names,
    "Precision": np.round(precision, 4),
    "Recall": np.round(recall, 4),
    "F1-Score": np.round(f1, 4),
    "F2-Score": np.round(f2_scores, 4),
    "Support": support
})

metrics_df.loc[len(metrics_df)] = [
    "Macro Avg",
    round(precision.mean(), 4),
    round(recall.mean(), 4),
    round(macro_f1, 4),
    "-",
    "-"
]

metrics_df.loc[len(metrics_df)] = [
    "Accuracy",
    "-",
    "-",
    "-",
    round(accuracy, 4),
    len(test_trues)
]

print("\nFINAL METRICS TABLE")
print(metrics_df.to_string(index=False))
