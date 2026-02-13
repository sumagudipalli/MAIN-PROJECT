
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install transformers datasets accelerate
!pip install scikit-learn pandas numpy tqdm spacy matplotlib seaborn
!pip install nlpaug  # for optional augmentation

# Download spaCy model
!python -m spacy download en_core_web_sm
# ============================================================
# FULL HYBRID ABSA CODE (WITH FULL METRICS, GRAPHS, F2, etc.)
# ============================================================

# -------------------- AUTH --------------------
from huggingface_hub import login
login(token="hftoken")  # optional

# -------------------- IMPORTS --------------------
import os, re, random, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel,
    LlamaModel, BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from torch.optim import AdamW
from tqdm.auto import tqdm

# -------------------- SEED & DEVICE --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------- AMP-SAFE RMSNorm & SwiGLU --------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(orig_dtype)

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

# -------------------- FOCAL LOSS --------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

# -------------------- HYBRID MODEL (unchanged) --------------------
class ABSA_RoBERTa_LLaMA32(nn.Module):
    def __init__(self, num_labels=3, dropout=0.3, tau=0.8):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("roberta-large")
        self.h = self.encoder.config.hidden_size

        bnb = BitsAndBytesConfig(load_in_8bit=True)
        self.llama = LlamaModel.from_pretrained(
            "meta-llama/Llama-3.2-3B",
            quantization_config=bnb,
            device_map="auto"
        )
        for p in self.llama.parameters():
            p.requires_grad = False

        self.llama_proj = nn.Linear(self.llama.config.hidden_size, self.h)
        self.llama_gate = nn.Parameter(torch.tensor(-4.0))

        self.cross_attn = nn.MultiheadAttention(self.h, 16, batch_first=True)
        self.self_attn = nn.MultiheadAttention(self.h, 16, batch_first=True)

        self.gate_mlp = nn.Sequential(
            nn.Linear(self.h * 3, self.h * 4),
            SwiGLU(),
            nn.Linear(self.h * 2, 3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.h, self.h * 4),
            SwiGLU(),
            nn.Linear(self.h * 2, self.h),
            RMSNorm(self.h),
            nn.Dropout(dropout),
            nn.Linear(self.h, num_labels)
        )

        self.norm = RMSNorm(self.h)
        self.tau = tau

    def forward(self, input_ids, attention_mask, dep_distances, aspect_mask):
        seq = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        aspect_rep = (seq * aspect_mask.unsqueeze(-1)).sum(1) / aspect_mask.sum(1, keepdim=True).clamp(min=1e-9)

        cross_out, _ = self.cross_attn(aspect_rep.unsqueeze(1), seq, seq,
                                      key_padding_mask=~attention_mask.bool())
        cross_rep = cross_out.squeeze(1)

        refined, _ = self.self_attn(seq, seq, seq, key_padding_mask=~attention_mask.bool())

        dep_w = torch.exp(-dep_distances / self.tau) * attention_mask.float()
        local_rep = (refined * dep_w.unsqueeze(-1)).sum(1) / dep_w.sum(1, keepdim=True).clamp(min=1e-9)
        global_rep = refined.mean(1)

        gates = torch.sigmoid(self.gate_mlp(torch.cat([local_rep, global_rep, cross_rep], dim=-1)))

        fused = (gates[:, 0:1] * local_rep + gates[:, 1:2] * global_rep + gates[:, 2:3] * cross_rep)

        with torch.no_grad():
            llama_out = self.llama(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            llama_rep = llama_out.mean(1)

        llama_rep = llama_rep.to(self.llama_proj.weight.dtype)
        llama_rep = self.llama_proj(llama_rep)

        fused = fused + torch.sigmoid(self.llama_gate) * llama_rep
        fused = self.norm(fused)

        return self.classifier(fused)

# -------------------- DATASET (unchanged) --------------------
class ABSADataset(Dataset):
    def __init__(self, df, tokenizer, max_len=192):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        enc = self.tok(
            r["sentence"], r["aspect"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)

        asp_ids = self.tok(r["aspect"], add_special_tokens=False)["input_ids"]
        aspect_mask = torch.zeros(self.max_len)
        for j in range(len(ids) - len(asp_ids) + 1):
            if ids[j:j+len(asp_ids)].tolist() == asp_ids:
                aspect_mask[j:j+len(asp_ids)] = 1
                break

        return {
            "input_ids": ids,
            "attention_mask": mask,
            "aspect_mask": aspect_mask,
            "dep_distances": torch.zeros(self.max_len),
            "label": torch.tensor(r["label"], dtype=torch.long)
        }

# -------------------- LOAD DATA --------------------
train = pd.read_csv("/content/restaurants_train_processed_final.csv")
test  = pd.read_csv("/content/Restaurent_test_dataset_shuffled.csv")

label_map = {"positive":0, "negative":1, "neutral":2}
train["label"] = train["polarity"].map(label_map)
test["label"] = test["polarity"].map(label_map)

train["sentence"] = train["Sentence"].str.lower()
train["aspect"]   = train["aspect_term"].str.lower()
test["sentence"]  = test["Sentence"].str.lower()
test["aspect"]    = test["aspect_term"].str.lower()

train_df, val_df = train_test_split(train, test_size=0.15, stratify=train["label"], random_state=42)

tokenizer = AutoTokenizer.from_pretrained("roberta-large")

train_loader = DataLoader(ABSADataset(train_df, tokenizer), batch_size=16, shuffle=True)
val_loader   = DataLoader(ABSADataset(val_df, tokenizer), batch_size=16)
test_loader  = DataLoader(ABSADataset(test, tokenizer), batch_size=16)

# -------------------- TRAINING SETUP --------------------
model = ABSA_RoBERTa_LLaMA32().to(device)

weights = compute_class_weight("balanced", classes=np.array([0,1,2]), y=train_df["label"])
criterion = FocalLoss(alpha=torch.tensor(weights).float().to(device))

optimizer = AdamW([
    {"params": model.encoder.parameters(), "lr": 5e-6},
    {"params": model.llama_proj.parameters(), "lr": 1e-5},
    {"params": [model.llama_gate], "lr": 5e-6},
    {"params": [p for n,p in model.named_parameters() if "encoder" not in n and "llama" not in n], "lr": 3e-5}
])

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=200,
    num_training_steps=len(train_loader) * 30
)

scaler = torch.cuda.amp.GradScaler()

# -------------------- TRACKING LISTS --------------------
val_f1_history = []
val_acc_history = []
best_f1 = 0.0

# -------------------- TRAIN LOOP WITH METRICS --------------------
for epoch in range(30):
    model.train()
    for b in tqdm(train_loader, desc=f"Epoch {epoch+1}/30 [Train]"):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(
                b["input_ids"].to(device),
                b["attention_mask"].to(device),
                b["dep_distances"].to(device),
                b["aspect_mask"].to(device)
            )
            loss = criterion(logits, b["label"].to(device))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    # ---------------- Validation ----------------
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for b in val_loader:
            out = model(
                b["input_ids"].to(device),
                b["attention_mask"].to(device),
                b["dep_distances"].to(device),
                b["aspect_mask"].to(device)
            )
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(b["label"].numpy())

    val_f1 = f1_score(trues, preds, average="macro")
    val_acc = accuracy_score(trues, preds)
    val_f1_history.append(val_f1)
    val_acc_history.append(val_acc)

    print(f"Epoch {epoch+1}/30 - Val Macro F1: {val_f1:.4f} | Val Accuracy: {val_acc:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "best_hybrid_absa.pt")
        print("  >>> New best model saved!")
