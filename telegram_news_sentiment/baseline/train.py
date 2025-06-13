import hydra
from omegaconf import DictConfig
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import joblib
from baseline.model import build_pipeline
from preprocessing import preprocess
import pandas as pd
from pathlib import Path
import json

def filter_for_binary_classification(records):
    filtered = []
    for r in records:
        if r["target"] == 0.0:
            r["target"] = 0
            filtered.append(r)
        elif r["target"] == 1.0:
            r["target"] = 1
            filtered.append(r)
    return filtered


@hydra.main(config_path="../../configs", config_name="baseline", version_base="1.3")
def train(cfg: DictConfig):
    with open(cfg.data.train, "r", encoding="utf-8") as f:
        train = [json.loads(line) for line in f]
        train = filter_for_binary_classification(train)

    train_df = pd.DataFrame(train)
    with open(cfg.data.test, "r", encoding="utf-8") as f:
        val = [json.loads(line) for line in f]
        val = filter_for_binary_classification(val)

    val_df = pd.DataFrame(val)   
    model = build_pipeline(cfg)
    model.fit(train_df["text"], train_df["target"])
    preds = model.predict(val_df["text"])
    probas = model.predict_proba(val_df["text"])[:, 1]
    f1 = f1_score(val_df["target"], preds, average="binary")
    pr = precision_score(val_df["target"], preds, average="binary")
    re = recall_score(val_df["target"], preds, average="binary")
    auc = roc_auc_score(val_df["target"], probas)
    print(f"F1: {f1:.4f}, AUC: {auc:.4f}, pr: {pr:.4f}, re: {re:.4f}")
    joblib.dump(model, Path(cfg.model.output_path) / "model.joblib")


if __name__ == "__main__":
    train()
