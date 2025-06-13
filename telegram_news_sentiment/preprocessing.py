import re
import json
from pathlib import Path
from omegaconf import DictConfig
import hydra
from sklearn.model_selection import train_test_split

def clean_text(text: str, cfg) -> str:
    if cfg.lowercase:
        text = text.lower()
    if cfg.strip:
        text = text.strip()
    if cfg.remove_punctuation:
        text = re.sub(r"[^\w\s]", "", text)
    if cfg.remove_digits:
        text = re.sub(r"\d+", "", text)
    return text

def tokenize(text: str, method: str) -> list:
    if method == "simple_split":
        return text.split()
    elif method == "nltk":
        import nltk
        nltk.download("punkt")
        from nltk.tokenize import word_tokenize
        return word_tokenize(text)
    else:
        raise ValueError(f"Unknown tokenization method: {method}")

def stratified_split(data, test_size, val_size, seed):
    train_val, test = train_test_split(
        data, test_size=test_size, stratify=[r["target"] for r in data], random_state=seed
    )
    train, val = train_test_split(
        train_val, test_size=val_size / (1 - test_size),
        stratify=[r["target"] for r in train_val], random_state=seed
    )
    return train, val, test

@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
def preprocess(cfg: DictConfig):
    input_path = Path(cfg.input_path)
    output_dir = Path(cfg.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    for r in records:
        text = clean_text(r["text"], cfg.text_cleaning)
        r["tokens"] = tokenize(text, cfg.tokenization.method)

    train, val, test = stratified_split(records, cfg.split.test_size, cfg.split.val_size, cfg.seed)

    for name, subset in zip(["train", "val", "test"], [train, val, test]):
        out_file = output_dir / f"{name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            for r in subset:
                json.dump(r, f, ensure_ascii=False)
                f.write("\n")

if __name__ == "__main__":
    preprocess()
