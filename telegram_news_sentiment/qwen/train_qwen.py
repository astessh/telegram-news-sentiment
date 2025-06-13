import hydra
import pytorch_lightning as pl
from datasets import Value, load_dataset
from models.qwen_module import QwenPTuningModule
from omegaconf import DictConfig


@hydra.main(config_path="../../configs", config_name="qwen", version_base=None)
def train_qwen(cfg: DictConfig):
    train_ds = load_dataset("json", data_files=cfg.train_path, split="train")
    train_ds = train_ds.map(lambda x: {"target": int(x["target"])})
    train_ds = train_ds.cast_column("target", Value("int64"))
    val_ds = load_dataset("json", data_files=cfg.val_path, split="train")
    val_ds = val_ds.map(lambda x: {"target": int(x["target"])})
    val_ds = val_ds.cast_column("target", Value("int64"))
    model = QwenPTuningModule(cfg, train_ds, val_ds)
    trainer = pl.Trainer(max_epochs=cfg.num_epochs, accelerator="auto")
    trainer.fit(model)


if __name__ == "__main__":
    train_qwen()
