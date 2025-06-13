import pytorch_lightning as pl
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class QwenPTuningModule(pl.LightningModule):
    def __init__(self, cfg, train_data, val_data):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base_model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name_or_path, num_labels=cfg.num_labels
        )
        base_model.config.pad_token_id = self.tokenizer.pad_token_id
        lora = LoraConfig(task_type=TaskType.SEQ_CLS)
        self.model = get_peft_model(base_model, lora)
        self.train_data = train_data
        self.val_data = val_data

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def tokenize_batch(self, batch):
        len = self.cfg.max_length
        return self.tokenizer(
            batch["text"], padding=True, truncation=True, max_length=len
        )

    def prepare_dataloader(self, data, shuffle=False):
        tokenized = data.map(self.tokenize_batch, batched=True)
        tokenized = tokenized.rename_column("target", "labels")
        tokenized.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        bs = self.cfg.batch_size
        return DataLoader(tokenized, batch_size=bs, shuffle=shuffle)

    def train_dataloader(self):
        return self.prepare_dataloader(self.train_data, shuffle=True)

    def val_dataloader(self):
        return self.prepare_dataloader(self.val_data)

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        self.log("train/loss", output.loss)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self(**batch)
        preds = torch.argmax(output.logits, dim=1).detach().cpu()
        labels = batch["labels"].detach().cpu()
        self.log("val/f1", f1_score(labels, preds), prog_bar=True)
        output = output.logits.softmax(dim=-1)[:, 1].detach().cpu()
        self.log(
            "val/auc",
            roc_auc_score(labels, output),
            prog_bar=True,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
