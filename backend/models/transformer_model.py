from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from backend.evaluation.metrics import compute_classification_metrics
from backend.utils.hf_datasets_import import import_hf_datasets

_hf_datasets = import_hf_datasets()
Dataset = _hf_datasets.Dataset


@dataclass
class TransformerEmbeddingProvider:
    """
    Production-grade embedding provider using Sentence Transformers.
    Defaults to 'all-MiniLM-L6-v2' as requested for optimal speed/performance ratio.
    """
    model_name: str = "all-MiniLM-L6-v2"
    device: str = 'cpu'
    batch_size: int = 32

    def __post_init__(self) -> None:
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def fit(self, texts: list[str]) -> 'TransformerEmbeddingProvider':
        del texts
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        # SentenceTransformer handle batching and pooling internally
        return self.model.encode(
            texts, 
            batch_size=self.batch_size, 
            show_progress_bar=True, 
            convert_to_numpy=True
        ).astype(np.float32)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)


class DistilBertBaselineTrainer:
    """
    Fine-tunes DistilBERT for production sentiment classification.
    """
    def __init__(self, model_name: str, output_dir: Path, label_to_id: dict[str, int], device: str = 'cpu') -> None:
        self.model_name = model_name
        self.output_dir = output_dir
        self.label_to_id = label_to_id
        self.id_to_label = {value: key for key, value in label_to_id.items()}
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(label_to_id),
            id2label=self.id_to_label,
            label2id=self.label_to_id,
        ).to(device)

    def _tokenize(self, batch: dict[str, list[Any]]) -> dict[str, Any]:
        return self.tokenizer(batch['text'], truncation=True, padding=False, max_length=256)

    @staticmethod
    def _compute_metrics(payload) -> dict[str, float]:
        predictions = payload.predictions[0] if isinstance(payload.predictions, tuple) else payload.predictions
        predicted_labels = np.asarray(predictions).argmax(axis=1)
        return compute_classification_metrics(predicted_labels.tolist(), payload.label_ids.tolist())

    def train(self, train_df, eval_df, epochs: int = 3, batch_size: int = 32) -> dict[str, float]:
        # Convert to HF Dataset for efficient memory mapped processing
        train_dataset = Dataset.from_pandas(train_df[['text', 'label_id']], preserve_index=False).rename_column('label_id', 'labels')
        eval_dataset = Dataset.from_pandas(eval_df[['text', 'label_id']], preserve_index=False).rename_column('label_id', 'labels')
        
        tokenized_train = train_dataset.map(self._tokenize, batched=True, remove_columns=['text'])
        tokenized_eval = eval_dataset.map(self._tokenize, batched=True, remove_columns=['text'])
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / 'distilbert_checkpoints'),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            learning_rate=2e-5,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            report_to=[],
            fp16=torch.cuda.is_available(),
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=self._compute_metrics,
        )
        
        trainer.train()
        metrics = trainer.evaluate()
        
        final_path = self.output_dir / 'distilbert_production'
        self.model.save_pretrained(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))
        
        return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
