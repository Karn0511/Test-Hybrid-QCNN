# Training workflow

1. Run dataset ingestion with `python scripts/run_data.py`.
2. Registered datasets are normalized into the unified schema.
3. Preprocessing applies cleaning, emoji mapping, language detection, stopword removal, and lemmatization.
4. The merged dataset is stratified into train, validation, and test splits.
5. An embedding backend is selected:
   - TF-IDF for lightweight baseline experimentation
   - multilingual DistilBERT embeddings for higher semantic fidelity
6. The hybrid QCNN is trained with progress tracking, checkpointing, throughput logging, and early stopping.
7. Reports and plots are saved into `backend/experiments/results/`.
