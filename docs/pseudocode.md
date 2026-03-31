# QCNN Pseudocode

```text
for text in dataset:
    preprocess(text)
    embedding = MiniLM(text)
    projected = Dense384to8(embedding)
    encoded = AngleEmbedding(projected)
    quantum = StronglyEntanglingLayers(encoded)
    measured = PauliZ(quantum)
    logits = Classifier(measured)
    prediction = argmax(logits)
```
