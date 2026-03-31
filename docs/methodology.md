# Methodology

## Objective
Assess whether a hybrid QCNN improves multilingual sentiment analysis compared with classical and transformer baselines under controlled ablations.

## End-to-end pipeline
1. Data ingestion and schema unification.
2. Text preprocessing and multilingual normalization.
3. Embedding with MiniLM (384-d).
4. Projection from 384 to 8 dimensions.
5. QCNN inference (AngleEmbedding + StronglyEntanglingLayers + PauliZ).
6. Classical classification head.
7. Final-only evaluation and ablation reporting.

## Ablation protocol
- E1: Full QCNN.
- E2: No QCNN.
- E3: Reduced QCNN depth.
- E4: No projection.
- E5: Transformer baseline.

## Reporting policy
- Report only final metrics (accuracy, precision, recall, f1).
- Keep multilingual slices in a dedicated JSON file.
- Include qualitative error analysis for low-resource languages.
