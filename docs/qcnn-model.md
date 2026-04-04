# Hybrid QCNN model explanation

## Pipeline

$$
\text{Text} \rightarrow \text{Embedding} \rightarrow \text{Projection to } n \text{ qubits} \rightarrow \text{Quantum Convolution} \rightarrow \text{Quantum Pooling} \rightarrow \text{Dense Classifier}
$$

## Quantum design

- 4 qubits by default
- angle embedding for classical feature injection
- parameterized `Rot` gates per qubit
- ring entanglement with `CNOT`
- pooling-inspired controlled rotations (`CRX`, `CRZ`)
- classical softmax output for three sentiment labels

## Why this hybrid design works

The classical embedding stack compresses high-dimensional language representations into a compact latent space. The QCNN then applies a learnable variational circuit over a low-dimensional quantum state, enabling expressive non-linear feature mixing while still running efficiently on the PennyLane simulator.

## Local execution note

The circuit runs on `default.qubit`, making the platform suitable for local experimentation without dedicated quantum hardware.
