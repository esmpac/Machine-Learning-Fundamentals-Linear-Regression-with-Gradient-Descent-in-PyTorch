# linear-regression-pytorch-gradient-descent




This project implements a full analysis pipeline for a simple linear regression model trained with PyTorch and Stochastic Gradient Descent (SGD).

The goal is to recover the parameters of a linear system from noisy synthetic data through a complete workflow including data generation, model definition, training, and evaluation.

The analysis follows a standard machine learning procedure and studies how gradient-based optimization can learn the underlying relation:

\[
y = 3x + 2 + \epsilon
\]

where \( \epsilon \sim \mathcal{N}(0, \sigma^2) \) represents Gaussian noise.

---

## Data

The dataset is synthetically generated to simulate a controlled regression problem.

- **Input variable (X)**: uniformly sampled in the range \([0, 10]\)  
- **Noise term (\(\epsilon\))**: Gaussian noise with zero mean and fixed variance  
- **Target variable (y)**:

\[
y = 3x + 2 + \epsilon
\]

The dataset is generated in NumPy and converted into PyTorch tensors for training.

---

## Requirements

1. Python version
- Python 3.9 or higher

2. Python packages
- torch
- numpy

3. Standard libraries
- pathlib (optional)

---

## Analysis Pipeline

The analysis is structured into four main stages.

---

### 1. Data Generation

- Synthetic dataset creation
- Sampling from a uniform distribution
- Addition of Gaussian noise
- Conversion to PyTorch tensors

---

### 2. Model Definition

The model is a simple linear regression of the form:

\[
\hat{y} = wx + b
\]

implemented using:

\[
\texttt{torch.nn.Linear(1, 1)}
\]

The parameters \(w\) and \(b\) are initialized randomly and optimized during training.

---

### 3. Training Procedure

- Loss function: Mean Squared Error (MSE)

\[
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
\]

- Optimizer: Stochastic Gradient Descent (SGD)  
- Batch size: 32  
- Learning rate: 0.01  
- Epochs: 10  

The optimization follows the update rule:

\[
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}
\]

---

### 4. Parameter Estimation and Inference

- Backpropagation via PyTorch autograd
- Gradient computation of loss w.r.t. parameters
- Iterative update of \(w\) and \(b\)
- Final evaluation on a test input

---

## Training Results

The loss decreases over time:
