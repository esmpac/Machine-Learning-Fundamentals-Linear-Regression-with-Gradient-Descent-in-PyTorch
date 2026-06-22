# linear-regression-pytorch-gradient-descent




This project implements a full analysis pipeline for a simple linear regression model trained using PyTorch and Stochastic Gradient Descent (SGD).

The goal is to recover the parameters of a linear system from noisy synthetic data through a complete workflow including data generation, model definition, training, and evaluation.

The analysis studies how gradient-based optimization can learn the underlying relation:

$$
y = 3x + 2 + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

---

## Data

The dataset is synthetically generated to simulate a controlled regression problem.

- Input variable \(x\): uniformly sampled in \([0, 10]\)
- Noise term \(\epsilon\): Gaussian noise with zero mean and fixed variance
- Target variable:

$$
y = 3x + 2 + \epsilon
$$

The dataset is generated using NumPy and converted into PyTorch tensors.

---

## Requirements

1. Python version:
- Python 3.9 or higher

2. Python packages:
- torch
- numpy

3. Standard libraries:
- pathlib (optional)

---

## Analysis Pipeline

The analysis is structured into four main stages.

---

### 1. Data Generation

- Sampling \(x \sim \mathcal{U}(0, 10)\)
- Generating noise \(\epsilon \sim \mathcal{N}(0, \sigma^2)\)
- Constructing target variable:

$$
y = 3x + 2 + \epsilon
$$

- Conversion to PyTorch tensors

---

### 2. Model Definition

The model is a linear function:

$$
\hat{y} = wx + b
$$

implemented using:

$$
\text{Linear}(1,1)
$$

---

### 3. Training Procedure

The model is trained by minimizing the Mean Squared Error (MSE):

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

Optimization is performed using Stochastic Gradient Descent:

$$
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}
$$

Parameters:
- batch size = 32  
- learning rate = 0.01  
- epochs = 10  

---

### 4. Parameter Estimation and Inference

Backpropagation computes gradients and updates parameters iteratively until convergence.

The final model is:

$$
\hat{y} = wx + b
$$

---
