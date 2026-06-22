# linear-regression-pytorch-gradient-descent




This project implements a full analysis pipeline for a simple linear regression model trained with PyTorch and Stochastic Gradient Descent (SGD).

The goal is to recover the parameters of a linear system from noisy synthetic data, through a complete workflow including data generation, model definition, training, and evaluation.

The analysis follows a standard machine learning procedure and illustrates how gradient-based optimization can learn the underlying relation:

\[
y = 3x + 2 + \epsilon
\]

where \( \epsilon \) is Gaussian noise.

---

## Data

The dataset is synthetically generated to simulate a controlled regression problem:

- **Input variable (X)**  
  Uniformly sampled values in the range \([0, 10]\)

- **Noise term (\(\epsilon\))**  
  Gaussian noise with zero mean and fixed variance

- **Target variable (y)**  
  Defined as:
  \[
  y = 3x + 2 + \epsilon
  \]

All data are generated directly in Python and converted into PyTorch tensors for training.

---

## Requirements

1. Python version
- Python 3.9 or higher

2. Python packages
- torch
- numpy

3. Standard libraries (no installation required)
- pathlib (optional for file management)

---

## Analysis Pipeline

The analysis is structured into four main stages:

---

### 1. Data Generation

- Creation of a synthetic linear dataset
- Sampling of input values from a uniform distribution
- Addition of Gaussian noise to the target variable
- Conversion of NumPy arrays to PyTorch tensors

---

### 2. Model Definition

- Linear regression model of the form:

\[
y = wx + b
\]

- Implementation using:

```python
torch.nn.Linear(1, 1)

implemented using:

```python
torch.nn.Linear(1, 1)
