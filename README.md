# Linear Regression with PyTorch (Gradient Descent)

This project implements a simple **linear regression model from scratch using PyTorch**, trained via **Stochastic Gradient Descent (SGD)** on a synthetic dataset.

The goal is to understand how gradient descent learns a linear relationship from noisy data.

---

## Problem Setup

We generate data from the following underlying model:

\[
y = 3x + 2 + \epsilon
\]

where:
- \(x \in [0, 10]\)
- \(\epsilon \sim \mathcal{N}(0, \sigma^2)\) (Gaussian noise)

The model does not know the true parameters and must learn them from data.

---

## Model

We use a simple linear model:

\[
y = wx + b
\]

implemented using:

```python
torch.nn.Linear(1, 1)
