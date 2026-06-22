# Linear Regression with PyTorch using Gradient Descent
# Synthetic setup: y = 3x + 2 + noise, learned via Sthocastic Gradient Descent (SDG)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def main():



    # -------------------------
    # 1. DATA GENERATION
    # -------------------------

    # Synthetic linear model with additive Gaussian noise
    # (standard controlled environment to verify convergence behavior)

    np.random.seed(42)

    N = 1000

    X = np.random.rand(N, 1) * 10
    noise = np.random.randn(N, 1) * 2

    y = 3 * X + 2 + noise 



    # Torch expects float tensors for autograd + model compatibility
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)



    dataset = TensorDataset(X_tensor, y_tensor)
    # creates paired samples (X, y_true)
    # each element in the dataset is a (feature, label) pair
    # used later by DataLoader to sample mini-batches during training





    # -------------------------
    # 2. DATALOADER (MINI-BATCH SAMPLING)
    # -------------------------

    # shuffle=True breaks sample ordering correlation across epochs
    # Batch size defines SGD approximation regime (stochastic vs full-batch)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)





    # -------------------------
    # 3. MODEL
    # -------------------------

    # Parametric hypothesis: affine map y = wx + b
    # Initialized randomly; parameters will be fitted via gradient-based updates

    model = nn.Linear(1, 1)





    # -------------------------
    # 4. LOSS + OPTIMIZER
    # -------------------------

    # Mean Squared Error (MSE) loss.
    # Under a Gaussian noise assumption, minimizing MSE is equivalent to maximum likelihood estimation.
    criterion = nn.MSELoss()

    # SGD performs incremental parameter updates using batch gradients
    optimizer = optim.SGD(model.parameters(), lr=0.01)





    # -------------------------
    # 5. TRAINING LOOP
    # -------------------------

    # Each epoch = one full pass over dataset (reshuffled)
    epochs = 10


    # Inner loop = stochastic gradient updates per mini-batch
    for epoch in range(epochs):

        total_loss = 0.0

        for batch_X, batch_y in dataloader:

            # forward pass under current-parameterization
            predictions = model(batch_X)

            # empirical risk on current mini-batch
            # measures discrepancy between predictions and ground truth on this batch   
            loss = criterion(predictions, batch_y)

            # reset accumulated gradients
            # PyTorch accumulates gradients by default, so we clear them before backpropropagation
            optimizer.zero_grad()

            # backpropagation: compute ∂Loss/∂θ for all parameters θ
            loss.backward()

            # parameter update step (SGD)
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")





    # -------------------------
    # 6. LEARNED PARAMETERS
    # -------------------------
    # Final fitted hypothesis parameters after optimization

    w = model.weight.item()
    b = model.bias.item()

    print("\nLearned parameters:")
    print(f"w = {w:.4f}")
    print(f"b = {b:.4f}")

    # -------------------------
    # 7. INFERENCE
    # -------------------------
    # Single-point evaluation under trained hypothesis

    test_x = torch.tensor([[7.0]])
    pred = model(test_x)

    print(f"\nPrediction for x=7: {pred.item():.4f}")


if __name__ == "__main__":
    main()