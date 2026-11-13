import torch
from .model import mse_loss

def train(model, X, y, lr=0.1, epochs=500):
    history = []
    for _ in range(epochs):
        loss = mse_loss(model.forward(X), y)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= lr * p.grad
                p.grad.zero_()
        history.append(loss.item())
    return history