import torch
class LinearRegressor:
    def __init__(self, n_features=1, n_outputs=1, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(n_features, n_outputs, generator=g, requires_grad=True)
        self.b = torch.zeros(n_outputs, requires_grad=True)

    def forward(self, X):
        return X @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]

def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()