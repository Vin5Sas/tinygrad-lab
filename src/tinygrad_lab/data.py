import torch

def make_linear_data(n=100, true_w=2.0, true_b=0.5, noise=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 1, generator=g) * 2 - 1
    noise_t = torch.randn(n, 1, generator=g) * noise
    y = X * true_w + true_b + noise_t
    return X, y