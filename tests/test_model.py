import torch
from tinygrad_lab.data import make_linear_data
from tinygrad_lab.model import LinearRegressor, mse_loss

def test_data_shapes():
    X, y = make_linear_data(n=50)
    assert X.shape == (50, 1) and y.shape == (50, 1)

def test_data_is_reproducible():
    X1, y1 = make_linear_data(seed=1)
    X2, y2 = make_linear_data(seed=1)
    assert torch.equal(X1, X2) and torch.equal(y1, y2)

def test_forward_shape():
    model = LinearRegressor(n_features=3, n_outputs=2)
    assert model.forward(torch.randn(5, 3)).shape == (5, 2)

def test_loss_zero_for_perfect_prediction():
    y = torch.tensor([[1.0], [2.0]])
    assert mse_loss(y, y).item() == 0.0

def test_loss_matches_hand_calculation():
    pred = torch.tensor([[0.3], [0.5], [0.7]])
    truth = torch.tensor([[2.5], [4.5], [6.5]])
    assert abs(mse_loss(pred, truth).item() - 18.16) < 1e-4