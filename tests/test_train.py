from tinygrad_lab.data import make_linear_data
from tinygrad_lab.model import LinearRegressor
from tinygrad_lab.train import train

def test_training_reduces_loss():
    X, y = make_linear_data()
    history = train(LinearRegressor(), X, y)
    assert history[-1] < history[0]
    assert history[-1] < 0.05

def test_training_recovers_true_parameters():
    X, y = make_linear_data(true_w=2.0, true_b=0.5)
    model = LinearRegressor()
    train(model, X, y)
    assert abs(model.W.item() - 2.0) < 0.1
    assert abs(model.b.item() - 0.5) < 0.1

def test_history_length_and_clean_gradients():
    X, y = make_linear_data()
    model = LinearRegressor()
    history = train(model, X, y, epochs=10)
    assert len(history) == 10
    assert (model.W.grad == 0).all()