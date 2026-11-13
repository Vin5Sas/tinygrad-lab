from .data import make_linear_data
from .model import LinearRegressor
from .train import train

X, y = make_linear_data()
model = LinearRegressor()
history = train(model, X, y)
print(f"loss {history[0]:.4f} -> {history[-1]:.4f}")
print(f"W={model.W.item():.3f} (true 2.0), b={model.b.item():.3f} (true 0.5)")