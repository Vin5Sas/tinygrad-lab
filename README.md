# tinygrad-lab
Linear regression trained from scratch with PyTorch tensors and autograd.

## What it shows
- Manual gradient descent loop (forward, loss, backward, update, zero grad)
- Data generator, model and loss kept in separate modules
- pytest suite: shapes, reproducibility, hand-checked loss, and training convergence

## Run
python -m pip install -r requirements.txt
python -m pytest -v
PYTHONPATH=src python -m tinygrad_lab