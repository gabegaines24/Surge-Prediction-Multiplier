"""
Legacy entrypoint: use `python -m backend.train_model` for the full training + save pipeline.
"""
from .train_model import train_and_save

if __name__ == "__main__":
    train_and_save()