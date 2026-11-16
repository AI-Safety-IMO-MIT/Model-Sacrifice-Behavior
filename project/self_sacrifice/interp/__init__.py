"""Interpretability of the main scenario: what is in Gemma-4-31B's
activations before and while it decides. Every module that loads the model
needs a GPU with 80 GB and the packages in `requirement-gpu.txt`; the
analyses over saved arrays need only scikit-learn."""

__all__: list[str] = []
