# Drowsiness Detection Project

This project detects drowsiness (sleeping vs awake) of multiple people in vehicles, predicts ages, and provides a GUI.

## Structure
- data/: Training/validation/test datasets
- notebooks/: Model training notebooks
- models/: Saved models and weights
- src/: Inference scripts
- gui/: Tkinter-based GUI

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Train model using `notebooks/train_drowsiness_model.ipynb`
3. Run GUI: `python gui/app.py`
