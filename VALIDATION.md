# Delivery validation

Verified in the build environment with Python 3.12, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, NumPy 2.3.5, Pillow 12.3.0, Matplotlib 3.10.8, and pytest 9.1.1.

- Python source compilation: passed.
- `python -m pytest -q`: **5 passed**.
- Tested forward probabilities, state-dictionary checkpoint round trips, input preprocessing, blank inputs, a training update, metric counts, and drawing/prediction/clear callbacks with a headless Matplotlib canvas.
- Synthetic-data integration: training, evaluation, JSON reports, learning curves, confusion matrix, and mistake plots verified.
- Full real-MNIST training/evaluation: **not completed**. The dataset download stalled in this environment. No trained weights or benchmark accuracy are included.
- Native desktop appearance and mouse interaction still require a local graphical session. Callback tests exercise functionality, not physical desktop rendering.

The synthetic checks confirm software paths, not learned digit recognition. To finish validating the project locally, run the README's small MNIST smoke check, then a full training/evaluation run and the drawing app. Keep your actual results and dependency versions.
