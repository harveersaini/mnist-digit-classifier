# MNIST Digit Classifier

A PyTorch computer vision project that trains a convolutional neural network to classify handwritten digits (0–9), evaluates its errors, and lets you draw digits in a desktop interface.

Reconstructed from my project screenshots, with an expanded training and evaluation workflow. The CNN follows the established PyTorch MNIST example; this is an applied learning project rather than a novel architecture. See [SOURCES.md](SOURCES.md) for attribution and [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for an explanation of the code.

## Features

- Two convolutional layers, max pooling, dropout, and fully connected classification.
- Seeded 55,000/5,000 training/validation split; checkpoint selection uses validation accuracy.
- Separate evaluation on the official 10,000-image test set.
- JSON metrics, per-digit precision/recall/F1, a confusion matrix, learning curves, and example mistakes.
- PNG/JPG prediction with cropping, resizing, centering, and optional polarity inversion.
- Matplotlib drawing canvas with a probability chart and blank-canvas handling.
- Automatic CPU/CUDA/Apple Silicon MPS selection, tests, and GitHub Actions.

## Setup

Use Python 3.11 or 3.12 in a virtual environment. An internet connection is needed to install dependencies and download MNIST the first time. No pretrained model is bundled: train one before running prediction or the drawing app.

From the extracted project folder:

```bash
python -m venv .venv
```

Activate on macOS/Linux:

```bash
source .venv/bin/activate
```

Activate in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

For CPU-only Linux/Windows installations, install the matched PyTorch pair first:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
```

For CUDA-specific builds, follow the [official PyTorch installer](https://pytorch.org/get-started/locally/). These requirements specify compatible version ranges; they are not an exact environment lock. After a successful run, record your environment with `python -m pip freeze > environment-lock.txt`.

## Train, evaluate, and draw

```bash
python train.py --epochs 5
python evaluate.py
python app.py
```

Training downloads MNIST to `data/`. It saves the best validation checkpoint to `outputs/best.pt`. Evaluation loads that checkpoint and writes reports. The drawing app opens a local desktop window: draw one large digit, click **Predict**, then **Clear** to start again. Test accuracy and handwritten-canvas accuracy are different measures.

A short pipeline check (not a useful final model):

```bash
python train.py --epochs 1 --train-limit 512 --device cpu --output-dir outputs/smoke
python evaluate.py --checkpoint outputs/smoke/best.pt --output-dir outputs/smoke --device cpu
```

Evaluate the final checkpoint only after choosing your training configuration using validation results. Repeatedly selecting configurations using test results makes the test score less independent.

Other training options:

```bash
python train.py --epochs 8 --batch-size 128 --lr 1.0 --seed 42 --device cpu
```

`--device` accepts `auto`, `cpu`, `cuda`, or `mps`; explicit unavailable devices raise an error. Output directories are relative to the working directory. Use separate `--output-dir` values for experiments so you preserve earlier results. Training starts a new run; checkpoint resume is not implemented.

## Predict an image

For a bright digit on black:

```bash
python predict.py digit.png --save-processed processed.png
```

For dark ink on white:

```bash
python predict.py digit.png --invert --save-processed processed.png
```

Use one isolated handwritten digit on a plain background. Preprocessing crops foreground pixels, scales the longest side to 20 pixels, places it in a 28×28 frame, centers its intensity distribution, and applies the same normalization as training. It does not segment multiple digits or remove complex photo backgrounds. Inspect `processed.png` if predictions seem wrong.

Scores are softmax probabilities, not calibrated guarantees of correctness. The model always chooses one of ten digits for nonblank input; letters or scribbles can still receive high scores.

## PyCharm

1. Extract this archive and open **mnist-digit-classifier** as a project.
2. Set the interpreter to a Python 3.11/3.12 virtual environment under **Settings → Project → Python Interpreter** (the menu label may vary).
3. Open PyCharm's terminal in the project root and install the requirements above.
4. Run `train.py`, then `evaluate.py`, then `app.py`. Defaults work without arguments.
5. To change epochs, add `--epochs 5` in the train run configuration's **Parameters** field. Set **Working directory** to the project root.
6. If the drawing window is embedded or unresponsive, disable Matplotlib plots in PyCharm's tool window or run `python app.py` in an external terminal. A desktop GUI backend is required; `Agg` cannot display the drawing window. If necessary, install a supported GUI backend such as PyQt6 in your environment and select `QtAgg` using the `MPLBACKEND` environment variable for the app.

Avoid naming files `torch.py`, `numpy.py`, or `matplotlib.py`, which can shadow the installed packages.

## Files

| File | Purpose |
| --- | --- |
| `model.py` | CNN architecture and forward pass |
| `data.py` | Normalization, train/validation split, test data |
| `engine.py` | Training epoch and evaluation metrics |
| `train.py` | Training CLI, scheduler, checkpoint selection |
| `evaluate.py` | Held-out test evaluation and reports |
| `plots.py` | Saved learning, confusion, and error plots |
| `preprocess.py` | Single-digit preprocessing |
| `predict.py` | Image prediction CLI |
| `app.py` | Interactive drawing interface |
| `utils.py` | Seeds, device selection, JSON, checkpoint loading |
| `tests/` | Offline functional checks using synthetic tensors |

## Results and reproducibility

No final MNIST benchmark result is claimed in this template. Run the full training and evaluation commands and use **your measured values** from `outputs/test_metrics.json`.

| Measure | Your final run |
| --- | --- |
| Test accuracy | Fill after evaluation |
| Macro F1 | Fill after evaluation |
| Selected epoch | Fill from checkpoint/report |
| Seed / epochs / batch size | Record your settings |
| Device and runtime | Record your hardware and elapsed time |

Save the configuration, dependency versions, and reports for any result you publish. Seeds make reruns easier to compare, but exact numerical agreement across devices or dependency versions is not guaranteed. If publishing plots, copy selected outputs into a `results/` directory and update this README to link them. Do not report the small smoke run as final model performance.

## Tests

```bash
python -m pytest -q
```

Tests do not download MNIST. They check probability normalization, checkpoint round trips, image polarity/centering, blank inputs, parameter updates, and metric accounting. The GitHub workflow runs these checks on CPU.

## GitHub

Create an empty GitHub repository (without an initial README), then run these commands inside this project folder. Replace the example URL:

```bash
git init
git add .
git commit -m "Add MNIST training, evaluation, and drawing interface"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/mnist-digit-classifier.git
git push -u origin main
```

The `.gitignore` excludes downloaded data, virtual environments, checkpoints, and generated outputs. Share a model through a GitHub Release if you choose; document which code/config produced it. Select a repository license before publishing if you want to grant reuse rights. Third-party assets retain their own licenses.

## Resume wording

After running, reviewing, and understanding the project, adapt this bullet to match your work:

> Built a PyTorch handwritten-digit classification pipeline with CNN training, validation-based checkpoint selection, per-digit error analysis, and an interactive drawing interface.

After measuring final performance, you can add “achieving **X% accuracy on 10,000 held-out MNIST images**.” Replace X only with your measured result. Credit tutorial foundations and AI-assisted reconstruction where relevant. This project demonstrates image classification; it does not establish demographic fairness or production readiness.

## Limitations and next experiments

MNIST contains standardized grayscale digits. Mouse-drawn digits and phone photos differ in stroke shape, scale, background, and placement. Test those separately. Useful extensions include comparing against a linear baseline, evaluating several seeds, testing rotation/translation robustness on a validation set, adding training augmentation, and measuring confidence calibration. Complete and document an experiment before claiming it as a feature.
