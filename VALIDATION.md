# Validation

## MNIST training and evaluation

Successfully trained and evaluated locally on macOS using Apple's MPS backend.

- Training images: 55,000
- Validation images: 5,000
- Training epochs: 5
- Best checkpoint: epoch 5, selected using validation accuracy
- Best validation accuracy: 98.90%
- Test images: 10,000 from the official MNIST test set
- Test accuracy: **99.12%**
- Test macro F1: **0.9911**

| Epoch | Training loss | Validation loss | Validation accuracy |
| --- | --- | --- | --- |
| 1 | 0.1976 | 0.0926 | 97.06% |
| 2 | 0.0724 | 0.0550 | 98.50% |
| 3 | 0.0522 | 0.0453 | 98.74% |
| 4 | 0.0415 | 0.0431 | 98.82% |
| 5 | 0.0348 | 0.0405 | 98.90% |

The selected checkpoint was saved to `outputs/best.pt`.
Evaluation completed successfully and saved reports to `outputs/`.

### Commands

```bash
python3 train.py --epochs 5
python3 evaluate.py
```

## Automated checks

Five tests passed in a separate Python 3.12 CPU build environment.
These checks covered:

- Output probability normalization and checkpoint loading.
- Image preprocessing, centering, and blank-input handling.
- Training parameter updates and evaluation metric accounting.
- Drawing, prediction, and clearing callbacks using a headless canvas.

Synthetic-data integration also verified JSON reports, learning curves,
confusion matrices, and misclassification plots.

## Limitations

The reported accuracy measures performance on MNIST test images.
It does not measure accuracy on mouse drawings, photographs, or other
datasets. Native desktop interaction has not yet been documented as
manually verified.
