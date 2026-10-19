# Understand and explain this project

## What happens during training?

`data.py` downloads the official MNIST training partition and splits it into 55,000 training and 5,000 validation examples using a fixed seed. Each image becomes a tensor with shape `[1, 28, 28]`. Normalization computes `(pixel - 0.1307) / 0.3081` after scaling pixels into `[0, 1]`.

`train.py` constructs the model, Adadelta optimizer, and a scheduler that multiplies the learning rate by 0.7 after each epoch. Each minibatch in `engine.py` performs a forward pass, calculates negative log-likelihood loss, clears old gradients, backpropagates, and updates parameters. Dropout is active during training.

After each epoch, the program switches to evaluation mode, disables gradients, and measures validation performance. It saves a checkpoint when validation accuracy improves. The test set is not loaded during training.

## How does the CNN work?

| Stage | Output shape for batch size B |
| --- | --- |
| Input | B × 1 × 28 × 28 |
| 3×3 convolution, 32 channels, ReLU | B × 32 × 26 × 26 |
| 3×3 convolution, 64 channels, ReLU | B × 64 × 24 × 24 |
| 2×2 max pooling and dropout | B × 64 × 12 × 12 |
| Flatten | B × 9216 |
| Linear, ReLU, dropout | B × 128 |
| Linear and log-softmax | B × 10 |

Convolutions learn local stroke patterns. Pooling reduces spatial resolution. Fully connected layers combine these features into class scores. `log_softmax` produces log probabilities, matching `nll_loss`; exponentiating returns probabilities for the chart. The output is a batch of ten class scores, not the digit itself. `argmax` selects the digit with the largest score.

The first dropout uses elementwise `Dropout(0.25)` instead of the screenshot's channelwise `Dropout2d(0.25)`, following the current established MNIST example. This reconstruction is not a byte-for-byte transcription of every screenshot. All modules in the delivered project work together; use its checkpoints rather than unrelated old checkpoints.

## What does evaluation measure?

Accuracy is correct predictions divided by all examples. A confusion matrix counts true digits by row and predicted digits by column. Recall asks how many examples of a digit were recovered; precision asks how many predictions of that digit were correct. F1 combines precision and recall. Macro F1 gives each of the ten classes equal weight. A class with no predicted or true samples receives zero for the otherwise undefined metric in this implementation.

The mistake figure shows the first 25 errors in dataset order, not a cherry-picked collection. Learning curves help identify a gap between training and validation behavior.

## Why preprocess drawings?

A raw 280×280 canvas does not match a 28×28 MNIST image. Preprocessing locates bright foreground pixels, crops empty margins, rescales the digit, and centers it before normalization. White-background input must be inverted. An empty image raises a clear error rather than displaying a misleading prediction.

## Questions to prepare for an interview

- Why use a validation set instead of choosing epochs based on test accuracy?
- Why are there 9,216 inputs to the first fully connected layer?
- What do `model.train()`, `model.eval()`, and `torch.inference_mode()` change?
- Why does `nll_loss` match the model's output?
- How can a model have high MNIST accuracy but struggle with your handwriting?
- Which errors occur most often in your measured confusion matrix?
- What did you change from the tutorial foundation, and what evidence shows it helped?

Be able to reproduce a run, explain these decisions, and discuss limitations before presenting this as a completed resume project.
