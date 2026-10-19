# Sources and attribution

- Original project screenshots supplied by the project owner: CNN, training loop, and Matplotlib drawing UI. The delivered repository reconstructs those ideas and fills missing pieces. The Java screenshot is contextual; this repository contains a complete Python/PyTorch implementation, not a Java port.
- [Official PyTorch MNIST example](https://github.com/pytorch/examples/tree/main/mnist): established CNN architecture and training conventions (two convolutions, 9,216-to-128-to-10 classifier, dropout, log-softmax/NLL loss, Adadelta, StepLR). No novelty is claimed for this architecture.
- [MNIST dataset overview](https://yann.lecun.com/exdb/mnist/): handwritten digits by Yann LeCun, Corinna Cortes, and Christopher J. C. Burges. Dataset distribution is handled by torchvision; the dataset is not bundled in this repository.
- [PyTorch installation](https://pytorch.org/get-started/locally/) and [torchvision MNIST documentation](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html).

This repository was reconstructed and expanded with AI assistance. Review, run, and modify it before describing your individual contribution. Added workflow components include validation-based model selection, reusable checkpoint loading, file prediction, error reports, synthetic-data tests, CI, and documentation. These are implemented features, not claims of measured benchmark improvement.
