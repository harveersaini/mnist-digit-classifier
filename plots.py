"""Noninteractive plots suitable for local runs and headless environments."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def learning_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ['loss', 'accuracy']):
        for split in ['train', 'val']:
            ax.plot([r['epoch'] for r in history], [r[f'{split}_{metric}'] for r in history],
                    marker='o', label=split)
        ax.set(xlabel='Epoch', ylabel=metric.capitalize())
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluation_plots(metrics, mistakes, out):
    matrix = np.array(metrics['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap='Blues')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=7,
                    color='white' if matrix[i, j] > matrix.max() / 2 else 'black')
    ax.set(xticks=range(10), yticks=range(10), xlabel='Predicted digit', ylabel='True digit',
           title=f"MNIST test accuracy: {metrics['accuracy']:.2%}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / 'confusion_matrix.png', dpi=160)
    plt.close(fig)
    if mistakes:
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        for ax in axes.flat:
            ax.axis('off')
        for ax, (image, truth, pred, confidence) in zip(axes.flat, mistakes):
            ax.imshow(image, cmap='gray')
            ax.set_title(f'True {truth}; predicted {pred}\nscore {confidence:.1%}', fontsize=9)
        fig.suptitle('First test errors in dataset order (up to 25)')
        fig.tight_layout()
        fig.savefig(out / 'misclassified.png', dpi=160)
        plt.close(fig)
