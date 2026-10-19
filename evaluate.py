"""Evaluate a saved model once on the official 10,000-image MNIST test set."""
import argparse
from pathlib import Path
from data import test_loader
from engine import evaluate
from utils import get_device, load_model, write_json
from plots import evaluation_plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='outputs/best.pt')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    args = parser.parse_args()
    device = get_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    metrics, mistakes = evaluate(model, test_loader(args.data_dir, 256), device)
    metrics['checkpoint_epoch'] = checkpoint['epoch']
    metrics['training_config'] = checkpoint['config']
    metrics['torch_version'] = checkpoint['torch_version']
    out = Path(args.output_dir)
    write_json(out / 'test_metrics.json', metrics)
    evaluation_plots(metrics, mistakes, out)
    print(f"Test accuracy: {metrics['accuracy']:.2%}; macro F1: {metrics['macro_f1']:.4f}")
    print(f'Reports saved to {out.resolve()}')


if __name__ == '__main__':
    main()
