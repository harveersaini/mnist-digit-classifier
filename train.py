"""Train a CNN and select a checkpoint using validation accuracy."""
import argparse
from pathlib import Path
import torch
from data import training_loaders
from engine import train_epoch, evaluate
from model import MNISTCNN
from utils import seed_everything, get_device, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--train-limit', type=int, help='Small training subset for a smoke run')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error('epochs, batch-size, and lr must be positive')
    seed_everything(args.seed)
    device = get_device(args.device)
    train, val = training_loaders(args.data_dir, args.batch_size, args.seed, args.train_limit)
    model = MNISTCNN().to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.0
    print(f'Device: {device}; training: {len(train.dataset)}; validation: {len(val.dataset)}')
    for epoch in range(1, args.epochs + 1):
        training = train_epoch(model, train, optimizer, device)
        validation, _ = evaluate(model, val, device)
        history.append({'epoch': epoch, 'train_loss': training['loss'],
                        'train_accuracy': training['accuracy'],
                        'val_loss': validation['loss'], 'val_accuracy': validation['accuracy']})
        if validation['accuracy'] > best:
            best = validation['accuracy']
            torch.save({'architecture': 'MNISTCNN-v1', 'model_state': model.state_dict(),
                        'epoch': epoch, 'validation_accuracy': best, 'config': vars(args),
                        'torch_version': str(torch.__version__)}, out / 'best.pt')
        scheduler.step()
        write_json(out / 'history.json', history)
        print(f"Epoch {epoch}/{args.epochs} | train loss {training['loss']:.4f} | "
              f"val loss {validation['loss']:.4f} | val accuracy {validation['accuracy']:.2%}")
    from plots import learning_curves
    learning_curves(history, out / 'learning_curves.png')
    print(f'Saved best checkpoint to {out / "best.pt"}. Run evaluate.py to measure test performance.')


if __name__ == '__main__':
    main()
