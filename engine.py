"""Training and evaluation without test-set-based model selection."""
import torch
from torch.nn import functional as F


def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_sum, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(labels)
        correct += (output.argmax(1) == labels).sum().item()
        count += len(labels)
    return {'loss': loss_sum / count, 'accuracy': correct / count}


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, count = 0.0, 0
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    mistakes = []
    for images, labels in loader:
        output = model(images.to(device))
        loss_sum += F.nll_loss(output, labels.to(device), reduction='sum').item()
        predictions = output.argmax(1).cpu()
        confidence = output.exp().max(1).values.cpu()
        confusion += torch.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10)
        for i in (predictions != labels).nonzero().flatten().tolist():
            if len(mistakes) < 25:
                mistakes.append((images[i].squeeze().clone(), int(labels[i]),
                                 int(predictions[i]), float(confidence[i])))
        count += len(labels)
    support = confusion.sum(1)
    recall = confusion.diag().float() / support.clamp_min(1)
    precision = confusion.diag().float() / confusion.sum(0).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    metrics = {'loss': loss_sum / count, 'accuracy': confusion.diag().sum().item() / count,
               'samples': count, 'macro_f1': f1.mean().item(),
               'per_digit': [{'digit': i, 'precision': precision[i].item(),
                              'recall': recall[i].item(), 'f1': f1[i].item(),
                              'support': int(support[i])} for i in range(10)],
               'confusion_matrix': confusion.tolist()}
    return metrics, mistakes
