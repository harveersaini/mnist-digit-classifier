"""Predict a digit from a PNG/JPG file."""
import argparse
import json
from PIL import Image
import torch
from preprocess import preprocess_image
from utils import get_device, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', help='One isolated digit, bright on a black background')
    parser.add_argument('--invert', action='store_true', help='Use for dark ink on white')
    parser.add_argument('--checkpoint', default='outputs/best.pt')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--save-processed', help='Optional output PNG for inspecting preprocessing')
    args = parser.parse_args()
    device = get_device(args.device)
    model, _ = load_model(args.checkpoint, device)
    with Image.open(args.image) as image:
        tensor, processed = preprocess_image(image, invert=args.invert)
    if args.save_processed:
        processed.save(args.save_processed)
    with torch.inference_mode():
        probabilities = model(tensor.to(device)).exp().cpu().squeeze()
    print(json.dumps({'digit': int(probabilities.argmax()),
                      'probabilities': probabilities.tolist()}, indent=2))


if __name__ == '__main__':
    main()
