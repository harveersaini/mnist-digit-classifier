"""Convert one isolated handwritten digit to a centered MNIST-style tensor."""
import numpy as np
from PIL import Image, ImageOps
import torch
from data import MEAN, STD


def preprocess_image(image, invert=False):
    # Composite transparent files over the expected background before grayscale.
    if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
        rgba = image.convert('RGBA')
        bg = Image.new('RGBA', rgba.size, 'white' if invert else 'black')
        image = Image.alpha_composite(bg, rgba)
    image = image.convert('L')
    if invert:
        image = ImageOps.invert(image)
    array = np.asarray(image)
    ys, xs = np.where(array > 30)
    if not len(xs):
        raise ValueError('No digit found. Draw a bright digit on black, or use --invert for dark ink.')
    crop = image.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    scale = 20 / max(crop.size)
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                       Image.Resampling.LANCZOS)
    canvas = Image.new('L', (28, 28))
    canvas.paste(crop, ((28 - crop.width) // 2, (28 - crop.height) // 2))
    pixels = np.array(canvas, dtype=np.float32) / 255
    yy, xx = np.indices(pixels.shape)
    total = pixels.sum()
    dx = round(13.5 - float((xx * pixels).sum() / total))
    dy = round(13.5 - float((yy * pixels).sum() / total))
    centered = Image.new('L', (28, 28))
    centered.paste(canvas, (dx, dy))  # Paste clips at edges; never wraps pixels.
    pixels = np.array(centered, dtype=np.float32) / 255
    tensor = torch.from_numpy((pixels - MEAN) / STD).unsqueeze(0).unsqueeze(0)
    return tensor, centered
