"""Draw a digit and inspect its class scores in a Matplotlib desktop window."""
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw
from preprocess import preprocess_image
from utils import get_device, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='outputs/best.pt')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    args = parser.parse_args()
    device = get_device(args.device)
    model, _ = load_model(args.checkpoint, device)
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    image = Image.new('L', (280, 280))
    pen = ImageDraw.Draw(image)
    fig, (canvas_ax, bar_ax) = plt.subplots(1, 2, figsize=(10, 5))
    fig.subplots_adjust(bottom=0.22)
    display = canvas_ax.imshow(image, cmap='gray', vmin=0, vmax=255,
                               extent=(0, 280, 280, 0))
    canvas_ax.set_title('Draw one digit in white')
    canvas_ax.axis('off')
    bars = bar_ax.bar(range(10), np.zeros(10))
    bar_ax.set(xticks=range(10), ylim=(0, 1), xlabel='Digit', ylabel='Model probability')
    title = fig.suptitle('Draw, then click Predict')
    previous = None

    def point(event):
        if event.inaxes is canvas_ax and event.xdata is not None and event.ydata is not None:
            return (max(0, min(279, int(event.xdata))), max(0, min(279, int(event.ydata))))
        return None

    def press(event):
        nonlocal previous
        if event.button == 1:
            previous = point(event)
            if previous:
                x, y = previous
                pen.ellipse((x-9, y-9, x+9, y+9), fill=255)
                display.set_data(np.array(image))
                fig.canvas.draw_idle()

    def move(event):
        nonlocal previous
        current = point(event)
        if previous is not None and current is not None:
            pen.line([previous, current], fill=255, width=18)
            x, y = current
            pen.ellipse((x-9, y-9, x+9, y+9), fill=255)
            previous = current
            display.set_data(np.array(image))
            fig.canvas.draw_idle()
        elif current is None:
            previous = None

    def release(event):
        nonlocal previous
        previous = None

    def clear(event):
        nonlocal previous
        previous = None
        pen.rectangle((0, 0, 280, 280), fill=0)
        display.set_data(np.array(image))
        for bar in bars:
            bar.set_height(0)
        title.set_text('Draw, then click Predict')
        fig.canvas.draw_idle()

    def predict(event):
        try:
            tensor, _ = preprocess_image(image)
        except ValueError:
            title.set_text('Canvas is empty — draw a digit first')
        else:
            with torch.inference_mode():
                probs = model(tensor.to(device)).exp().cpu().squeeze().numpy()
            for bar, score in zip(bars, probs):
                bar.set_height(float(score))
            title.set_text(f'Prediction: {probs.argmax()} | model score: {probs.max():.1%}')
        fig.canvas.draw_idle()

    clear_button = Button(fig.add_axes([0.25, 0.06, 0.18, 0.07]), 'Clear')
    predict_button = Button(fig.add_axes([0.57, 0.06, 0.18, 0.07]), 'Predict')
    clear_button.on_clicked(clear)
    predict_button.on_clicked(predict)
    fig.canvas.mpl_connect('button_press_event', press)
    fig.canvas.mpl_connect('motion_notify_event', move)
    fig.canvas.mpl_connect('button_release_event', release)
    plt.show()


if __name__ == '__main__':
    main()
