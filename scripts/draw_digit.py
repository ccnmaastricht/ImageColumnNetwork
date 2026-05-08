import cv2
import numpy as np
import torch

from src.utils import *
from column_ode_digits import init_network, run_batch



device = torch.device('cpu')

size_img_flattened = 100
nr_output_classes = 10

# ---- Load network, prepare initial state, time_vec ----
network, time_vec, initial_state = init_network(size_img_flattened, nr_output_classes, 16, device)
network = load_pkl_file('../results/10_digits!/network_post_training_epoch_204.pkl')
initial_tiled = torch.tile(initial_state, (1, 1))
model_predictions = torch.zeros(1, nr_output_classes)


# ---- Canvas setup ----
canvas_size = 300

# Float canvas with values in [0,16]
canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)

drawing = False
prediction_text = "Press 'p' to predict"

# ---- Soft Gaussian brush ----
brush_size = 41
sigma = 10

ax = np.linspace(-(brush_size // 2), brush_size // 2, brush_size)
xx, yy = np.meshgrid(ax, ax)

brush = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

# Normalize brush to max intensity 16
brush = brush / brush.max() * 16.0


# ---- Drawing callback ----
def draw(event, x, y, flags, param):
    global drawing, canvas

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:

            h, w = brush.shape
            half = h // 2

            # Canvas coordinates
            x1 = max(0, x - half)
            y1 = max(0, y - half)
            x2 = min(canvas.shape[1], x + half + 1)
            y2 = min(canvas.shape[0], y + half + 1)

            # Brush coordinates
            bx1 = half - (x - x1)
            by1 = half - (y - y1)
            bx2 = bx1 + (x2 - x1)
            by2 = by1 + (y2 - y1)

            # Add soft brush
            canvas[y1:y2, x1:x2] += brush[by1:by2, bx1:bx2]

            # Clamp values to [0,16]
            canvas = np.clip(canvas, 0, 16)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


cv2.namedWindow("Draw")
cv2.setMouseCallback("Draw", draw)

# ---- Main loop ----
while True:

    # Convert [0,16] -> [0,255] for display
    display = (canvas / 16.0 * 255).astype(np.uint8)

    # Show last prediction
    cv2.putText(display, prediction_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)

    cv2.imshow("Draw", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):

        canvas[:] = 0
        prediction_text = "Cleared"

    elif key == ord('p'):

        # Resize directly to model input size
        img = cv2.resize(
            canvas,
            (10, 10),
            interpolation=cv2.INTER_AREA
        )

        # Optional: inspect what the network sees
        print(np.round(img, 1))

        with torch.no_grad():

            img_flat = (
                torch.tensor(img, dtype=torch.float32)
                .flatten()
                .unsqueeze(0)
            )

            model_predictions, _ = run_batch(
                network,
                time_vec,
                initial_tiled,
                model_predictions,
                img_flat,
                device
            )

            pred = torch.argmax(model_predictions, dim=1).item()

            print(model_predictions)

        prediction_text = f"Pred: {pred}"

    elif key == 27:  # ESC
        break

cv2.destroyAllWindows()

