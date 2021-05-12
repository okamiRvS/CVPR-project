import numpy as np
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm


filename = 'src/WSC_trimmed_video.mp4'
cap = cv2.VideoCapture(filename)

mask = cv2.imread('src/WSC_mask.png', 0)
n_el = np.sum(mask == 255)

r, g, b = [], [], []

Path("images/").mkdir(parents=True, exist_ok=True)

if cap.isOpened():
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    it = 0
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        it += 1
        pbar.update(1)

        res = cv2.bitwise_and(frame, frame, mask=mask)

        r.append(np.sum(res[:, :, 2] * (mask == 255)) / n_el)
        g.append(np.sum(res[:, :, 1] * (mask == 255)) / n_el)
        b.append(np.sum(res[:, :, 0] * (mask == 255)) / n_el)

        if 115 <= g[-1] <= 135 and r[-1] < 15 and b[-1] < 25:
            cv2.imwrite(f'images/{it}.png', res)
    pbar.close()
    cap.release()

plt.plot(r, color='red')
plt.plot(g, color='green')
plt.plot(b, color='blue')
plt.show()