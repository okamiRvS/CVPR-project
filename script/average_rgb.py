import cv2
import numpy as np
import time

img = cv2.imread('../src/WSC_sample.png', cv2.IMREAD_COLOR)
mask = cv2.imread('../src/WSC_mask.png', 0)

t0 = time.time()

res = cv2.bitwise_and(img, img, mask=mask)
rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

n_el = np.sum(mask == 255)

avg = [
    np.sum(res[:, :, 2] * (mask == 255)) / n_el,
    np.sum(res[:, :, 1] * (mask == 255)) / n_el,
    np.sum(res[:, :, 0] * (mask == 255)) / n_el,
]

t1 = time.time()

print(f'Average RGB values in sample: {avg}')
print(f'Elapsed time: {t1 - t0:.5f} sec')
