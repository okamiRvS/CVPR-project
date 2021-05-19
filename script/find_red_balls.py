import numpy as np
import cv2
from matplotlib import pyplot as plt
import pdb
import time
from tqdm import tqdm

'''

C = np.array([0, -7.71, 3.67, 1])
P = np.array([[3022.22222300000, 571.002888933672, -289.060029847040, 5463.28258321725],
              [0., -1043.81657182601, -2858.99879855856, 2444.69982193140],
              [0., 0.892192013958863, -0.451656296636000, 8.53637903627695]])

print(P @ C.T)

point = [.889, -1.7845, 0, 1]
x = P @ point
x = x / x[2]
print(x)
'''

t0 = time.time()

img = cv2.imread('../src/WSC_sample_good.png')
mask = cv2.imread('../src/WSC_mask.png', 0)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv_copy = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
x = np.logical_and(hsv[:, :, 1] < 120, hsv[:, :, 2] > 220)

'''
plt.imshow(x, cmap='Greys', interpolation='nearest')
plt.show()

plt.imshow(img)
plt.imshow(x, cmap='Greys', interpolation='nearest', alpha=0.5)
plt.show()
'''

pos_comp = np.where(x)

_, labels = cv2.connectedComponents(x.astype(np.uint8) * 255)

lengths = [np.sum(labels == label) for label in range(1, np.max(labels) + 1)]
for label, length in enumerate(lengths):
    if length > 30:
        labels[labels == (label + 1)] = 0

'''
plt.imshow(labels)
plt.show()
'''

_, _, stats, centroids = cv2.connectedComponentsWithStats(labels.astype(np.uint8))

#pdb.set_trace()

centroids = centroids[1:, :]

t1 = time.time()

'''
plt.imshow(img)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', alpha=0.5, marker='X', s=1)
plt.show()
'''

print(f'Elapsed time: {t1-t0:.4f} sec')

tinte = []
pts = []
for y, x in centroids:
    x, y = int(x), int(y)
    #print(f'{x} x {y}')
    mean = []
    for ch in range(3):
        mean.append(np.average(hsv_copy[x-2:x+3, y-2:y+3, ch]))
        if ch == 0 and mean[-1] > 210:
            pts.append([y, x])
    tinte.append(mean)

plt.plot(tinte)
plt.hlines(210, 0, 30)
plt.legend(['R', 'G', 'B'])
plt.show()

#pdb.set_trace()

pts = np.array(pts)

plt.imshow(hsv_copy)
plt.scatter(pts[:, 0], pts[:, 1], c='white', alpha=0.5, marker='X', s=10)
plt.show()



'''
corr = np.empty(img.shape)
red_ball = cv2.imread('../src/red_ball.png')

for x in tqdm(range(0, img.shape[0] - red_ball.shape[0])):
    for y in tqdm(range(0, img.shape[1] - red_ball.shape[1]), leave=False):
        for ch in range(3):
            #corr[x, y, ch] = np.corrcoef(red_ball[:, :, ch], img[x:(x+red_ball.shape[0]), y:(y+red_ball.shape[1]), ch])
            corr[x, y, ch] = np.sum(np.abs(red_ball[:, :, ch] - img[x:(x+red_ball.shape[0]), y:(y+red_ball.shape[1]), ch]))



plt.imshow(corr)
plt.colorbar()
plt.show()
'''