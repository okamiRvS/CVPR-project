import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import distance


P = np.array([[3022.22222300000, 571.002888933672, -289.060029847040, 5463.28258321725],
              [0., -1043.81657182601, -2858.99879855856, 2444.69982193140],
              [0., 0.892192013958863, -0.451656296636000, 8.53637903627695]])

stars = np.array([[115, 59],
                  [353, 132],
                  [442, 493],
                  [45, 547]])
center = np.array([np.mean(stars[:, 0]),
                   np.mean(stars[:, 1])])
stars = stars - center
stars = stars / np.max(stars)
stars = np.array([np.append(s, 0) for s in stars])
stars = np.array([np.append(s, 1) for s in stars])

stars_2d = []
for s in stars:
    x = P @ s
    x = x / x[2]
    stars_2d.append(x)
stars_2d = np.array(stars_2d)

'''
img = cv2.imread('../src/WSC_sample_good.png')

for (x, y, _) in stars_2d:
    cv2.circle(img, (int(x), int(y)), 4, (255, 0, 0), 2)

plt.imshow(img)
plt.show()
'''



min_error = float('+inf')

alpha = 0
x, y = 0, 0
rt = np.array([[math.cos(alpha), -math.sin(alpha), 0, x],
               [math.sin(alpha), math.cos(alpha), 0, y],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])
cx = 0
cy = 0
cz = 0
sc = np.array([[cx, 0, 0, 0],
               [0, cy, 0, 0],
               [0, 0, cz, 0],
               [0, 0, 0, 1]])


M_3d = sc @ rt @ stars
M_2d = P @ M_3d
# M_2d = np.array([M_2d[row, :] / M_2d[row, 3] for row in range(M_2d.shape[0])])
M_2d = M_2d[:-1]

points_frame = np.zeros((4, 2))

error = 0
for i, point in enumerate(M_2d.T):
    error += (distance.euclidean(point, points_frame[i, :])) ** 2

if error < min_error:
    min_error = error

# discesa del gradiente per aggiornare alpha, cx, cy, cz per ottenere min_error < threshold
# per ogni frame
