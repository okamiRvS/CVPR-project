from scipy.linalg import null_space
import numpy as np
import pdb
import cv2

def createGrid(pxstep=20):
    img = cv2.imread('../src/WSC_sample.png', cv2.IMREAD_COLOR)
    width, height, channels = img.shape

    grid = np.zeros((width, height))
    x = pxstep
    y = pxstep

    #Draw all x lines
    while x < img.shape[1]:
        cv2.line(grid, (x, 0), (x, width), color=(255, 0, 255), thickness=1)
        x += pxstep
    
    while y < img.shape[0]:
        cv2.line(grid, (0, y), (height, y), color=(255, 0, 255),thickness=1)
        y += pxstep

    pts1 = np.float32([[0, 0], [width, 0], [0,height], [width, height] ])
    pts2 = np.float32([[100, 0], [width-100, 0], [0,height], [width, height] ])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(grid, matrix, (height, width))

    pdb.set_trace()

    # [blend_images]
    alpha = 0.5
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(img, alpha, warped, beta, 0.0)

    cv2.imshow('dst', dst)
    cv2.imshow("Warped", warped)
    cv2.imshow('Hehe', grid)
    cv2.waitKey(0)



def compute(x_i, X_i):
    x = np.zeros((2, 12))
    x[0, 4:8] = X_i * -1
    x[0, 8:12] = x_i[1] * X_i
    x[1, 0:4] = X_i
    x[1, 8:12] = -x_i[0] * X_i
    return x

x = np.array([
    [360, 48, 1],
    [920, 48, 1],
    [304, 277, 1],
    [975, 277, 1],
    [217, 630, 1],
    [1059, 630, 1],
])

X = np.array([
    [8.89, 1.7845, 0, 1], # top-right-inside corner
    [-8.89, 1.7845, 0, 1], # top-left-inside corner
    [-8.89, -1.7845, 0, 1], # bottom-left
    [8.89, -1.7845, 0, 1], # bottom-right

    [8.94, 1.7895, 0.035, 1], # top-right-outside corner
    [-8.94, 1.7895, 0.035, 1], # top-left-outside corner
    [-8.94, -1.7895, 0.035, 1], # bottom-left-outside corner
    [8.94, -1.7895, 0.035, 1], # bottom-right-outside corner

    [0, 0, 0, 1], # blue ball
    [0, -0.89225, 0, 1], # pink ball
    [0, -1.4605, 0, 1], # black ball
    [0, 1.0475, 0, 1], # brown ball
    [0.292, 1.0475, 0, 1], # green ball
    [-0.292, 1.0475, 0, 1], # yellow ball

])

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

A = None
for x_i, X_i in zip(x, X):
    A = compute(x_i, X_i) if A is None else np.concatenate((A, compute(x_i, X_i)))
    
#createGrid()

rank = np.linalg.matrix_rank(A) 
pdb.set_trace()

p = null_space(A)
p = normalize(p)
p = p.reshape((3, 4))

