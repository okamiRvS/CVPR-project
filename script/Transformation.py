from scipy.linalg import null_space, svd
import numpy as np
from matplotlib import pyplot as plt
import cv2

class Transformation():

    X = np.array([
        [.889, 1.7845, 0, 1],  # top-right-inside corner
        [.889, -1.7845, 0, 1],  # bottom-right-inside corner
        [-.889, -1.7845, 0, 1],  # bottom-left-inside corner
        [-.889, 1.7845, 0, 1],  # top-left-inside corner

        [.894, 1.8345, 0.04, 1],  # top-right-outside corner
        [.894, -1.8345, 0.04, 1],  # bottom-right-outside corner
        [-.894, -1.8345, 0.04, 1],  # bottom-left-outside corner
        [-.894, 1.8345, 0.04, 1],  # top-left-outside corner

        [-0.292, 1.0475, 0, 1],  # yellow ball,
        [0.292, 1.0475, 0, 1],  # green ball
        [0, 1.0475, 0, 1],  # brown ball
        [0, 0, 0, 1],  # blue ball
        [0, -0.89225, 0, 1],  # pink ball
        [0, -1.4605, 0, 1],  # black ball
    ])


    def __init__(self, imgpath, angle_points, point_balls):
        self.imgpath = imgpath
        self.x = np.vstack((
            np.array([
                [903, 55, 1],  # top-right-inside corner
                [1026, 610, 1],  # bottom-right-inside corner
                [255, 610, 1],  # bottom-left-inside corner
                [378, 55, 1]  # top-left-inside corner
            ]),
            angle_points,
            point_balls
        ))

    def run(self):
        self.ProjectionMatrix()

    def ProjectionMatrix(self):
        A = None
        for x_i, X_i in zip(self.x, self.X):
            A = self.compute(x_i, X_i) if A is None else np.concatenate(
                (A, self.compute(x_i, X_i)))

        # createGrid()

        rank = np.linalg.matrix_rank(A)
        print(f"The rank of the matrix is: {rank}")

        p = self.decomposeWithSvd(A)
        print(p)
        p = self.decomposeWithNullSpace(A)
        print(p)

        img = cv2.imread(self.imgpath, cv2.IMREAD_COLOR)
        M, N, _ = img.shape

        B = np.zeros((M,N,3))

        [K, R, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles] = cv2.decomposeProjectionMatrix(p)

        res = np.linalg.inv(K) @ [640, 285, 1]
        print("Linalg inv")
        print(res/res[2])

        C = -np.linalg.inv(p[:,:-1])@p[:,3]
        C = np.append(C,1)
        print("C (camera center):")
        print(C)

        print(p @ C)
        l1 = 1

        inv = np.linalg.inv(K @ R) @ np.array([640, 285, 1]).T
        res = C + l1 * np.append(inv,0)
        # pdb.set_trace()
        print(res)

        plt.imshow(img)
        plt.scatter(self.x[:,0], self.x[:,1], color='red')
        plt.show()

    def createGrid(self, pxstep=20):
        img = cv2.imread('src/WSC_sample_good.png', cv2.IMREAD_COLOR)
        width, height, channels = img.shape

        grid = np.zeros((width, height))
        x = pxstep
        y = pxstep

        # Draw all x lines
        while x < img.shape[1]:
            cv2.line(grid, (x, 0), (x, width), color=(255, 0, 255), thickness=1)
            x += pxstep

        while y < img.shape[0]:
            cv2.line(grid, (0, y), (height, y), color=(255, 0, 255), thickness=1)
            y += pxstep

        pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        pts2 = np.float32([[100, 0], [width-100, 0], [0, height], [width, height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(grid, matrix, (height, width))

        # [blend_images]
        alpha = 0.5
        beta = (1.0 - alpha)
        dst = cv2.addWeighted(img, alpha, warped, beta, 0.0)

        cv2.imshow('dst', dst)
        cv2.imshow("Warped", warped)
        cv2.imshow('Hehe', grid)
        cv2.waitKey(0)


    def compute(self, x_i, X_i):
        x = np.zeros((2, 12))
        x[0, 4:8] = X_i * -1
        x[0, 8:12] = x_i[1] * X_i
        x[1, 0:4] = X_i
        x[1, 8:12] = -x_i[0] * X_i
        return x

    def nullspace(self, A, atol=0.04, rtol=0):
        A = np.atleast_2d(A)
        u, s, vh = svd(A)
        tol = max(atol, rtol * s[0])
        nnz = (s >= tol).sum()
        ns = vh[nnz:].conj().T
        return ns

    def decomposeWithSvd(self, A):
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        #print(u.shape, s.shape, vh.shape)
        p = vh.T[:,-1] 
        p = p.reshape((3, 4))

        res = p @ np.array([-0.292, 1.0475, 0, 1])
        print(p)
        print(res/[res[2]])
        return p

    def decomposeWithNullSpace(self, A):
        p = self.nullspace(A)
        p = p.reshape((3, 4))

        res = p @ np.array([-0.292, 1.0475, 0, 1]).T
        print(p)
        print(res/[res[2]])
        return p