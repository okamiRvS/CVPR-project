import numpy as np
import cv2, math
from matplotlib import pyplot as plt


class Find_angles():

    def __init__(self, imgpath):
        self.imgpath = imgpath

    def run(self):
        return self.HoughLine()

    def HoughLine(self):
        A = cv2.imread(self.imgpath)
        w = A.shape[0]
        h = A.shape[1]
        plt.figure(figsize=(4, 3), dpi=120, facecolor='white')
        plt.imshow(cv2.cvtColor(A, cv2.COLOR_BGR2RGB))
        plt.show()

        ## Crop image to snooker table boundaries
        hsv = cv2.cvtColor(A, cv2.COLOR_BGR2HSV)
        H = hsv[:, :, 0].astype(np.float)
        # isolate green
        H = np.zeros((w, h))
        H[np.logical_and(hsv[:, :, 0] > 20, hsv[:, :, 0] < 90)] = 1
        plt.imshow(H)
        plt.viridis()
        plt.colorbar()
        plt.show()

        kernel = np.ones((31, 31), np.uint8)
        erosion = cv2.erode(H, kernel, iterations=1)

        plt.imshow(erosion)
        plt.viridis()
        plt.colorbar()
        plt.show()

        dilation = cv2.dilate(erosion, kernel, iterations=1)

        plt.imshow(dilation)
        plt.viridis()
        plt.colorbar()
        plt.show()

        W = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        Gx = cv2.filter2D(dilation, -1, W, cv2.BORDER_REPLICATE)
        Gy = cv2.filter2D(dilation, -1, W.T, cv2.BORDER_REPLICATE)
        G = np.abs(Gx) + np.abs(Gy)
        G = G.astype('uint8')
        plt.imshow(G)
        plt.show()

        #  Standard Hough Line Transform
        lines = cv2.HoughLines(G, 1, np.pi / 180, 300)

        new_lines = [list(lines[0][0])]
        for [[rho, theta]] in lines:
            insert = True
            for [new_rho, _] in new_lines:
                if abs(new_rho - rho) < 10:
                    insert = False
                    break
            if insert:
                new_lines.append([rho, theta])

        def intersection_pts(l1, l2):
            [rho1, theta1] = l1
            [rho2, theta2] = l2
            A = np.array([[math.cos(theta1), math.sin(theta1)],
                        [math.cos(theta2), math.sin(theta2)]])
            b = np.array([rho1, rho2]).T
            X = np.linalg.solve(A, b)
            return X

        points = []

        # Show results
        plt.imshow(A)
        for i in range(4):
            x = intersection_pts(new_lines[i], new_lines[(i+1) % 4])
            points.append(list(x))
            plt.plot(x[0], x[1], color='red', marker='o', markersize=12)
        plt.show()

        points = np.array(points, dtype=np.int32)

        mask = np.zeros(A.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, pts=[points], color=255)

        plt.imshow(mask)
        plt.show()

        cv2.imwrite('src/WSC_mask.png', mask)
        return np.hstack((points, np.ones((4,1), dtype=np.int32)))