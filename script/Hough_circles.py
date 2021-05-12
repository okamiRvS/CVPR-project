import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class Hough_circles():

    def __init__(self, imgpath, img_maskpath):
        self.imgpath = imgpath
        self.img_maskpath = img_maskpath

    def run(self):
        self.HoughCircles()

    def HoughCircles(self):
        mask = cv.imread(self.img_maskpath, 0)

        src = cv.imread(cv.samples.findFile(self.imgpath), cv.IMREAD_COLOR)
        src = cv.bitwise_and(src, src, mask=mask)

        ycbcr = cv.cvtColor(src, cv.COLOR_BGR2YCrCb)
        #yuv = cv.cvtColor(src, cv.COLOR_BGR2YUV)
        #hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)

        gray = ycbcr[:, :, 0]
        #gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        '''
        plt.imshow(gray)
        plt.show()
        '''

        gray = cv.medianBlur(gray, 5)
        '''
        plt.imshow(gray)
        plt.show()
        '''

        rows = gray.shape[0]
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 20,
                                  param1=100, param2=10,
                                  minRadius=8, maxRadius=12)

        my_circles = circles[0][np.argsort(circles[0][:, 1])]
        balls = np.vstack((my_circles[1],
                           my_circles[2],
                           my_circles[3],
                           my_circles[5],
                           my_circles[9]))

        print(balls[:, 1])
        balls[:, 1] = balls[:, 1] + (balls[:, 2] * 2 / 3)
        print(balls[:, 1])

        plt.imshow(src)
        plt.plot(balls[0, 0], balls[0, 1], 'ro')
        #plt.Circle((int(balls[0,0]), int(balls[0,1])), 1, color='r')
        plt.show()

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv.circle(src, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv.circle(src, center, radius, (255, 0, 255), 3)

        cv.imshow("detected circles", src)
        cv.waitKey(0)

        return balls