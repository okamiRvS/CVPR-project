import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import distance
import pdb
import csv
import time
import multiprocessing
import math


class Worker(multiprocessing.Process):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def run(self):
        '''
        imgpath = "../src/WSC_sample_good.png"
        img = cv2.imread(imgpath)
        '''
        frames = []
        pointsVideo = []
        with open(self.filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                tmp = row[0].split(";")
                frames.append(tmp[0])
                p1x, p1y = int(tmp[1][1:]), int(tmp[2][:-1])
                p2x, p2y = int(tmp[3][1:]), int(tmp[4][:-1])
                p3x, p3y = int(tmp[5][1:]), int(tmp[6][:-1])
                p4x, p4y = int(tmp[7][1:]), int(tmp[8][:-1])
                pointsVideo.append([[p1x, p1y], [p2x, p2y], [p3x, p3y], [p4x, p4y]])

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

        min_error = float('+inf')

        alphas = [0, math.pi / 2, math.pi, 3 / 2 * math.pi]

        xs = np.arange(-1, 1.5, 1 / 3)
        ys = np.arange(-1, 1.5, 1 / 3)

        cxs = np.arange(0.5, 2, 1 / 3)
        cys = np.arange(0.5, 2, 1 / 3)
        # cz = 1

        numIterations = len(xs) * len(ys) * len(cys) * len(cxs) * len(alphas)
        timeInformation = 0

        count = 0
        for x in xs:
            for y in ys:
                for cy in cys:
                    for cx in cxs:
                        for alpha in alphas:
                            start_time = time.time()

                            rt = np.array([[math.cos(alpha), -math.sin(alpha), 0, x],
                                           [math.sin(alpha), math.cos(alpha), 0, y],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]])
                            sc = np.array([[cx, 0, 0, 0],
                                           [0, cy, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]])

                            M_3d = []
                            for i in range(0, 4):
                                M_3d.append(sc @ rt @ stars[i].T)

                            M_3d = np.array(M_3d)

                            M_2d = []
                            for i in range(0, 4):
                                M_2d.append(P @ np.array([M_3d[i]]).T)
                                M_2d[i] = (M_2d[i] / M_2d[i][2]).T[0]
                                px, py = int(M_2d[i][0]), int(M_2d[i][1])
                                M_2d[i] = np.array([px, py])
                                # cv2.circle(img, (px, py), 1, (255, 0, 0), 10)

                            errors = np.zeros(len(pointsVideo))
                            for i, points in enumerate(pointsVideo):
                                for j, pointsFrame in enumerate(points):
                                    errors[i] += distance.euclidean(M_2d[j], pointsFrame)

                            errors = np.array(errors)
                            min = np.argmin(errors)
                            max = np.argmax(errors)

                            '''
                            for p in pointsVideo[max]:
                                cv2.circle(img, (p[0], p[1]), 1, (0, 0, 255), 20)
                            for p in pointsVideo[min]:
                                cv2.circle(img, (p[0], p[1]), 1, (0, 255, 255), 15)
                            '''

                            final_time = time.time() - start_time
                            if count != 0:
                                timeInformation += final_time
                                mean_time = timeInformation / count
                                estimated_time = mean_time * (numIterations - count)
                            else:
                                estimated_time = final_time
                            print(f"Iteration: {count}, remainingIteration: {numIterations - count}, timeIteration: "
                                  f"{final_time} seconds, estimated_time: {estimated_time} seconds\n\terror "
                                  f"{errors[min]}, min_index {min}, alpha: {alpha}, (cx,cy): ({cx, cy})\n\n")
                            count += 1
                            if errors[min] < min_error:
                                min_error = errors[min]
                                print(f"New min_error: {min_error}\n")

                            # img = cv2.imread(imgpath)


def split_csv(filename, n):
    filenames = []

    lines = open(filename, 'r').readlines()
    nlines = math.ceil(len(lines) / n)
    for nfile in range(n):
        open(str(f'file{nfile}.csv'), 'w').writelines(lines[nfile * nlines:(nfile + 1) * nlines])
        filenames.append(f'file{nfile}.csv')
    return filenames


if __name__ == '__main__':

    n_proc = multiprocessing.cpu_count() - 1
    filenames = split_csv('../ciccio.csv', n_proc)

    jobs = []
    for i, filename in enumerate(filenames):
        p = Worker(filename)
        jobs.append(p)
        p.start()

    for j in jobs:
        j.join()
