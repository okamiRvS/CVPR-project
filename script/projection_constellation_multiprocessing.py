from operator import mul
from typing import final
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

res = {}

def worker(id, queue, filename):
    '''
    imgpath = "../src/WSC_sample_good.png"
    img = cv2.imread(imgpath)
    '''
    frames = []
    pointsVideo = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            tmp = row[0].split(";")
            frames.append(int(tmp[0]))
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

    alphas = [0, math.pi / 4, math.pi / 2, 1/2 * math.pi, math.pi, 5/4 * math.pi, 3/2 * math.pi, 7/4 * math.pi]

    xs = np.arange(-1, 1.5, 1 / 8)
    ys = np.arange(-1, 1.5, 1 / 8)

    cxs = np.arange(0.5, 2, 1 / 8)
    cys = np.arange(0.5, 2, 1 / 8)
    # cz = 1

    numIterations = len(xs) * len(ys) * len(cys) * len(cxs) * len(alphas)
    timeInformation = 0

    min_index = 0
    max_index = 0
    count = 0
    max_error = 0
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
                                # aggiungere un errore anche in base al baricentro se coincide, farlo pesare di più

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

                        print(f"running: Process-{id}\n"
                                f"Iter: {count}, remainingIter: {numIterations - count}, timeIter: {final_time} s"
                                f", estimated_time: {estimated_time} s, min_error: {min_error}\n\terror {errors[min]}"
                                f", min_index {min}, alpha: {alpha}, (cx,cy): ({cx, cy})\n\n")
                        count += 1
                        if errors[min] < min_error:
                            min_error = errors[min]
                            max_error = errors[max]
                            print(f"New min_error: {min_error}\n")
                            final_info = f"running: Process-{id} Iter: {count}, remainingIter: {numIterations - count}, timeIter: {final_time} s, estimated_time: {estimated_time} s, min_error: {min_error}\n\terror {errors[min]}, min_index {min}, alpha: {alpha}, (cx,cy): ({cx, cy})\n\n"
                            min_index = min
                            max_index = max
                            bestM_2d = M_2d
                            bestcx = cx 
                            bestcy = cy
                            bestalpha = alpha
                            bestcount = count

                            #cv2.imshow("img", img)
                            #cv2.waitKey(0)

                        # img = cv2.imread(imgpath)

    res = queue.get()
    res[id] = {"final_info" : final_info, "min_error" : min_error, "max_error" : max_error, "iter" : bestcount, "min" : min_index, "max" : max_index, "alpha" : bestalpha, "cx" : bestcx, "cy" : bestcy, "M_2d" : bestM_2d, "pointsVideoMin" : pointsVideo[min_index], "pointsVideoMax" : pointsVideo[max_index], "frameMin" : frames[min_index], "frameMax" : frames[max_index]}
    queue.put(res)
            
def split_csv(filename, n):
    filenames = []

    lines = open(filename, 'r').readlines()
    nlines = math.ceil(len(lines) / n)
    for nfile in range(n):
        open(str(f'file{nfile}.csv'), 'w').writelines(lines[nfile * nlines:(nfile + 1) * nlines])
        filenames.append(f'file{nfile}.csv')
    return filenames


if __name__ == '__main__':

    n_proc = multiprocessing.cpu_count() #- 1
    filenames = split_csv('../ciccio.csv', n_proc)

    queue = multiprocessing.Queue()
    queue.put(res)
    
    jobs = []
    for i, filename in enumerate(filenames):
        p = multiprocessing.Process(target=worker, args=(i, queue, filename))
        jobs.append(p)
        p.start()

    for j in jobs:
        j.join()

    data = queue.get()
    
    imgpath = "../src/WSC_sample_good.png"

    print(data)

    cap = cv2.VideoCapture("../src/output.avi")
    # get total number of frames
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    for d in data:
        myMinFrameNumber = data[d]["frameMin"]
        
        # check for valid frame number
        #pdb.set_trace()
        if myMinFrameNumber >= 0 and myMinFrameNumber <= totalFrames:
            # set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES,myMinFrameNumber)
        
        ret, img = cap.read()

        #img = cv2.imread(imgpath)
        
        for i in range(0,4):
            px, py = data[d]["M_2d"][i][0], data[d]["M_2d"][i][1]
            cv2.circle(img, (px, py), 1, (255,0,0), 10 )

        for p in data[d]["pointsVideoMin"]:
            cv2.circle(img, (p[0], p[1]), 1, (0,0,255), 20 )

        cv2.imshow(f"imgMin{d}", img)

        '''
        myMaxFrameNumber = data[d]["frameMax"]
        
        # check for valid frame number
        if myMaxFrameNumber >= 0 and myMaxFrameNumber <= totalFrames:
            # set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES,myMaxFrameNumber)
        
        ret, img = cap.read()

        for i in range(0,4):
            px, py = data[d]["M_2d"][i][0], data[d]["M_2d"][i][1]
            cv2.circle(img, (px, py), 1, (255,0,0), 10 )

        for p in data[d]["pointsVideoMax"]:
            cv2.circle(img, (p[0], p[1]), 1, (0,255,255), 15 )
        
        cv2.imshow(f"imgMax{d}", img)
        '''

        print("Best result:")
        print(data[d]["final_info"])

    
    cv2.waitKey(0)
