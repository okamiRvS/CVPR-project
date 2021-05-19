import cv2
import pdb
import numpy as np


def get_red_balls(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    x = np.logical_and(hsv[:, :, 1] < 120, hsv[:, :, 2] > 220)

    pts = []

    _, labels, stats, centroids = cv2.connectedComponentsWithStats(x.astype(np.uint8) * 255)
    for label, area in enumerate(stats[:, 4]):
        if area <= 30:
            y, x = centroids[label]
            x, y = int(x), int(y)
            if np.average(frame[x - 2:x + 3, y - 2:y + 3, 0]) > 210:
                pts.append([y, x])

    return np.array(pts)

def main():
    filename = '../src/output.avi'
    cap = cv2.VideoCapture(filename)

    if cap.isOpened():
        it = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            it += 1

            points = get_red_balls(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for y, x in points:
                cv2.circle(frame, (y, x), 10, (255, 255, 255), -1)

            #pdb.set_trace()

            cv2.imshow('johnny', frame)
            cv2.waitKey(1)

        cap.release()

if __name__ == '__main__':
    main()
