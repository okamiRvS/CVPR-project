import cv2
import pdb
import numpy as np
from matplotlib import pyplot as plt


def get_red_balls(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    x = np.logical_and(hsv[:, :, 1] < 120, hsv[:, :, 2] > 220)

    pts = []
    r = []
    g = []
    b = []

    _, labels, stats, centroids = cv2.connectedComponentsWithStats(
        x.astype(np.uint8) * 255)
    for label, area in enumerate(stats[:, 4]):
        if area <= 40:
            y, x = centroids[label]
            y = int(y)
            x = stats[label, 1] + stats[label, 3] + 3

            if np.average(frame[x - 2:x + 3, y - 2:y + 3, 0]) > 110 and \
                    np.average(frame[x - 2:x + 3, y - 2:y + 3, 1]) < 55 and \
                    np.average(frame[x - 2:x + 3, y - 2:y + 3, 2]) < 55:
                # r.append(np.average(frame[x - 2:x + 3, y - 2:y + 3, 0]))
                # g.append(np.average(frame[x - 2:x + 3, y - 2:y + 3, 1]))
                # b.append(np.average(frame[x - 2:x + 3, y - 2:y + 3, 2]))
                pts.append([y, x])

    # plt.plot(r, label="r")
    # plt.plot(g, label="g")
    # plt.plot(b, label="b")
    # plt.legend()
    return np.array(pts)


def main():
    filename = './src/output.avi'
    cap = cv2.VideoCapture(filename)

    if cap.isOpened():
        it = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            it += 1

            points = get_red_balls(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for i, (y, x) in enumerate(points):
                cv2.circle(frame, (y, x), 4, (255, 255, 255), 2)
                # (img, text, org, fontFace, fontScale, color
                # cv2.putText(frame, str(i), (y, x),
                #             cv2.FONT_HERSHEY_DUPLEX, .6, (0, 0, 0))

            # pdb.set_trace()

            cv2.imshow('johnny', frame)
            # plt.show()
            cv2.waitKey(1)

        cap.release()


if __name__ == '__main__':
    main()
