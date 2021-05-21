import numpy as np
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm


class Video_frame_read():

    def __init__(self, videopath, img_maskpath):
        self.videopath = videopath
        self.img_maskpath = img_maskpath

    def run(self):
        self.read_video()

    @staticmethod
    def group_consecutives(vals, step=1):
        """Return list of consecutive lists of numbers from vals (number list)."""
        run = []
        result = [run]
        expect = None
        for v in vals:
            if (v == expect) or (expect is None):
                run.append(v)
            else:
                run = [v]
                result.append(run)
            expect = v + step
        return result

    def read_video(self):
        cap = cv2.VideoCapture(self.videopath)

        mask = cv2.imread(self.img_maskpath, 0)
        n_el = np.sum(mask == 255)

        r, g, b = [], [], []
        frames = []

        Path("images/").mkdir(parents=True, exist_ok=True)

        myFrameNumber = 17000

        # get total number of frames
        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # check for valid frame number
        if myFrameNumber >= 0 & myFrameNumber <= totalFrames:
            # set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, myFrameNumber)

        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc(*'DIVX')

            it = 17000
            pbar = tqdm(total=totalFrames)
            pbar.n = it
            pbar.last_print_n = it
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                it += 1
                pbar.update(1)


                res = cv2.bitwise_and(frame, frame, mask=mask)

                r = np.sum(res[:, :, 2] * (mask == 255)) / n_el
                g = np.sum(res[:, :, 1] * (mask == 255)) / n_el
                b = np.sum(res[:, :, 0] * (mask == 255)) / n_el

                if 120 <= g <= 130 and r < 12 and b < 23:
                    frames.append(it - 1)
                    cv2.imshow("Frame", res)
                    cv2.waitKey(2)
            pbar.close()
            cap.release()

        # plt.plot(r, color='red')
        # plt.plot(g, color='green')
        # plt.plot(b, color='blue')
        for f in Video_frame_read.group_consecutives(frames):
            plt.axvspan(min(f), max(f), color='yellow', alpha=0.4)
        plt.show()
