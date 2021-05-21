import numpy as np
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pdb


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
        inv_mask = 255 - mask

        n_el = np.sum(mask == 255)
        n_el_inv = np.sum(inv_mask == 255)

        r, g, b = [], [], []
        ro, go, bo = [], [], []
        frames = []

        Path("images/").mkdir(parents=True, exist_ok=True)

        myFrameNumber = 25000

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

            it = myFrameNumber
            pbar = tqdm(total=totalFrames)
            pbar.n = it
            pbar.last_print_n = it
            while True:
                ret, frame = cap.read()
                good = False
                if not ret:
                    break
                it += 1
                pbar.update(1)


                res = cv2.bitwise_and(frame, frame, mask=mask)

                r = np.sum(res[:, :, 2] * (mask == 255)) / n_el
                g = np.sum(res[:, :, 1] * (mask == 255)) / n_el
                b = np.sum(res[:, :, 0] * (mask == 255)) / n_el

                if 120 <= g <= 130 and r < 12 and b < 23:
                    inv_res = cv2.bitwise_and(frame, frame, mask=inv_mask)

                    ro = np.sum(inv_res[:, :, 2] * (inv_mask == 255)) / n_el_inv
                    go = np.sum(inv_res[:, :, 1] * (inv_mask == 255)) / n_el_inv
                    bo = np.sum(inv_res[:, :, 0] * (inv_mask == 255)) / n_el_inv

                    if ro > 90 and go < 55 and bo < 50:
                        frames.append(it - 1)
                        good = True


                if good:
                    cv2.circle(res, (50, 50), 10, (0, 255, 0), -1)
                else:
                    cv2.circle(res, (50, 50), 10, (0, 0, 255), -1)

                cv2.imshow("Frame", res)
                cv2.waitKey(1)

            pbar.close()
            cap.release()
        '''
        plt.plot(r, color='red')
        plt.plot(g, color='green')
        plt.plot(b, color='blue')
        
        for f in Video_frame_read.group_consecutives(frames):
            plt.axvspan(min(f), max(f), color='yellow', alpha=0.4)
        
        plt.show()
        '''