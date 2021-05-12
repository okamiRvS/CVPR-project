from Find_angles import Find_angles
from Average_rgb import Average_rgb
from Hough_circles import Hough_circles
from Transformation import Transformation
from Video_frame_read import Video_frame_read
import numpy as np
import pdb

def main():
    # First task
    imgpath = "../src/WSC_sample_good.png"
    angle_points = Find_angles(imgpath).run()

    img_maskpath = "../src/WSC_mask.png"
    point_balls, radius = Hough_circles(imgpath, img_maskpath).run()

    Transformation(imgpath, angle_points, point_balls).run()

    # Second task
    #Average_rgb(imgpath, img_maskpath).run()

    videopath = "../src/WSC_trimmed.mp4" # test mp4 file
    Video_frame_read(videopath, img_maskpath).run()
    
if __name__ == "__main__":
    main()