from Find_angles import Find_angles
from Average_rgb import Average_rgb
from Hough_circles import Hough_circles
from Transformation import Transformation


def main():
    imgpath = "src/WSC_sample_good.png"
    #find_angles(imgpath).run()

    img_maskpath = "src/WSC_mask.png"
    #average_rgb(imgpath, img_maskpath).run()

    points = Hough_circles(imgpath, img_maskpath).run()

    Transformation(imgpath, points).run()


if __name__ == "__main__":
    main()