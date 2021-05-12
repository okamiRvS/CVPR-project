from Find_angles import Find_angles
from Average_rgb import Average_rgb
from Hough_circles import Hough_circles
from Transformation import Transformation
from Video_frame_read import Video_frame_read

def main():
    # First task
    imgpath = "src/WSC_sample_good.png"
    #find_angles(imgpath).run()

    img_maskpath = "src/WSC_mask.png"
    #points = Hough_circles(imgpath, img_maskpath).run()

    #Transformation(imgpath, points).run()

    # Second task
    #average_rgb(imgpath, img_maskpath).run()

    videopath = "src/WSC.mp4"
    Video_frame_read(videopath, img_maskpath).run()
    

if __name__ == "__main__":
    main()