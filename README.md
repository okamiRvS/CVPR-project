# Computer Vision & Pattern Recognition Project
> The last project for Computer Vision & Pattern Recognition course @USI 20/21. 

Link repository: https://github.com/okamiRvS/CVPR-project

#### Contributors

- **[Umberto Cocca](https://github.com/okamiRvS)**

- **[Tommaso Vitali](https://github.com/tommivitali)**

- **[Lorenzo Ferri](https://github.com/LorenzoFerri)**

- **[Lidia Alecci](https://github.com/LidiaU)**

#### Prerequisites
- Python 3
- Download all the source files at the follow links [DATA](https://drive.google.com/drive/folders/1f2SiojF4bl9cYvgPKp8Cj6da0ohv8JRs?usp=sharing) and put into the src folder

#### Installation
To run the main script:
```bash
	git clone https://github.com/okamiRvS/CVPR-project.git
	cd CVPR-project/script
	python3 main.py
```

The goal is to reconstruct the snooker table and balls from side view starting from a video. Specifically, the tasks are:
 1. pre-process video and filter only those frames that show side view;
 2. reconstruct camera position;
 3. reconstruct ball positions;
 4. automatically detect balls;
 5. find 2D patterns among the red balls.

#### Usage

##### Tasks

1) Use the (DLT) camera calibration algorithm to find the position of the camera that generated the picture "WSC sample.png". Use a world coordinate system, where the origin is at the centre of the snooker table, the (positive) x-axis is pointing to the right (w.r.t to what you see in this picture) middle pocket, the (positive) y-axis to the baulk line (that is, up, in the picture), and the (positive) z-axis towards the ceiling. Let us use m (meters) as the unit in all directions. Using the official dimensions of the snooker table and the exact positions of the markings (see https://en.wikipedia.org/wiki/Billiard_table#Snooker_and_English_billiards_tables), you can then work out the world coordinates (in meters) of certain key points (e.g. the spots of the balls, etc.), which you can then use as the X_i. You should try to find the corresponding points x_i in the picture automatically, but if you do not manage, you can also do it manually. Find as many correspondences as possible, using these key points, so that the camera calibration becomes as accurate as possible. Once you have found the camera matrix P, decompose it to find the camera calibration matrix K, as well as the external parameters R and C. We will then know where the camera is located (in world coordinates, in meters), into which direction it is looking, which focal length is used, etc.

We detect the 2d points of the inside and outside corners with different approches. To detect the outside corner we first of all moved from RGB to HSV channel to take the Hue, then we applied a filter to focus only on green values. Afterwards we applied morfology on the picture, to clean a little bit the area, through erosion and then dilation operator to return at the original dimension. Finally we computed the edges and used them as input for the Hough Line Transform (we threw away the non-local maxima with 10 as threshold). For the outside corners we moved from RGB to YCbCr channel and we used the luminance as input for the Hough Circle Transform. The inside corners were computed manually. The follow array represents the homogeneous coordinates of the 2d points.

```bash
[[ 903   55    1] 	# top-right-inside corner
 [1026  610    1] 	# bottom-right-inside corner
 [ 255  610    1] 	# bottom-left-inside corner
 [ 378   55    1] 	# top-left-inside corner
 [ 920   37    1] 	# top-right-outside corner
 [1056  625    1] 	# bottom-right-outside corner
 [ 216  625    1] 	# bottom-left-outside corner
 [ 362   37    1] 	# top-left-outside corner
 [ 548  143    1] 	# yellow ball
 [ 726  143    1] 	# green ball
 [ 638  143    1] 	# brown ball
 [ 640  288    1] 	# blue ball
 [ 638  433    1] 	# pink ball
 [ 638  544    1]]	# black ball
```

<div align="center">
<img src="https://github.com/okamiRvS/CVPR-project/blob/master/src/SnookerPoints.png" >
<p>Fig. 1: This represents the image after we detect the balls and the inside and outside corners of the snooker</p>
</div>

Then we mapped each 2d point in 3d homogeneous coordinates using the information of the official dimensions of the snooker table:

```bash
[[ 0.889    1.7845   0.       1.     ]		# top-right-inside corner
 [ 0.889   -1.7845   0.       1.     ]		# bottom-right-inside corner
 [-0.889   -1.7845   0.       1.     ]		# bottom-left-inside corner
 [-0.889    1.7845   0.       1.     ]		# top-left-inside corner
 [ 0.894    1.8345   0.04     1.     ]		# top-right-outside corner
 [ 0.894   -1.8345   0.04     1.     ]		# bottom-right-outside corner
 [-0.894   -1.8345   0.04     1.     ]		# bottom-left-outside corner
 [-0.894    1.8345   0.04     1.     ]		# top-left-outside corner
 [-0.292    1.0475   0.       1.     ]		# yellow ball
 [ 0.292    1.0475   0.       1.     ]		# green ball
 [ 0.       1.0475   0.       1.     ]		# brown ball
 [ 0.       0.       0.       1.     ]		# blue ball
 [ 0.      -0.89225  0.       1.     ]		# pink ball
 [ 0.      -1.4605   0.       1.     ]] 	# black ball
```

With these points we computed the camera matrix P that describes how a camera maps world points (in 3d) to image points (in 2d).

2) Come up with an algorithm for pre-processing the frames of the whole video "WSC.mp4" in the sense that you should extract all frames from the video that show the table as in the picture "WSC sample.png" and discard all other frames showing the table from a different viewpoint, focusing on a player, or containing advertisement.

To solve this task we have seen that for each frame, where there is the snooker table as top view, the camera is totally fixed (so no camera movement). Therefore, we apply a mask in each frames of the video with the goal of reduce the domain of pixel information. Since the top view of the snooker table is composed for the most of green colour, we averaged the amount of colour in the "WSC sample.png" picture to get a threshold and with a slight range flexibility we used it to to extract only the frames where the table is showed as in "WSC sample.png" picture. We've got approximately 28gb of images and for this reason we rendered them as a video (visibile in the drive folder).

<div align="center">
<img src="https://github.com/okamiRvS/CVPR-project/blob/master/src/Figure_1.png" >
<p>Fig. 2: This plot show us where bad and good frames are visibile in the "WSC_trimmed" as test</p>
</div>
