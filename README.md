# Computer Vision & Pattern Recognition Project
> The last project for Computer Vision & Pattern Recognition course @USI 20/21. 

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

2) Come up with an algorithm for pre-processing the frames of the whole video "WSC.mp4" in the sense that you should extract all frames from the video that show the table as in the picture "WSC sample.png" and discard all other frames showing the table from a different viewpoint, focusing on a player, or containing advertisement.
