# **Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

## Overview
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project you will detect lane lines in images using Python and OpenCV.  OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.  

To complete the project, I have sumitted three files.

**1.** File containing project code in form of jupyter notebook file [P1.ipynb](https://github.com/agoswami/sdcnd/blob/master/Term1-P1-Finding-Lanes/P1.ipynb)

**2.** File containing project code in form of pure python code [Term1-P1-Finding-Lanes.py](https://github.com/agoswami/sdcnd/blob/master/Term1-P1-Finding-Lanes/Term1-P1-Finding-Lanes.py)

**3.** File containing brief writeup [README.md](https://github.com/agoswami/sdcnd/blob/master/Term1-P1-Finding-Lanes/README.md)

Specifications for submitting the project defined in the [project rubric](https://review.udacity.com/#!/rubrics/322/view) are met.


Reflection
---

There are three parts to the reflection:
 1. Description of the pipeline

    1. Project is completed in jupyter notebook. Following are the Jupyter Notebook Steps
        1. Importing required python packages.
           ```python
              #importing some useful packages
              import matplotlib.pyplot as plt
              import matplotlib.image as mpimg
              import numpy as np
              import cv2
              %matplotlib inline>
           ```
        1. Reading a test image. 
        ![Test Image](https://github.com/agoswami/sdcnd/blob/master/Term1-P1-Finding-Lanes/test_images/solidWhiteRight.jpg)
        1. Enhancing the python helper function. Following are the steps involved in enhancing the method 
           ```python 
           def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
           ```
           1. Input arguments of draw_lines = original image, list of hough transform lines, overlay color, thickness
           1. Iterate of over each line in list of lines. Find center and slope of lines aligned with right and left lanes                   and add them of their own respective lists.
           1. Calculate average of list of center and slopes for left and right lane lines.
           1. Assuming y of right and left lines to be 0.6 of max for y-mid and 0.9 of max for y-max. 
           1. Calculate the x (mid and max) of both right and left lines, using line extrapolation formula. When the list of                 lines are non-empty. If list of lines are empty then skip line generation.
           
        1. Testing on images provided as part of project, through image processing pipeline. Following are the image outputs.
           1. Following are steps to process image, which is same for video image processing as well
              1. read the image
              1. perform grayscaling of image
              1. perform gaussian blur to reduce image noise.
              1. perform canny transform to highlight the edges (lanes).
              1. perform croppping of images to only region of interest.
              1. perform a hough transform to find all the lines of interest (its find all overlapping lines, conditions to choose set of overlapping is defined in params to hough_lines function).
              1. perform a superimpose of hough transform image(lane lines) on original image
              1. perforn RGB conversion to BGR to satify the cv2.imwrite parameter condition.
              1. write out the superimposed image in to a file.
              1. following is the output super-imposed images on test images provided.
                 1. Solid White Curve
                    ![Test Image1](https://github.com/agoswami/sdcnd/blob/master/Term1-P1-Finding-Lanes/test_images/output_solidWhiteCurve.jpg)
                 1. Solid White Right
                    ![Test Image2](https://github.com/agoswami/sdcnd/blob/master/Term1-P1-Finding-Lanes/test_images/output_solidWhiteRight.jpg)
                 1. Solid Yellow Curve
                    ![Test Image3](https://github.com/agoswami/sdcnd/blob/master/Term1-P1-Finding-Lanes/test_images/output_solidYellowCurve.jpg)
                 1. Solid Yellow Curve2
                    ![Test Image4](https://github.com/agoswami/sdcnd/blob/master/Term1-P1-Finding-Lanes/test_images/output_solidYellowCurve2.jpg)
                 1. Solid Yellow Left
                    ![Test Image5](https://github.com/agoswami/sdcnd/blob/master/Term1-P1-Finding-Lanes/test_images/output_solidYellowLeft.jpg)
                 1. White Car Lane Switch
                    ![Test Image6](https://github.com/agoswami/sdcnd/blob/master/Term1-P1-Finding-Lanes/test_images/output_whiteCarLaneSwitch.jpg)
        1. Building a pipeline to process images which are part of frames of a video.
           1. Steps to process video frames are same as steps to process images specified above.
        1. Test the pipeline on couple of video files provided as project input.
           1. Watch output of video1 on youtube
              [![Solid White Right](https://github.com/agoswami/sdcnd/blob/master/Term1-P1-Finding-Lanes/test_videos/output_solidWhiteRight.png)](https://youtu.be/z1puf_4EOHA)
           1. Watch output of video2 on youtube
              [![Solid Yellow Left](https://github.com/agoswami/sdcnd/blob/master/Term1-P1-Finding-Lanes/test_videos/solidYellowLeft.png)](https://youtu.be/UG2gvsgE7Kg)
        1. Finally test the pipeline on challenge video, here you can see on youtube.
           1. [![Challenge Video](https://github.com/agoswami/sdcnd/blob/master/Term1-P1-Finding-Lanes/test_videos/output_challenge.png)](https://youtu.be/aLHsfvypoJ0)


1. Identify any shortcomings
   1. It does not identify lanes with curved lines, we only overlay straight lines.
   1. It only identifies lanes which are clearly marked.
   1. It does not idenity lanes in city roads.
   1. It does not generated overlay lines, when frame data is missing for a little time period.

1. Suggest possible improvements
   1. Possibly predict lane lines, when frame data is not available.
   1. Possibly generate lane line, when lanes are not clearly marked.
   1. Possibly generate lane lines as curved lines instead of staight lines.
   


The Project
---

## If you have already installed the [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) you should be good to go!   If not, you should install the starter kit to get started on this project. ##

**Step 1:** Set up the [CarND Term1 Starter Kit](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/83ec35ee-1e02-48a5-bdb7-d244bd47c2dc/lessons/8c82408b-a217-4d09-b81d-1bda4c6380ef/concepts/4f1870e0-3849-43e4-b670-12e6f2d4b7a7) if you haven't already.

**Step 2:** Open the code in a Jupyter Notebook

You will complete the project code in a Jupyter notebook.  If you are unfamiliar with Jupyter Notebooks, check out <A HREF="https://www.packtpub.com/books/content/basics-jupyter-notebook-and-python" target="_blank">Cyrille Rossant's Basics of Jupyter Notebook and Python</A> to get started.

Jupyter is an Ipython notebook where you can run blocks of code and see results interactively.  All the code for this project is contained in a Jupyter notebook. To start Jupyter in your browser, use terminal to navigate to your project directory and then run the following command at the terminal prompt (be sure you've activated your Python 3 carnd-term1 environment as described in the [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) installation instructions!):

`> jupyter notebook`

A browser window will appear showing the contents of the current directory.  Click on the file called "P1.ipynb".  Another browser window will appear displaying the notebook.  Follow the instructions in the notebook to complete the project.  

**Step 3:** Complete the project and submit both the Ipython notebook and the project writeup

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

