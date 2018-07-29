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
        1. Importing required python packages, 
        1. Reading a test image, 
           ![Test Image](https://github.com/agoswami/sdcnd/blob/master/Term1-P1-Finding-Lanes/test_images/solidWhiteRight.jpg]
        1. Enhancing the python helper function, 
        1. Testing on images provided as part of project,
        1. Building a pipeline to process images which are part of frames of a video, 
        1. Test the pipeline on couple of video files provided as project input [Video1](https://github.com/agoswami/sdcnd/blob/master/Term1-P1-Finding-Lanes/test_videos/solidWhiteRight.mp4) [Video2](https://github.com/agoswami/sdcnd/blob/master/Term1-P1-Finding-Lanes/test_videos/solidYellowLeft.mp4), 
        1. Finally test the pipeline on [Challenge Video](https://github.com/agoswami/sdcnd/blob/master/Term1-P1-Finding-Lanes/test_videos/challenge.mp4)


1. Identify any shortcomings

1. Suggest possible improvements

We encourage using images in your writeup to demonstrate how your pipeline works.  

All that said, please be concise!  We're not looking for you to write a book here: just a brief description.

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup. Here is a link to a [writeup template file](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md). 


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

