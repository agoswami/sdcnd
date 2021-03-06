## Advanced Lane Finding Project

The goals / steps of this project are the following:

### The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


[//]: # (Image References)

[chessboard1]: ./images/chessboard-corners.png "Chessboards"
[chessboard2]: ./images/chessboard-undistorted.png "Chessboard Undistorted"
[test_image1]: ./images/test_image-undistorted1.png "Image Undistorted1"
[test_image2]: ./images/test_image-undistorted.png "Image Undisotorted2"
[test_image3]: ./images/test_image-transformed.png "Transformed Image"
[test_image4]: ./images/test_image-unwarped.png "Unwarped Images"
[histogram1]: ./images/test_image-histogram1.png "Full Image Histogram"
[histogram2]: ./images/test_image-histogram2.png "Lower Half Image Histogram"
[sliding-window]: ./images/test_image-sliding-window.png "Sliding Window of Lane Lines"
[polyfit-lines]: ./images/test_image-polyfit-lines.png "Polyfit of Lane Lines"
[polyfit-isolation]: ./images/test_image-polyfit-isolation.png "Polyfit of Lane Lines in Isolation"
[draw-lane-lines1]: ./images/test_image-draw-lane-lines1.png "Draw lane line on original image"
[draw-lane-lines2]: ./images/test_image-draw-lane-lines2.png "Draw lane line and data on original image"
[project-video]: ./videos/project_video_output.mp4 "Final project output video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  
You're reading it!

#### 2. Please refer to the following jupyter notebook for code details
https://github.com/agoswami/sdcnd/blob/master/Term1-P3-Advanced-Lane-Finding/Advanced_Lane_Lines.ipynb

### Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

#### 1. Finding the chessboard corners from camera calibration images. 
The code for this step is contained in the 1st code cell (titled "Find the chessboard corners from the camera calibration images") of the IPython notebook located in "https://github.com/agoswami/sdcnd/blob/master/Term1-P3-Advanced-Lane-Finding/Advanced_Lane_Lines.ipynb".  

I start by finding the chessboard corners from camera calibration images. I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  Below is the output of the detected chessboard corners.

![chessboard-corners][chessboard1]

#### 2. Calculate the camera calibration matrix and distortion coefficients.Briefly state how you computed the camera matrix and distortion coefficients. 

Now, we calculate the camera calibration matrix and distortion coefficients. We switch to the 3rd code cell, titled "Write a function to return undistored image for a input images". I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to one of the chessboard images using the `cv2.undistort()` function and obtained this result: 

![chessboard-undistorted][chessboard2]

#### 3. Provide an example of a distortion corrected calibration image.

Now, lets try to use 'cal_undistort' on one test images and see the the undistorted version.

![test_image1][test_image1]



### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

1. Convert the BGR/RGB image GRAY using 'cv2.cvtColor'
2. Use 'cv2.calibrateCamera' function, use objpoints and imgpoints calculated from chessboard images, to get calibration matrices and distortion coefficient.
3. Use 'cv2.undistort' with camera calibration matrix and distortion coefficient to output image.
4. Display that image.

![test_image2][test_image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in code section titled 'Compute thresholded binary image, using color and gradients transforms' in the reference jupyter notebook mentioned above).  

Here's an example of my output for this step. 

1. Convert image to HLS color space, use S channel.
2. Then, perform Sobel x derivative on gray scale image from previous step.
3. Apply threshold on x gradient.
4. Apply threshold on S color channel.
5. Combine all above layers in binary output, example is shown below.

![test image3][test_image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `binary_to_unwarp()`, which appears in code section titled 'Apply Perspective Transform into birds eye view(binary image)' in the jupyter notebook referenced above (Advanced_Lane_Lines.ipynb).  The `binary_to_unwarp()` function takes as inputs an image (`img`), as well as distortion matrix (`mtx`)  and distortion coefficient (`dist`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[330,613], [441,546], [952,613], [854,546]])  
dst = np.float32([[330,613], [330,546], [952,613], [952,546]]) 
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 330, 613      | 330, 613        | 
| 441, 546      | 330, 546      |
| 952, 613     | 952, 613      |
| 854, 546      | 952, 546        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![test image4][test_image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Here are the steps that I performed to identify the lane-lines pixels:

##### a. Identify where the binary activations occur in the binary image, it is shown in code section titled "Plot a histogram where binary activations occur in image". Image of how this activation is shown in histogram below:

![histogram1][histogram1]

##### b. Identify where the binary activations occur in lower half of binary image, it is shown in code section titled "Take a histogram along all the columns in the lower half of the image". Image of how this activation is shown in histogram below:

![histogram2][histogram2]

Histogram of binary image of lower-half of image, is much clearer than the binary image of full binary image.

##### c. Find the peaks of left and right halves of the histogram and these will become the starting point for the left and right lines

##### d. Fit the polynomial through two lane lines, using sliding window to find all the pixels for left and right lane lines, then use these lane pixels to find polynomial to fit each of left and right lane lines (using numpy function `polyfit`). After, we identify these lane lines, we will use it on our next steps. Code is located in the jupyter notebook at code section with title "Fit the polynomial through the two lane lines". Image of how this will look is shown in picture below for our sample image:

![sliding-window][sliding-window]

##### e. Show the margin containing pixels around the detected lane lines. The following picture will show the pixels which were used in generating `polyfit` lane lines. It also shows the margin around lane lines used for line generation.

![polyfit-lines][polyfit-lines]

The following picture is visualization of above in isolation from our example image:

![polyfit-isolation][polyfit-isolation]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Here the idea is to take the measurements of where the lane lines are and estimate how much the road is curving and where the vehicle is located with respect to the center of the lane. The radius of curvature may be given in meters assuming the curve of the road follows a circle. For the position of the vehicle, you may assume the camera is mounted at the center of the car and the deviation of the midpoint of the lane from the center of the image is the offset you're looking for. As with the polynomial fitting, convert from pixels to meters.

I performed this calculation in code section titled "Measure Curvature of each lane line" in the jupyter notebook. The formula used to calculate is mentioned in the lesson, and written below:

f(y) = Ay<sup>2</sup>+By+C

R <sub>curve</sub> = (1+(2Ay+B)<sup>2</sup>) <sup>3/2</sup> / |2A|

Mesured curvature comes out to be: left: 1155.65 right: 1073.58 pixels

Another set of calculations is performed to find curvature in real world, with code in section titled "Measure curvature of each line in real world". Following is the value of left lane line curvature, right lane line curvature and distance from center for example image:

1354.25m, 1021.10m, -0.27 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The fit from the rectified image has been warped back onto the original image and plotted to identify the lane boundaries. This should demonstrate that the lane boundaries were correctly identified. An example image with lanes, curvature, and position from center should be included in the writeup (or saved to a folder) and submitted with the project.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

I have implemented the function `draw_lane` in the jupyter notebook in code section titled `Project your measyrement back down onto the road`. This function performs the following steps:

##### a. Uses example image dimensions to create a black template to draw lane lines from warped perspective, using the functiond mentioned above.

##### b. Then draws the image mentioned in step a. to original example image by unwarping the perspective.

##### c. Then filling color between the left and right lane lines with green color.

![draw-lane-lines1][draw-lane-lines1]

##### d. The following output is after lane curvature and distance from center data is super-imposed on the images in step c. The function to perform the data calculation is mentioned in code section titled "Draw radius of curvature and distance from center of lane onto the image" in the jupyter notebook.

![draw-lane-lines2][draw-lane-lines2]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's final [project-video]

The final project video shows the identified lane with lane boundary lines. The lane between the lines is colored in green and data about curvature and distance from the center is displayed on top left of video.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

The approach, I took which led to lanes being identified on project video is following:

###### a. For each image in video, apply undistortion function.

###### b. Then, apply color and gradient pipeline to convert it to greyscale with highlighted lane lines.

###### c. Then, apply perspective transform function to convert it to top-down view.

###### d. Then, use the object which represent left lane line and right lane line from the line class, which i defined before in code section "Define a class to receive the parameters of each line detection". Initially, these line objects are initialized to defaults.

###### e. Use, the following algorithm:
if both left and right lines were detected in previous frame, use `polyfit_using_prev_fit` function (which will use previous best fit lines), otherwise use sliding window to find new lane lines from scratch.

In either case you will get left lane line and right lane line. Add, these to the list of lines in both left and right line objects.

Then, draw the current best fit if it exists

else, return the exisiting image

Few of the scenarios, where pipeline will fail:

1. When there is difference the color inside the lane boundary, for example in case of new road construction.

2. When the lane lines sometimes go out of the range from the camera, meaning the left or right lane boundaries go out of bounda from the camera view.

3. When the turn on the road are very steep, and turns are going upwards with road elevation or going fownwards with road degradation.


To make it more robust, we can try following:

1. We need to find a better way to highlight lane lines for gradaded coloring of lanes, or try using the boundary of pixel values when they can fit a polynomial curve.

2. We to get activation from all the three sides of camera to identify left and right lane boundaries.

3. We need to identify lane lines which double yellow lane line markers and solid white lane line markers.


