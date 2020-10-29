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

![polyfit-isolation][polyfit-isolation]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.
