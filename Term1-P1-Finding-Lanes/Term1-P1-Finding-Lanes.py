#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')




import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    
    # Two array each for right center and right slope
    right_center = []
    left_center = []
    
    # Similar for left center and left slope
    right_slope = []
    left_slope = []
    
    # Iterate over each line in the lines list
    for line in lines:
        
        # Get values of x1,x2 and y1,y2 from each line
        for x1,y1,x2,y2 in line:
            
            # calculate slope of each line
            slope = (y2-y1)/(x2-x1)
            #print (slope)
            
            # calculate center of each line
            center = [(x2+x1)/2,(y2+y1)/2]
            #print (center)
            
            # slope is between 0.2 to 0.8 add to right list
            if (slope > 0.20) and (slope < 0.80):
                right_slope.append(slope)
                right_center.append(center)
                
            # slope is between -0.2 and -0.8 add to left list    
            elif (slope < (-0.20)) and (slope > (-0.80)):
                left_slope.append(slope)
                left_center.append(center)
                
                
            
    # Find the average slope for right lines and average slope for left lines
    average_right_slope = np.sum(right_slope)/len(right_slope)
    #print ("average_right_slope",average_right_slope)
    average_left_slope = np.sum(left_slope)/len(left_slope)
    #print ("average_left_slope",average_left_slope)
    
    # Find the average center for right lines and average slope for left lines
    average_right_center = np.divide(np.sum(right_center,axis=0),len(right_center))
    #print (average_right_center)
    average_left_center = np.divide(np.sum(left_center,axis=0),len(left_center))
    #print (average_left_center)
    
    # Use the height of center of image, average center and average slope to get
    # left line segment end points and right line segment end points
    y_half = 0.6 * img.shape[0]
    y_max = 0.9 * img.shape[0]
    
    global x_left_half 
    global x_right_half
        
    if not average_left_center != 'NaN':
        # left coordinates x_left_max and y_max
        x_left_max = average_left_center[0] + (y_max - average_left_center[1])/average_left_slope
    
        # left coordinates x_left_half and y_half
        x_left_half = average_left_center[0] + (y_half - average_left_center[1])/average_left_slope
    
    if not average_right_center != 'NaN':
        # left coordinates x_right_max and y_max
        x_right_max = average_right_center[0] + (y_max - average_right_center[1])/average_right_slope
    
        # left coordinates x_right_max and y_max
        x_right_half = average_right_center[0] + (y_half - average_right_center[1])/average_right_slope
    
    if x_left_half != 'NaN' and x_right_half != 'NaN':
        # draw overlay left line segment
        cv2.line(img, (int(x_left_half), int(y_half)), (int(x_left_max), int(y_max)), color, thickness)
    
        # draw overlay right line segment
        cv2.line(img, (int(x_right_half), int(y_half)), (int(x_right_max), int(y_max)), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)



import os

# get list of files
files = os.listdir("test_images/")

for file in files:
    if file[0:6] != "output":
    
        # load the image from file
        image = mpimg.imread("test_images/"+file)

        # grayscale helper function, grays the image
        gray = grayscale(image)


        # Define a kernel size and apply Gaussian smoothing
        kernel_size = 5
        blur_gray = gaussian_blur(gray, kernel_size)

        # applying canny edge detection transform
        #canny_image = canny(blur_gray, 200, 255)
        canny_image = canny(blur_gray, 50, 150)

        # Get the dimensions of image
        image_shape = image.shape
        
        # region of interest
        #vertices = np.array( [[[0,540],[500,290],[960,540],[960,540]]], dtype=np.int32 )
        vertices = np.array( [[(0.51*image_shape[1], 0.58*image_shape[0]),(0.49*image_shape[1],0.58*image_shape[0]),(0,image_shape[0]),(image_shape[1],image_shape[0])]], dtype=np.int32 )
        roi_image = region_of_interest(canny_image, vertices);

        #display image with only region of interest
        #plt.imshow(roi_image, cmap='gray')
        
        # generate hough image 
        hough_image = hough_lines(roi_image,1,np.pi/180,35,5,2) 
        
        # generate weighted line
        weighted_image = weighted_img(hough_image,image, α=0.8, β=1.0)
        
        # display the image overlayed with lane lines
        plt.imshow(weighted_image, cmap='gray')
        
        r,g,b = cv2.split(weighted_image)
        weighted_image = cv2.merge((b,g,r))
        
        # write canny_image with original image
        cv2.imwrite("test_images/output_"+file,weighted_image) 



# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    # grayscale helper function, grays the image
    gray = grayscale(image)


    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)

    # applying canny edge detection transform
    #canny_image = canny(blur_gray, 200, 255)
    canny_image = canny(blur_gray, 200, 255)

    # Get the dimensions of image
    image_shape = image.shape
        
    # region of interest
    #vertices = np.array( [[[0,540],[500,290],[960,540],[960,540]]], dtype=np.int32 )
    vertices = np.array( [[(0.51*image_shape[1], 0.58*image_shape[0]),(0.49*image_shape[1],0.58*image_shape[0]),(0,image_shape[0]),(image_shape[1],image_shape[0])]], dtype=np.int32 )
    roi_image = region_of_interest(canny_image, vertices);

    #display image with only region of interest
    #plt.imshow(roi_image, cmap='gray')
        
    # generate hough image 
    hough_image = hough_lines(roi_image,1,np.pi/180,20,5,2) 
        
    # generate weighted line
    weighted_image = weighted_img(hough_image,image, α=0.8, β=1.0)
    
    # setting result to weighted_image
    result = weighted_image

    return result;


white_output = 'test_videos/output_solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)



HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))

challenge_output = 'test_videos/output_challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
%time challenge_clip.write_videofile(challenge_output, audio=False)


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))

