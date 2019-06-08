## Project: Traffic Sign Recognition
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, I will use what I've learned about deep neural networks and convolutional neural networks to classify traffic signs. I will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I will then try out your model on images of German traffic signs that you find on the web.

For this project, I will submit three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file 

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed among classes

Below is visualization on Training Data Set

![Test Image1](https://github.com/agoswami/sdcnd/blob/master/Term1-P2-Traffic-Sign-Classifier-Project/data_set_images/training-data-set.png)

Below is visualization on Validation Data Set

![Test Image2](https://github.com/agoswami/sdcnd/blob/master/Term1-P2-Traffic-Sign-Classifier-Project/data_set_images/validation-data-set.png)

Below is visualization on Test Data Set

![Test Image3](https://github.com/agoswami/sdcnd/blob/master/Term1-P2-Traffic-Sign-Classifier-Project/data_set_images/test-data-set.png)

Sample Images with Class as label

![Sample Image](https://github.com/agoswami/sdcnd/blob/master/Term1-P2-Traffic-Sign-Classifier-Project/data_set_images/sample-images.png)
### Design, Train and Test a Model Architecture

#### 1. Technique used for preprocessing - NORMALIZATION

As, the first step with the data set. I performed three step pre-processing of images.

1. Convert the RGB image into Greyscaling.
2. Take the Greyscale Image and perform Histogram Equilization.
3. Then, perform normalization on pixel value so that there zero mean and equal variance. 
   By, performing: pixel = (pixel - 128)/ 128
   
Below is example of one such image. 

![Test Image4](https://github.com/agoswami/sdcnd/blob/master/Term1-P2-Traffic-Sign-Classifier-Project/normalized_images/normalized-image.png)

#### 2. Following is the final model architecture in the tabular format ( including model type, layers, layer sizes, connectivity, etc.) 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Greyscale image   							| 
| Layer 1: Convolution 3x3     	| Convolutional. Filter - 5,5,1,6 Input = 32x32x1. Output = 28x28x6. 	|
| Activation:| RELU					|
| Max Pooling: | Stride:2x2 Input = 28x28x6. Output = 14x14x6.|
| Layer 2: Convolutional| Output = 10x10x16|
| Activation:|  RELU|
| Max Pooling: | Input = 10x10x16. Output = 5x5x16|
| Flatten:| Input = 5x5x16. Output = 400|
| Layer 3: Fully Connected.| Input = 400. Output = 120|
| Activation:| RELU|
| Layer 4: Fully Connected| Input = 120. Output = 84|
| Activation:| RELU|
| Applying dropouts | keep_prob = 1.0 |
| Layer 5: Fully Connected | Input = 84. Output = 43|

#### 3. Following is the training approach used including ( the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate)

To train the model, I have used adam optimizer. There are other options available for Optimizers ex. Adagrad, Adadelta etc. However, it is found that Adam optimizer converges the fastest. So, it is being used here. There are  The BATCH is 16, it is taken a power of 2, Since the CPU and GPU memory architecture usually organizes the memory in power of 2, which can speed up the fetch of data to memory. And, the number of EPOCHS is 100. I have used learning rate as rate = 0.00060. To initialize weight variable I have used mean = 0, stddev = 0.1

#### 4. Following section describes the model results, architecture decision, process used to train the model and get to final model and hyper-parameters.

My final model results were:
* training set accuracy of 93.6
* validation set accuracy of 93.6
* test set accuracy of 90.5


### Accuracy with EPOCHS
![Test Epoch1](https://github.com/agoswami/sdcnd/blob/master/Term1-P2-Traffic-Sign-Classifier-Project/accuracy-with-epoch.png)


### Learning curve for training
![Test Curve1](https://github.com/agoswami/sdcnd/blob/master/Term1-P2-Traffic-Sign-Classifier-Project/data_set_images/training-learning-curve.png)

### Learning curve for validation
![Test Curve2](https://github.com/agoswami/sdcnd/blob/master/Term1-P2-Traffic-Sign-Classifier-Project/data_set_images/validation-learning-curve.png)

### Learning curve for test
![Test Curve3](https://github.com/agoswami/sdcnd/blob/master/Term1-P2-Traffic-Sign-Classifier-Project/data_set_images/test-learning-curve.png)

Architecture:

* LeNet achitecture was chosen. 
  
* Classification problem in LeNet is similar to Traffic Sign Classification. A small change in the LeNet  architecture can make it adapt to Traffic Sign classification.
  
* Training accuracy on training set, validation set and test set are very close. This indicates that model has tun well.

Fine tuning the model:

* Learning rate was achieved by using delta increase or decrease technique and was set to 0.00060, also while learning rate was changed everything else was kept constant.
* BATCH size was also chosen by binary method, and keeping other variables constant. It was found optimal at 16. Since, smaller BATCH size generalizes the model more than larger BATCH size.
* EPOCHS was also chosen by binary method, while keeping other variables contant. It was finally set to 100.
* The training exit when training accuracy goes above 93% and model is saved at this moment.
* DROPOUTS could have been used for regularization, but decided not to use it. Finally setting keep_prob to 1.0 before the last layer, Or not dropping out anything.
* EARLY TERMINATION, epochs are set to 100, but model training is terminated when the accuracy goes greater than 93%.
* AVOIDING UNDERFITTING: By using deep learning lenet architecture and using a large data set for training, underfitting is avoided.
* AVOIDING OVERFITTING: By using regularization technique like dropouts and early termination, overfitting can be avoided.


### Test a Model on New Images

#### 1. Following are five German traffic signs found on the web. 

Here are five German traffic signs that I found on the web:

![Test Image German 1](https://github.com/agoswami/sdcnd/blob/master/Term1-P2-Traffic-Sign-Classifier-Project/five_german_signs/speed_60.png) ![Test Image German 2](https://github.com/agoswami/sdcnd/blob/master/Term1-P2-Traffic-Sign-Classifier-Project/five_german_signs/wild_animal.png) ![Test Image German 3](https://github.com/agoswami/sdcnd/blob/master/Term1-P2-Traffic-Sign-Classifier-Project/five_german_signs/speed_120.png) 
![Test Image German 4](https://github.com/agoswami/sdcnd/blob/master/Term1-P2-Traffic-Sign-Classifier-Project/five_german_signs/speed_100.png) ![Test Image German 5](https://github.com/agoswami/sdcnd/blob/master/Term1-P2-Traffic-Sign-Classifier-Project/five_german_signs/speed_30.png)

All first four images are difficult to classify because it is dark. Specially Speed Limit (120km/h) because it is hard to see it visually.

#### 2. Following are the predictions on five german traffic signs discussed above, and reason for one failure in prediction.

Here are the results of the prediction:

| Image			        |     Prediction	        					| Image Description |
|:---------------------:|:---------------------------------------------:|:--------------------------:|
| ![Test Image German 1](https://github.com/agoswami/sdcnd/blob/master/Term1-P2-Traffic-Sign-Classifier-Project/five_german_signs/speed_60.png) 		| Speed limit (60km/h)								| straight angle, very dark, low contrast, very jittery and no obvious background object | 
| ![Test Image German 2](https://github.com/agoswami/sdcnd/blob/master/Term1-P2-Traffic-Sign-Classifier-Project/five_german_signs/wild_animal.png)   			| Wild animals crossing										| slight angle, very dark, low contrast, very jittery and vehicle in background |
| ![Test Image German 3](https://github.com/agoswami/sdcnd/blob/master/Term1-P2-Traffic-Sign-Classifier-Project/five_german_signs/speed_120.png)			| Speed limit (100km/h)								| straight angle, very dark, low contrast, very jittery and no obvious background object |
| ![Test Image German 4](https://github.com/agoswami/sdcnd/blob/master/Term1-P2-Traffic-Sign-Classifier-Project/five_german_signs/speed_100.png)    		| Speed limit (100km/h)				 				| straight angle, slightly light, low contrast, very jittery and no obvious background object|
| ![Test Image German 5](https://github.com/agoswami/sdcnd/blob/master/Term1-P2-Traffic-Sign-Classifier-Project/five_german_signs/speed_30.png)		| Speed limit (30km/h)      							| straight angle, bright image, low contrast, very jittery and no obvious background object|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 90.5%

#### 3. Following section will explain softmax probabilities on each image prediction. And, how the prediction chose a particular class.

The code for making predictions on my final model is located in the 9th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Speed limit (100km/h) sign (top probability of 0.99), and the image does contain a Speed limit (100km/h) sign. Everyother image is correctly classified with top probability of 1.0.

Below is a table of all the prediction and top 5 probabilities.

|  Image  |     Traffic Sign   	|     	     Top 5  Probability			| Prediction |
|:-------:|:---------------------:|:---------------------------------------------:| :---------------------------------------------:|
|![Test Image German 1](https://github.com/agoswami/sdcnd/blob/master/Term1-P2-Traffic-Sign-Classifier-Project/five_german_signs/speed_60.png) | Speed limit (60km/h) |  1.00000000e+00,   2.32628268e-17,   2.36262311e-21,  1.26511731e-22,   1.74394900e-23 | Speed limit (60km/h) | 
| ![Test Image German 2](https://github.com/agoswami/sdcnd/blob/master/Term1-P2-Traffic-Sign-Classifier-Project/five_german_signs/wild_animal.png) | Wild animals crossing  | 1.00000000e+00,   1.98907540e-11,   6.79139582e-16, 2.57423477e-16,   3.04496698e-17	| Wild animals crossing|
| ![Test Image German 3](https://github.com/agoswami/sdcnd/blob/master/Term1-P2-Traffic-Sign-Classifier-Project/five_german_signs/speed_120.png) | Speed limit (120km/h)	 | 1.00000000e+00,   7.42253684e-14,   4.36191447e-24, 6.63499814e-27,   1.04985321e-27 | Speed limit (120km/h)|
| ![Test Image German 4](https://github.com/agoswami/sdcnd/blob/master/Term1-P2-Traffic-Sign-Classifier-Project/five_german_signs/speed_100.png) | Speed limit (100km/h)  | 9.99829054e-01,   1.70878790e-04,   5.09370039e-08, 1.28797755e-08,   6.90450852e-09 | Speed limit (100km/h)|
| ![Test Image German 5](https://github.com/agoswami/sdcnd/blob/master/Term1-P2-Traffic-Sign-Classifier-Project/five_german_signs/speed_30.png) | Speed limit (30km/h)	|	1.00000000e+00,   1.52510602e-26,   4.73973719e-28, 1.60767514e-29,   6.90052212e-33 | Speed limit (30km/h)|
