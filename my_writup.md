# **Traffic Sign Classifier**

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[bar_chart]: ./images/bar_chart.png "Bar Chart"
[random_image]: ./images/random_image.png "Random Image"
[web_image_1]: ./additional-traffic-signs-data/3.jpg  "Traffic Sign 1"
[web_image_2]: ./additional-traffic-signs-data/4.jpg  "Traffic Sign 2"
[web_image_3]: ./additional-traffic-signs-data/11.jpg "Traffic Sign 3"
[web_image_4]: ./additional-traffic-signs-data/13.jpg "Traffic Sign 4"
[web_image_5]: ./additional-traffic-signs-data/14.jpg "Traffic Sign 5"
[softmax_image_1]: ./images/softmax_60.png "Softmax 60 km/h"
[softmax_image_2]: ./images/softmax_70.png "Softmax 70 km/h"
[softmax_image_3]: ./images/softmax_right_of_way.png "Softmax Right of Way"
[softmax_image_4]: ./images/softmax_yield.png "Softmax Yield"
[softmax_image_5]: ./images/softmax_stop.png "Softmax STOP"


## Rubric Points
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/tyler-lewis-udacity/Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library and python methods to calculate summary statistics of the traffic signs data set:

* Number of images in the training set   = 34799
* Number of images in the validation set = 4410
* Number of images in the test set       = 12630
* Image data shape                       = (32, 32, 3)
* Number of classes                      = 43


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the number of images per class in the training dataset:

![alt text][bar_chart]

A mapping of sign descriptions to class ID can be found in [signnames.csv](./signnames.csv)


Here is a random image from the training dataset shown along with useful information about the image including name, size, datatype, and range of pixel values:

![alt text][random_image]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale.  I tried training the classifyer on grayscale, RGB, and HLS images.  The results were about the same for all three color mappings, so I stuck with grayscale which reduced training time.

I then normalized the grayscaled images so that the range of pixel values was changed from (0,256) to (-1,1).  Without normalization, the classifier would end up putting too much emphasis on differences in images within a class that are insignificant.  This way, the classifier does not need learn the difference between "bright" and "dim" signs of a particular class.

Finally, the images in the training dataset were shuffled to ensure that any ordering of the images had no effect on the training process.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   					|
| Convolution        	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution           | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 5x5x16                   |
| Flatten               | outputs 400                                   |
| Fully Connected       | outputs 120                                   |
| RELU                  |                                               |
| Fully Connected       | outputs 84                                    |
| RELU                  |                                               |
| Fully Connected       | outputs 43                                    |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained the model using Tensorflow with GPU support.  The training parameters used were:

* Optimizer = Adam Optimizer
* Batch Size = 128
* Epochs = 100
* Learning Rate = 0.001

I experemented with different batch sizes, epochs, and learning rates.  Increasing the batch size to 256 yielded much faster training times but less accurate results.  I tried reducing the learning rate several times (0.0005, 0.0006, 0.0007, 0.0008, 0.0009) but none yielded better results than 0.001.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I used the well-known LeNet architecture as a basis for my classifier.  The LeNet model does a good job of classifying 32x32 pixel images of hand-written numbers.  It worked relatively well for traffic signs, which are composed of simple shapes and numbers.  Adding more convolutional layers might improve the performance.

My final model results were:

* Validation set accuracy of 95.1%
* Test set accuracy of 93.8%

What architecture was chosen?

* I used the LeNet architecture as a basis for my classifier.

Why did you believe it would be relevant to the traffic sign application?

* The LeNet model does a good job of classifying 32x32 pixel images of hand-written numbers.  It worked relatively well for traffic signs, which are composed of simple shapes and numbers.  Adding more convolutional layers might improve the performance.

How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

* The validation accuracy was only slightly better than the test set accuracy.  The test set contained images that the classifier had never seen before which means that the validation process was working well.  Increasing the number of images in the training dataset by augmenting existing images through rotation, scaling, etc. would probably bring the gap between validation accuracy and test accuracy even closer.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][web_image_1] ![alt text][web_image_2] ![alt text][web_image_3]
![alt text][web_image_4] ![alt text][web_image_5]

The "60" and "70" km/h speed limit signs might be difficult to classify because of the similarities in shape between the numbers.  For instance, "7" is similar to a "1" in shape.  The bottom half of the "6" is similar in shape to the bottom half of a "5" or an "8".  The other three images should be easy to classify based on their clarity, shape, and orientation.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 60 km/h        		| 60 km/h   									|
| 70 km/h     			| 30 km/h 										|
| Right of way          | Right of way									|
| Yield 	      		| Yield     					 				|
| STOP      			| STOP                 							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is about the same as the accuracy on the training set.  A better comparison between the two datasets could only be made with a larger set of web images.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The 5 images gathered from the web are shown alongside a bar chart that shows the Softmax probabilities for each sign:

![alt text][softmax_image_1]
![alt text][softmax_image_2]
![alt text][softmax_image_3]
![alt text][softmax_image_4]
![alt text][softmax_image_5]

The classifier did well on all signs other than 70 km/h speed limit sign, which the classifyer thought was a 30 km/h speed limit sign.  The classifyer was wrong yet it was very confident in its prediction (over 80%).  A larger and broader training set, through augmentation of the provided set, would likely help.
