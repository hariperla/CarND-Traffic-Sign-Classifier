# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./ClassDistribution.png "Visualization of the class distribution"
[image2]: ./TrainingImage.jpg "Example of a Training Image"
[image3]: ./TrainingImageGrayScale.jpg "Grayscale converted training image"
[image4]: ./00001.jpg "Traffic Sign 1"
[image5]: ./00009.jpg "Traffic Sign 2"
[image6]: ./08745.jpg "Traffic Sign 3"
[image7]: ./Test_.jpg "Traffic Sign 4"
[image8]: ./12627.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 5219
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data for the training and the testing sets are skewed towards the left which indicates that there is better distribution for the initial dataset. Since both training and testing data sets show similar pattern, we will not have issues with training our network

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it helps with extracting features. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2] ![alt text][image3]

As a last step, I normalized the image data because because it helps us to standardize the data around the mean. We chose to have a mean of zero and standard deviation of 0.1


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flatten	        	| outputs 400                   				|
| Fully connected		| outputs 120  									|
| Fully connected		| outputs 84  									|
| Fully connected		| outputs 43  									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Initially used EPOCHS of 10 and a BATCH_SIZE of 180 which when ran through the network I got an accuracy on the validation set of 98%. I used a learning rate of 0.002. After this I tested my network on the testing set and got an accuracy of 91.9%. Since this did not give me the result that was expected, I changed the EPOCHS to 20 and BATCH_SIZE to 130 and upped the learning rate to 0.003 which improved my validation set accuracy to 98.7% and improved my test set accuracy to 93.2%. I tried different other hyper parameter values which proved to be less accurate than the previous ones. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100
* validation set accuracy of 98.9 
* test set accuracy of 93.2

If a well known architecture was chosen:
* What architecture was chosen?
The network architecture is based on the excercise that we did for convolution neural networks, using a different dataset. I found it beneficial to feed the dense layers with the output of both the previous convolutional layers. Indeed, in this way the classifier is explicitly provided both the local "motifs" (learned by conv1) and the more "global" shapes and structure (learned by conv2) found in the features. I tried to replicate the same architecture, made by 2 convolutional layers flattened and 2 fully connected layers. I also used dropouts to avoid overfitting the data which is a common problem with large data sets. If dropouts were not used, the accuracy went down by 1% on validation set and 0.7% on the test set. 
* Why did you believe it would be relevant to the traffic sign application?
I believed it was relavant because this model works on large data sets with greater accuracy than LeNet or other previous methods mentioned
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
As mentioned above, the final model results prove that the accuracy of the model is pretty good with some mis-classifications. This is acceptable. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The second image is very difficult to classify because, it is very dark and the resolution on the image is also very bad to do any processing. The other images are well lit and hence can be classified with greater accuracy. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 kph        		| 30 kph    									| 
| No-speeding  			| U-turn 										|
| Straight				| Straight										|
| 120 km/h	      		| 120 km/h  					 				|
| Deer Crossing			| Deer Crossing      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 5 because 5 is statistically not a large enough data set

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a 30kph sign (probability of 0.35), and the image does contain a 30kph sign. The top 3 soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| 30 kph    									| 
| 0.0     				| 80 kph    									|
| 0.0					| 40 kph										|
| 0.0					| 50 kph										|
| 0.0					| 20 kph										|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


