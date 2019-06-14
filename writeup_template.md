# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Data Augmentation
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/visualization.png "Visualization"
[image2]: ./output_images/bar_chart_plot.png "Bar Chart"

[image3]: ./output_images/bumpy_ahead_tested.png "Bump Ahead"
[image4]: ./output_images/down_arrow_tested.png "Down Arrow"
[image5]: ./output_images/general_caution_tested.png "General Caution"
[image6]: ./output_images/priority_road_tested.png "Priority Road"
[image7]: ./output_images/roundabout_tested.png "Roundabout"
[image8]: ./output_images/speed_limit_30_tested.png "Speed_limit_30"
[image9]: ./output_images/speed_limit_70_tested.png "Speed_limit_70"
[image10]: ./output_images/stop_sign_tested.png "Stop"
[image11]: ./output_images/test_7_tested.png "Yield"
[image12]: ./output_images/upper_aerrow_tested.png "Upper Arrow"
[image13]: ./output_images/walk_tested.png "Walk"
[image14]: ./output_images/working_tested.png "Work"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The Dataset used is German Traffic Signs Dataset which contains images of the shape (32x32x3) i.e. RGB images. I used the Numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Visualization of the dataset is done in two parts. In the first part, a very simple basic approach is taken to display a single random image taken from the dataset. After that, there is an exploratory visualization of the data set, by drawing the first image of 42 classes, 43 classes in total.

![visualization][image1]

Then here it is a bar chart showing sample distribution for 43 traffic signs

![bar_chart][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to shuffle my X_train, y_train. Then, I used Normalization as one of the preprocessing technique. In which, the dataset (X_train, X_test, X_valid) is fed into the normalization(x_label) function which converts all the data and returns the normalized one.

Applied the Following preprocesses:

1.Randomally changed brightness of training image.
2.Randomally changed contrast of training image.
3.Standardized the training, validation and test images.
Augmentation technique also include fliping, rotating by some degree, distortation etc.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

|       Layer      		 |        Description        					       		| 
|:---------------------: |:--------------------------------------------------------:| 
|         Input          |       32x32x3 RGB image   					      		| 
| Convolution Layer 1    |      Outputs 28x28x6 	         		                |
| RELU					 |	Activation applied to output of layer 1	          		|
| Pooling	      	     | Input = 28x28x6, Output = 14x14x6 			       		|
| Convolution Layer 2    | Outputs 10x10x16    							      		|
| RELU		             | Activation applied to output of layer 2   				|
| Pooling				 | Input = 10x10x16, Output = 5x5x16        				|
| Flatten				 |		Input = 5x5x16, Output =400			        		|
| Fully Connected Layer 1|		Input = 400, Output = 120			        		|
| RELU		             | Activation applied to output of Fully Connected layer 1 	|
| Fully Connected Layer 2|		Input = 120, Output = 84						    |
| RELU		             | Activation applied to output of Fully Connected layer 2 	|
| Fully Connected Layer 3|		Input = 84, Output = 43					       		|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used: 
EPOCHS = 25, BATCH_SIZE = 128, rate = 0.001, mu = 0, sigma = 0.1. 
I used LeNet model architecture which consisting 2 convolutional layers and 3 fully connected layers. The input is an image of size (32x32x3) and output is 43 i.e. the total number of distinct classes. In the middle, I used RELU activation function after each convolutional layer as well as the first 2 fully connected layers. Flatten is used to convert the output of 2nd convolutional layer after pooling i.e. 5x5x16 into 400. Pooling is also done in between after the 1st and the 2nd convolutional layer.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.980 
* test set accuracy of 0.886

If a well known architecture was chosen:
* What architecture was chosen?

I used LeNet architecture as used in the lecture which initially gave me validation accuracy of around 98% but when I this applied some changes like augmentation,normalization for better performance the accuracy decreased to 97%  with epoch value 20 and batch size 128. So to get higher accuracy I increased epoch size=25.

* Why did you believe it would be relevant to the traffic sign application?

While there might be other architectures which can be applied and modified to train this model for which I'll keep studying and experimenting further but for the sake of this project, I choose to go with the architecture used in the lecture as I was already comfortable & familiar with its implementation.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
Accuracy of 88% on test set gave me the confidence to test this model on new images taken from web. 9 out of 12 such images were predicted correctly using this model which proved that this model is working model. Although I'll try to improve this model infuture to get better accuracy.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8]
![alt text][image9] ![alt text][image10] ![alt text][image11] ![alt text][image12] ![alt text][image13] ![alt text][image14]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        		| 
|:---------------------:|:---------------------------------:| 
| Stop Sign      		| Stop sign   						| 
| Bump Ahead  			| Bump Ahead 						|
| Yield					| Yield								|
| 30 km/h	      		| 30 km/h			 				|
| 70 km/h   			| 30 km/h           				|
| Keep right	        | Keep right		 				|
| Roundabout Mendatory	| Keep right		 				|
| Ahead only      		| Ahead only		 				|
| Road work	      		| Road work					 		|
| Pedestrians	  		| General Caution		 			|
| General Caution		| General Caution	 				|
| Priority Road	 		| Priority Road		 				|


The model was able to correctly guess 9 of the 12 traffic signs, which gives an accuracy of 75%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.

The top five soft max probabilities were

| Image			        |     Prediction	        		| Top 5 Predictions    | Top 5 probabilities |
|:---------------------:|:---------------------------------:| 
| Stop Sign      		| Stop sign   						|    14  0  1  2  3    |    1  0  0  0  0    |
| Bump Ahead  			| Bump Ahead 						|    22  9  2  25  15  |    1  0  0  0  0    |
| Yield					| Yield								|    13  0  1  2  3    |    1  0  0  0  0    |
| 30 km/h	      		| 30 km/h			 				|    1  32  2  38  12  | 0.99  0  0  0  0    |
| 70 km/h   			| 30 km/h           				|    1   0   2  3  4   |    1  0  0  0  0    |    
| Keep right	        | Keep right		 				|    38  0  1  2  3    |    1  0  0  0  0    |
| Roundabout Mendatory	| Keep right		 				|    38  0  1  2  3    |    1  0  0  0  0    |
| Ahead only      		| Ahead only		 				|   35  36  13  38  8  |    1  0  0  0  0    |
| Road work	      		| Road work					 		|    25  0  1  2  3    |    1  0  0  0  0    |
| Pedestrians	  		| General Caution		 			|    18  0  1  2  3    |    1  0  0  0  0    |
| General Caution		| General Caution	 				|  18  27  11  24  35  |    1  0  0  0  0    |
| Priority Road	 		| Priority Road		 				|    12  0  1  2  3    |    1  0  0  0  0    |


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


