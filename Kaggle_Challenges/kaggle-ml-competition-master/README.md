# Kaggle Machine Learning Competition
The goal of this competition is to detect solar panels in satellite imagery data. You will be given a series of image files. Your goal is to develop a machine learning technique that is able to make a binary decision: is there a solar panel in the image?

## Steps

1. Download and explore the data. It's always best to get a feel for the data before diving in too deeply. Take a look at the training imagery data (which includes class labels) and the test data (which does not include class labels). 
2. Feature extraction. What are common identifying characteristics that separate the images with solar arrays from those without? Is it color? Shapes? Textures? Think about how these could be quantified into feature vectors.
3. Classification. What techniques would be best to apply to these features that you identified? Visualize your training data feature space. Are the two classes linearly separable? Is the data high dimensional? What techniques would work best in each situation?
4. Evaluate performance. Train and test your techniques on separate data. Consider using cross validation or at least subdividing the training data images into two subsets for this purpose (you'll want to avoid incestuously training and testing on the same data). 
5. Estimate test image labels and submit your solution. Use your trained and validated algorithm to produce estimates of the unlabeled test data provided through this contest. Follow the guidelines described in "Evaluation" for submission. Note: You don't have unlimited numbers of submissions, but 5 per day, so use them wisely. Random guessing should yield an AUC of about 0.5 (so this is the lower bound on performance - if you're below 0.5, reversing all of your decisions or confidence (subtracting them from 1) will yield a result above 0.5.

For some of you, this may be your first time tackling an actual machine learning problem. These steps are laid out to help you through that process. Never hesitate to ask for help, just first check with your teammates to see if they have an idea on how to push forward.

## Resources

The following two books have been made freely available by the authors and serve as an excellent two-part series on machine learning, with excellent presentations of classification and regression. James et al. is the introductory book, then Hastie, Tibshirani, and Friedman is the more advanced text.

James, Witten, Hastie and Tibshirani. [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/).

Hastie, Tibshirani, Friedman. [The Elements of Statistical Learning](http://statweb.stanford.edu/~tibs/ElemStatLearn/).

# Evaluation
The evaluation metric for this competition is the area under the curve (AUC) of the Receiver Operating Characteristic (ROC) curve. You can either submit confidence values for each test observation (which amounts to providing the full ROC and a more accurate measure of AUC) or simply binary decision values (0 or 1) which will lead to an ROC with a single point between (0,0) and (1,1).

Two resources on ROCs include this [page](http://ebp.uga.edu/courses/Chapter%204%20-%20Diagnosis%20I/8%20-%20ROC%20curves.html) and this [video](https://www.youtube.com/watch?v=OAl6eAyP-yo).

## Submission Format

Submission files should contain two columns: id and class. The submission should be a comma separated file (csv). Each file should have one entry for each image ranging from id 201 through 330 (for a total of 130 entries).

You have two options for submission: (1) binary decisions, or (2) confidence values / scores.

### Option 1: Binary Decisions

The file should contain a header with "id" and "class", where class is a binary variable of 0 (no solar array) or 1 (solar array). An example is shown below:


id,class  
201,0  
202,0  
203,1  
204,1  
...  
...  
330,1  


### Option 2: Confidence values

Most classifiers produce a confidence value that is thresholded to produce the final binary decision. These confidence values range from 0 to 1 and are the confidence in the given observation belonging to the target class (in this case, the class of solar arrays). The file should contain a header and have the following format:

id,class  
201,0.44  
202,0.89  
203,0.33  
204,0.98  
...  
...  
330,0.05  

# Data Description
Participants should be able to answer questions like these after reading the data description: What files do I need? What should I expect the data format to be? What am I predicting? What acroynms will I encounter?

## File descriptions

* **train_images.zip** - folder containing the training image data set of 200 images. All images are in color .tif format.
* **test_images.zip** - folder containing the test image set of 130 images.  All images are in color .tif format.
* **sampleSubmission.csv** - a sample submission file in the correct format. Replace the class labels in this file with those that you have learned from the training data and submit that file for evaluation
* **trainSolution.csv** - a solution containing the IDs and the class labels for each of the training images.
* **runExample.m** - a Matlab file that provides (1) a sample testbed for exploring the performance of a cross-validated machine learning algorithm for solving this solar array detection problem and (2) code for running a trained classifier on the test data and producing a valid Kaggle submission.

## Fields

* **id** - corresponds to the id of the image. For example, an id of 12 corresponds to "image12.tif"
* **class** - For denoting class, 0 = "no solar panels" and 1 = "solar panels present". For the submission, the class field can also contain confidence values / scores.

## Images

All images are .tif (tagged image file format) images with three color channels. The size of each image is 101px-by-101px.

# Grading
In addition to your solution submission here on Kaggle, you'll turn in via Box.com a two-page description of your approach to solving this problem and an appendix of the code you used.

## Report

This report should include:

1. Overview of the problem being solved
2. A description of the method(s) used including your rational for choosing each approach. This should include a description of: (a) data preprocessing, (b) feature extraction, (c) classification algorithm(s), (d) performance evaluation.
3. A discussion of (a) the efficacy of your approach, (b) the challenges you faced along the way, and (c) how performance could be improved.
4. A well-commented appendix of cleanly-written code (this is in addition to the 2 page write-up). See the Matlab coding best practices document on Box for more details.
5. References cited.

The written report will be graded based on completeness (covers all of the topics described above), logical flow (tells a coherent and organized story), and professionalism (is well-written and contains header information including a title, your name, etc).

## Competition Grade

The grade for this sprint assignment will be based out of 100 points. 95/100 points will be based on the writeup above and the code appendix. The remaining 5 points will be scored as follows:

* 5/5 for 1st place (same grade if there is a tie)
* 4/5 for 2nd place (same grade if there is a tie)
* 3/5 for 3rd place (same grade if there is a tie)
* 2/5 for all other places
* 0/5 if no submission is made

You're encouraged to work collaboratively on this project, but each individual should turn in their own original work.
