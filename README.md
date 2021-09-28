# Detecting Diabetic Retinopathy with Image Classification/CNN

# Background and problem statement

## Diabetic Retinopathy 

Diabetic retinopathy is a complication of diabetes that affects the eyes. Generally, it is caused by damage to the blood vessels in tissue at the back of the eye. Diabetic patients will develop diabetic retinopathy after they have had diabetes for between three to five years. DR is generally diagnosed within three years of type-1 diabetes, however it may already be present when type-2 diabetes is diagnosed. A diet with poorly controlled blood sugar may increase the risk of developing diabetes, and therefore indadvertently developing diabetic retinopathy as well. Mild cases of DR may be treated with careful diabetes management, but advanced cases may require laser treatment or surgery.

## Background

Diabetic retinopathy is a leading cause of blindness among working-age adults, and afflicts millions of people annually. Diabetic retinopathy is prevelant in rural parts of India, which are often times remote and technologically isolated, making medical screenings for the ailment much more difficult to conduct. Historically, medical technicians are physically sent to these remote locations, to screen DR by photographing the retina and then manually assessing each image as a basis for DR diagnosis. 

These professionals look for a variety of identifying characteristics visibily present on the retina such as hemorhages, aneurysms, abnormal growth of blood cells, among others. All of these can indicate if DR is present, and how severe the diagnosis is. 

This method of relying on trained professionals to classify each image is laborious, inefficient, and costly. In an effort to save resources, automating parts of the process could be very beneficial and could drastically increase the scope of screening accessibility. 

## Problem Statement

Our goal is to expedite DR detection by building a multi-classification model that will be trained on thousands of images collected in rural India. This will be used to automatically predict if a patient has DR and how severe it is on a scale from zero to four, with severity ascending:

- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR

# Premodeling

## Metadata

The data used in this repository was retrieved from the APTOS 2019 Blindness Detection competetion, hosted on Kaggle.com. It was compiled over several years and was released in July, 2019. It contains fours folders, two for the training images and training labels, and the other for the testing images and testing labels. Images in the image folders are .png files, and the labels are .csv files. The training images folder contained 3,662 images, each one 640 x 480 pixels in area, with three color channels (RGB). Furthermore, the images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

## Loading

To load in our data, we used a created a for loop to populate two lists, based on code provided to us by our lovely instructors at General Assembly. It utlizes plyplot imread and numpy resize to read in the images, and them scale down from 640 x 480 pixels to 128 x 128 pixels. We needed our image file names to match the names of images in the labels lists, and used a list comprehension to return the file names without the .png suffix. In order for our data to be feed into our neural networks, we rescaled our images by dividing each one by 255, which is the range of the grey scale in 8-bit representation. Our intial target variable value counts will provide us with our baseline scores for our five possible outcomes, as follows:

 - 0: 49.49%
 - 1: 10.10% 
 - 2: 27.28% 
 - 3: 5.27% 
 - 4: 8.06%

 We then train-test split on our training data, with 75% going to training and 25% going to testing, and stratify our target variables due to the imbalanced nature of our labels. Finally, we used utils.to_categorical to one-hot encode our target variable, creating a dummy array that can be used in our multi-classification CNN. 

# Approach

As a three person team, we split up the task of testing different models with a common goal of maximizing our model's test metric: the quadratic-weighted kappa cohen. To go through the details of each model that was tested, there are folders for each of the team members with their model explanations and corresponding code. 

# Production Model Evaluation

Nick's EfficientNet model scored the highest quadratic weighted kappa score (0.895), so we chose this as our production model; this type of neural network was ubiquotous amont the highest scoring entities on this Kaggle competition's leaderboard. The structure of an EfficientNet CNN, as well as its scaling method work to scale all dimensions (width, depth, resolution) uniformly, distinquishes itself from the artibitrary scalers of typical CNNs. Furthermore, by increasing the image size from the initial 128x128 resize, we were able to see an even better QWK.

## Accuracy / Overfit

In terms of plain accuracy, our model did not do very well with a 0.5841 training accuracy score and 0.5328 testing accuracy score. Scores like these are concerning and raise questions about the over all efficacy of our model, considering our testing accuracy score narrowly beats our baseline accuracy score of 49.49%. Additionally, aside from poor predictive power, the discrepancy in training and testing scores suggests that our model is overfit and suffers from high-variance.   


## QWK

 The quadratic weighted Kappa is a metric used to compare the agreement between two observed "raters". The quadratic weighted kappa is calculated between the scores which are expected/known and the predicted scores and gives extra penalty to predicted outcomes that are further from the actual outcomes. 
 
 In our case, the efficient net model can predict class 0 (No DR) well with 96%, but struggles with predicting classes 1,3,4 as it has a tendency to overpredict these classes and as class 2 outcomes. Our model only ever misclassifies outcomes within a two class range, which is better than if the its range were larger, but this proclivity towards class 2 conflation still hurts our quadratic weighted kappa score. 


# Conclusion

In the end, we created a model that is not perfect, but may be useful - especially depending on what metric you are evaluated by. When working in a group, we learned clear communication is essential, especially when confirming that all team members are working from the same dataset. We learned that hyper-tuning the parameters of a CNN is not necessarily as useful as modifying the image itself such as augumenting the image rotationally, or resizing as large as possible. Looking forward, we posit that we could have achieved a better quadratic weighted kappa score if we had more processing power. By enabling larger computational abilities, we could then perform large scale augmentation and/or read our images into our CNNs without scaling them down which can harm models by diminishing nuances - ultimately blunting the graphical information. Finally, there is a little body of research on the internet on how to apply KMeans clustering for image classification and would be curious to see how well it would perform on this dataset.
