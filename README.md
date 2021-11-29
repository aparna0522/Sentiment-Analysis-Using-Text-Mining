# Sentiment-Analysis-Using-Text-Mining
<h3>Brief Description about the project</h3>

Sentiment Analysis of Patients' Blogs from various online forums and analysing them into categories 
 ```
 Disease Exists (Neutral Sentiment)
 Health Deteriorating (Negative Sentiment) 
 Health Recovering (Positive Sentiment) 
 ```
The classification is done with the help of Naive Bayes Probabilistic Classifier. The final aim is to find the accuracy for various accuracies of training and testing datasets. 
The programming is done in R language.

Datasets Source: Online Website - https://patient.info/ (Educational purpose only)

<h3>How to run this project?</h3> 

1. Clone this repository. 
2. Create a database consisting of two columns: Label and Blogs \
   In the "Label" column, the sentiment of the blog will be mentioned, i.e. Exists, Deteriorate or Recover. \
   In the "Blogs" column, input the blogs from any online forums, or self articulated blogs from various sources. 
3. Open R compiler, run the entire code. 

<h3>How to increase the accuracy?</h3>

1. Increase or decrease the number of times the dataset is randomized, it can help in increasing the accuracy by 10% at most.   
2. Try to label the dataset more accurately.

<h4> Results </h4>

Results from the dataset considered show the sentiment scores for the given emotion (anger, anticipation, fear, ....)

<img width="666" alt="Screenshot 2021-11-29 at 1 56 29 PM" src="https://user-images.githubusercontent.com/36110304/143833046-f61e9369-9d83-4805-ac6b-b81b3d28cfef.png">

Utilizing different proportions of training and testing datasets to find the accuracy changes

<img width="666" alt="Screenshot 2021-11-29 at 1 57 04 PM" src="https://user-images.githubusercontent.com/36110304/143833056-b9de6618-fb8f-4407-af3f-9de2ae882f6b.png">
<img width="666" alt="Screenshot 2021-11-29 at 1 57 17 PM" src="https://user-images.githubusercontent.com/36110304/143833063-68ecaf65-f099-4963-b0d2-a434b219e7a3.png">

Accuracy verses the proportion of dataset used for training

<img width="666" alt="Screenshot 2021-11-29 at 1 57 28 PM" src="https://user-images.githubusercontent.com/36110304/143833067-44c9e377-98c3-4d76-8e4a-3d3cf0fca276.png">
