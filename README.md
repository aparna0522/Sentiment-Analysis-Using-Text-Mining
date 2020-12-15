# Sentiment-Analysis-Using-Text-Mining-In-Health-Care-Field
Sentiment Analysis of Patients' Blogs from various online forums and analysing them into categories 
 ```
 Disease Exists (Neutral Sentiment)
 Health Deteriorating (Negative Sentiment) 
 Health Recovering (Positive Sentiment) 
 ```
The classification is done with the help of Naive Bayes Probabilistic Classifier. The final aim is to find the accuracy for various accuracies of training and testing datasets. 
The programming is done in R language.

Datasets Source: Online Website - https://patient.info/ (Educational purpose only)

<h1>How to run this project?</h1> 
1. Clone this repository.
2. Create a database consisting of two columns: Label and Blogs
   In the "Label" column, the sentiment of the blog will be mentioned, i.e. Exists, Deteriorate or Recover
   In the "Blogs" column, input the blogs from any online forums, or self articulated blogs from various sources.
3. Open R compiler, run the entire code. 

<h2>How can the accuracy be increased?</h2> 
1. Increase or decrease the number of times the dataset is randomized. It can help in increasing the accuracy by 10% at most. 
2. Try to label the dataset more accurately.
