
@@ -1,4 +1,6 @@
# Identifying-Customer-Segments

![](https://alidropship.com/wp-content/uploads/2019/12/2.-Customer-Segmentation.jpg)

## Project Motivation
In this project, I employ several supervised algorithms to accurately model individuals' income using data regarding 3000 existing customers which has been provided to us by our anonymous client. I then choose the best candidate algorithm from preliminary results and further optimize this algorithm to best model the data.



The goal is to construct a model that accurately identifies which of 1000 new customers our client should target, based on thhe data of the 1000 new clients. 

## Libraries Used
* numpy
* pandas
* time
* seaborn 
* matplotlib.pyplot
* Scikitlearn

## Files in this repository
There are four spreadsheets associated with this project:

CustomerDemographics: Demographics data for the 4000 existing customers.

CustomerAddress: Address data for existing customers.

Transactions: Showing the transactions data for the past 3 months.

NewCustomerList: A new list of 1000 potential customers with their demographics and attributes

## Results of Analysis
I have tested four machine learning algorithms: Gaussian Naive Bayes, Adaptive Boosting, the SVM and the 
Random Forest Algorithm.

The Random Forest and SVM models has the best performance in terms of accuracy and computational time,
and so I used it to build the prediction model I needed. 
