
# Identifying-Customer-Segments

![](https://alidropship.com/wp-content/uploads/2019/12/2.-Customer-Segmentation.jpg)

## Project Motivation
In this project, I employ several supervised algorithms to accurately model the value of customers to a business using relevant data of 3000 existing customers such as demographic, financial, transactions, etc. which has been provided to us by our anonymous client. The best candidate algorithm from preliminary results is chosen and then optimised this algorithm to best model the data.

This model was then applied to accurately identify the segment of 1000 potential new customers so as to show their value to the business and identify which of the new customers our client should target with marketing and advertising. 

![](https://cdn3.notifyvisitors.com/blog/wp-content/uploads/2020/03/types-of-customer-segment1.jpg)



## Libraries Used
* numpy
* pandas
* time
* seaborn 
* matplotlib.pyplot
* Scikitlearn

## Files in this repository
There are four spreadsheets associated with this project:

* CustomerDemographics: Demographics data for the 4000 existing customers.
* CustomerAddress: Address data for existing customers.
* Transactions: Showing the transactions data for the past 3 months.
* NewCustomerList: A new list of 1000 potential customers with their demographics and attributes

## Results of Analysis
I have tested four machine learning algorithms: Gaussian Naive Bayes, Adaptive Boosting, the SVM and the  Random Forest Algorithm. The Random Forest and SVM models has the best performance in terms of accuracy and computational time, and so it was applied it to build the prediction model and predict the segment/ values that each of the new 1000 customers would bring to the business. 
