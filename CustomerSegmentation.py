#!/usr/bin/env python
# coding: utf-8

# # Project: Customer Segments Identification

# In this project, I employ several supervised algorithms to accurately model individuals' income using data regarding 3000 existing customers which has been provided to us by our anonymous client. I then choose the best candidate algorithm from preliminary results and further optimize this algorithm to best model the data. The deta of 1000 new clients are then provided. The goal is to construct a model that accurately identifies which of the 1000 customers our client should target, based on this dataset.

# In[1]:


import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import seaborn as sns
import scipy.stats as stats
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import os


# # Step 1: Load the data

# There are four spreadsheets associated with this project:
# 
# CustomerDemographics: Demographics data for the 4000 existing customers.
# 
# CustomerAddress: Address data for existing customers.
# 
# Transactions: Showing the transactions data for the past 3 months.
# 
# NewCustomerList: A new list of 1000 potential customers with their demographics and attributes
# 
# The client wants to know which of the new 1000 customers to target. The aim is to provide useful customer insights which could help optimise the allocation of resource for targeted marketing and improve performance by focusing on high value customers.

# In[2]:


import os
  
# Function to Get the current 
# working directory
def current_path():
    print("Current working directory before")
    print(os.getcwd())
    print()
# Driver's code
# Printing CWD before
current_path()
  
# Changing the CWD
os.chdir("C:/Users/preco/OneDrive/Desktop/ML with Python")
  
# Printing CWD after
current_path()


# In[3]:


df = pd.read_excel('CustomerData.xlsx',sheet_name = ['Transactions','NewCustomerList','CustomerDemographic', 'CustomerAddress']) 
CustomerDemo_df = pd.read_excel('CustomerData.xlsx',sheet_name = 'CustomerDemographic')
Transactions_df = pd.read_excel('CustomerData.xlsx',sheet_name = 'Transactions')
CustomerAddress_df = pd.read_excel('CustomerData.xlsx',sheet_name = 'CustomerAddress')
NewCustomers_df = pd.read_excel('CustomerData.xlsx',sheet_name = 'NewCustomerList')


# In[4]:


# Let's see the structure of the data after it's loaded (e.g. print the number of
# rows and columns, print the first few rows).
print("The Customer Demographic dataset has %s rows and %s columns" %(CustomerDemo_df.shape[0], CustomerDemo_df.shape[1]))
print("The Transactions dataset has %s rows and %s columns"  %(Transactions_df.shape[0], Transactions_df.shape[1]))
print("The Customer Address dataset has %s rows and %s columns" %(CustomerAddress_df.shape[0], CustomerAddress_df.shape[1]))
print("The New Customers dataset has %s rows and %s columns"  %(NewCustomers_df.shape[0], NewCustomers_df.shape[1]))


# # 1. Let us work on the Customer Demographic Spreadsheet

# In[5]:


display(CustomerDemo_df.head(5))


# In[6]:


#Let us see the data types
pd.DataFrame({"Data type":CustomerDemo_df.dtypes})


# In[7]:


# Let's get the perentage of Missing Data in the data
CustomerDemo_df.isnull().sum()
pd.DataFrame({"Missing Data (%)":CustomerDemo_df.isnull().sum()/len(CustomerDemo_df.index)*100})


# We can see the percentage of missing data. We will deal with these later

# Let us deal with the genders and replace "Female","Femal" with "F and "Male" with "M"

# In[8]:


CustomerDemo_df["gender"] = CustomerDemo_df["gender"].replace(to_replace =["Female","Femal"], value = "F")
CustomerDemo_df["gender"] = CustomerDemo_df['gender'].replace(to_replace =["Male"], value = "M")
CustomerDemo_df


# We do not need the need the data where the "deceased indicator = Y". So will drop these rows of data. We will do this later

# Let us drop some columns that will not be necsaary for our prediction

# In[9]:


Final_Customer_Demo_df= CustomerDemo_df[["customer_id","gender","past_3_years_bike_related_purchases",
                                         "DOB","job_industry_category","wealth_segment","owns_car","tenure"]]


# In[10]:


Final_Customer_Demo_df


# # 2. Let us work on the Transaction Spreadsheet

# In[11]:


pd.DataFrame({"Data type":Transactions_df.dtypes})


# In[12]:


Transactions_df.isnull().sum()
pd.DataFrame({"Missing Data (%)":Transactions_df.isnull().sum()/len(Transactions_df.index)*100})


# In[13]:


Transactions_df


# # 3. Let us work on the Customer Address Spreadsheet

# In[14]:


CustomerAddress_df.isnull().sum()
pd.DataFrame({"Missing Data (%)":CustomerAddress_df.isnull().sum()/len(CustomerAddress_df.index)*100})


# Let us obtain the columns that will be used in the prediction model

# In[15]:


Final_Customer_Address_df= CustomerAddress_df[["customer_id","postcode","state","property_valuation"]]


# The names in some of the columns id mismatched. For example,Victoria and VIC seen in some columns mean the same thing. Let us replace these with state codes to make the state column uniform

# In[16]:


Final_Customer_Address_df["state"] = Final_Customer_Address_df["state"].replace(to_replace =["Victoria"], value = "VIC")
Final_Customer_Address_df["state"] = Final_Customer_Address_df["state"].replace(to_replace =["New South Wales"], value = "NSW")
Final_Customer_Address_df


# **Let us see the number of customer that completed their order**

# In[17]:


#define a function for plotting the order_status data
def plot_stacked_bar_charts(dataframe, title_, size_=(20,10), rot_=0, legend_ = "lower right"):
    #plot stacked bars with annotations
    ax =dataframe.plot(kind="bar",
                       stacked=True,
                       figsize=size_,
                       rot=rot_,
                       title=title_)
    #Annotate bars
    #annotate_stacked_bar_charts(ax, textsize=14)
    #Rename legend
    plt.legend(["Approved", "Cancelled"], loc=legend_)
    #Labels
    plt.ylabel("Pr(%)")
    plt.show()
    
# This function shows the values of the percentage calculated on the stacked bar charts
def annotate_stacked_bar_charts(ax, pad=0.99, colour="white", textsize=13):
    
    # Iterate over the plotted rectangles/bars
    for p in ax.patches:
        # Calculate annotation
        value=str(round(p.get_height(),1))
        
        # If value is 0, do not annotate
        if value == '0.0':
            continue
        ax.annotate(value,
                    ((p.get_x()+p.get_width()/2)*pad-0.05, (p.get_y()+p.get_height()/2)*pad),
                           color=colour,size=textsize,
                          )


# In[18]:


order_data = Transactions_df[["product_id","online_order","order_status"]]
order_data.head()


# In[19]:


order_d = order_data.groupby([order_data["order_status"], order_data["online_order"]])["product_id"].count().unstack(level=0)


# In[20]:


order_d


# In[21]:


order_type = ['Online', 'Not Online']


# In[ ]:





# In[22]:


ax = sns.catplot( x='online_order',
             kind="count", 
             hue="order_status", 
             height=5, 
             aspect=1.5, 
             data=order_data)
plt.xlabel("Online Order", size=14)
plt.ylabel("No. of Transactions", size=14)
plt.tight_layout()
plt.savefig("Grouped_barplot_with_Seaborn_catplot.png")


# A very small percentage of the online and non-online transactions were cancelled. This means that more sales were made for "in-person" orders

# We will drop the rows of data in which the order are cancelled. 

# In[23]:


#Let us drop the transacctions that were canccelled because these transactions can not be used as successful
cancel_df = Transactions_df[ Transactions_df['order_status'] == "Cancelled"].index


# In[24]:


# drop these row indexes
# from dataFrame
Transactions_df.drop(cancel_df, inplace = True)


# In[25]:


Transactions_df


# We will now obtain the profit made by the business per customer over the given period of time. This data would be the sum of the profits (list price - standard cost) made on all the transactions by a customer. This will be used together with the demographic and address spreadsheets to create features to be used in the prediction model. This column may be used as our target variable in our prediction model

# In[26]:


Transactions_df["Profit_per_customer"] = Transactions_df["list_price"] - Transactions_df["standard_cost"]
Profit_per_customer = pd.DataFrame(Transactions_df["Profit_per_customer"].groupby
                                   (Transactions_df["customer_id"]).sum())
Profit_per_customer.head()


# We will also obtain the number of transaction made per customer over the given period of time. This data would be used together with the demographic and address spreadsheets to create the prediction model. This column may be used as our target variable in our prediction model

# In[27]:


Transactions_No = Transactions_df[["transaction_id","customer_id"]]


# In[28]:


Transaction_per_customer = Transactions_No.groupby(Transactions_No["customer_id"]).count()
Transaction_per_customer.head()


# In[29]:


#mereg the No. of Transactions and trhe profit per customer columns
Transactions_Profit_per_Customer = pd.merge(Transaction_per_customer,Profit_per_customer, on ='customer_id')
Transactions_Profit_per_Customer.head()


# ### Data merging

# Now let us merge the data frame we want to use in the modelling

# In[30]:


merged_data  = pd.merge(Final_Customer_Address_df,Transactions_Profit_per_Customer, on ='customer_id')
#pd.merge(train, categories_channel, left_index=True, right_index=True)


# In[31]:


Existing_Customers_df = pd.merge(merged_data,Final_Customer_Demo_df, on = 'customer_id')
Existing_Customers_df.head()


# Let us transform the DOB column to the no. of years (age of the customers) using a reference date of 01/01/2021

# In[32]:


Existing_Customers_df = Existing_Customers_df.dropna()


# In[33]:


import datetime


# In[34]:


#Define a function that converts the dates to months
def convert_date_to_years(reference_date, dataframe, column):
    """
    Input a column with timedeltas and return years
    """
    delta_time = REFERENCE_DATE - dataframe[column]
    years = (delta_time / np.timedelta64(1, "Y")).astype(int)
    
    #int returns the number of months as a whole number, not decimal
    
    return years


# In[35]:


# Create reference date as provided
REFERENCE_DATE = datetime.datetime(2021,1,1)


# In[36]:


Existing_Customers_df["DOB"] = pd.to_datetime(Existing_Customers_df["DOB"],format='%Y-%m-%d')


# In[37]:


Existing_Customers_df["DOB"] = convert_date_to_years(REFERENCE_DATE, Existing_Customers_df, "DOB")


# In[38]:


transactions_by_state = pd.DataFrame(Existing_Customers_df["transaction_id"].groupby(Existing_Customers_df["state"]).sum())
transactions_by_gender = pd.DataFrame(Existing_Customers_df["transaction_id"].groupby(Existing_Customers_df["gender"]).sum())
transactions_by_brand = pd.DataFrame(Existing_Customers_df["transaction_id"].groupby(Transactions_df["brand"]).sum())


# In[39]:


states = ['NSW', 'QLD', 'VIC']
# Creating explode data
explode = (0.1, 0.1, 0.1)  
# Creating color parameters
colors = ( "red", "green", "white", "indigo", "beige")  
# Wedge properties
wp = { 'linewidth' : 1, 'edgecolor' : "black" }  
# Creating autocpt arguments
def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d} transactions)".format(pct, absolute) 
# Creating plot
fig, ax = plt.subplots(figsize =(10, 7))
wedges, texts, autotexts = ax.pie(transactions_by_state, 
                                  autopct = lambda pct: func(pct, transactions_by_state),
                                  explode = explode, 
                                  labels = states,
                                  shadow = True,
                                  colors = colors,
                                  startangle = 90,
                                  wedgeprops = wp,
                                 textprops = dict(color ="black"))  
# Adding legend
ax.legend(wedges, states,
          title ="states",
          loc ="center left",
          bbox_to_anchor =(1, 0, 0.5, 1))  
plt.setp(autotexts, size = 8, weight ="bold")
ax.set_title("Successful transactions by state chart") 
# show plot
plt.show()


# In[40]:


brands = ['Giant Bicycles','Norco Bicycles','OHM Cycles','Solex','Trek Bicycles','WeareA2B']
# Creating explode data
explode = (0.1, 0.0, 0.1, 0.1, 0.0, 0.0) 
# Creating color parameters
colors = ( "red", "green", "white", "orange", "beige", "grey") 
# Wedge properties
wp = { 'linewidth' : 1, 'edgecolor' : "black" } 
# Creating autocpt arguments
def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d} transactions)".format(pct, absolute)  
# Creating plot
fig, ax = plt.subplots(figsize =(10, 7))
wedges, texts, autotexts = ax.pie(transactions_by_brand, 
                                  autopct = lambda pct: func(pct, transactions_by_brand),
                                  explode = explode, 
                                  labels = brands,
                                  shadow = True,
                                  colors = colors,
                                  startangle = 90,
                                  wedgeprops = wp,
                                  textprops = dict(color ="black"))  
# Adding legend
ax.legend(wedges, brands,
          title ="brands",
          loc ="upper left",
          bbox_to_anchor =(1, 0, 0.5, 1)) 
plt.setp(autotexts, size = 8, weight ="bold")
ax.set_title("Successful transactions by brand chart") 
# show plot
plt.show()


# # 2. Feature Engineering

# Transform the owns_car column to Boolean through one hot encoding

# In[41]:


#In the column has_gas, replace t with Tr`1ue or 1 and f with False or 0 . 
#This process is known as onehot encoding
Existing_Customers_df["owns_car"]=Existing_Customers_df["owns_car"].replace(["Yes", "No"],[1,0])


# **Dealing with Categorical Data and Dummy Variables**

# a. State Column

# In[42]:


# Transform column into categorical data type
Existing_Customers_df["state"] = Existing_Customers_df["state"].astype("category")


# In[43]:


# How many categories of channel sales are there?
pd.DataFrame({"Samples in category": Existing_Customers_df["state"].value_counts()})
# Create 3 dummy variables for each variable
state_dummies = pd.get_dummies(Existing_Customers_df["state"], prefix = "state")
# We rename columns for simplicity
state_dummies.columns = [col_name[:11] for col_name in state_dummies.columns]


# b. Gender column

# In[44]:


# Transform column into categorical data type
Existing_Customers_df["gender"] = Existing_Customers_df["gender"].astype("category")


# In[45]:


# How many categories of channel sales are there?
pd.DataFrame({"Samples in category": Existing_Customers_df["gender"].value_counts()})
# Create 3 dummy variables for each variable
gender_dummies = pd.get_dummies(Existing_Customers_df["gender"], prefix = "gender")
# We rename columns for simplicity
gender_dummies.columns = [col_name[:11] for col_name in gender_dummies.columns]


# c. Job_industry_category

# In[46]:


# Transform column into categorical data type
Existing_Customers_df["job_industry_category"] = Existing_Customers_df["job_industry_category"].astype("category")


# In[47]:


# How many categories of channel sales are there?
pd.DataFrame({"Samples in category": Existing_Customers_df["job_industry_category"].value_counts()})
# Create 9 dummy variables for each variable
job_industry_category_dummies = pd.get_dummies(Existing_Customers_df["job_industry_category"], prefix = None)
# We rename columns for simplicity
job_industry_category_dummies.columns = [col_name[:11] for col_name in job_industry_category_dummies.columns]


# d. Wealth Segment

# In[48]:


Existing_Customers_df["wealth_segment"] = Existing_Customers_df["wealth_segment"].astype("category")


# In[49]:


# How many categories of channel sales are there?
pd.DataFrame({"Samples in category": Existing_Customers_df["wealth_segment"].value_counts()})


# In[50]:


# Create 3 dummy variables for each variable
wealth_segment_dummies = pd.get_dummies(Existing_Customers_df["wealth_segment"], prefix = "Wealth_seg")
wealth_segment_dummies.columns = [col_name[:11] for col_name in wealth_segment_dummies.columns]


# **Merge the dummy variables to main dataframe**

# In[51]:


# Use common index to merge
Existing_Customers_df = pd.merge(Existing_Customers_df, state_dummies, left_index=True, right_index=True)
Existing_Customers_df = pd.merge(Existing_Customers_df, gender_dummies, left_index=True, right_index=True)
Existing_Customers_df = pd.merge(Existing_Customers_df, job_industry_category_dummies, left_index=True, right_index=True)
Existing_Customers_df = pd.merge(Existing_Customers_df, wealth_segment_dummies, left_index=True, right_index=True)


# In[52]:


#Let us remove the old categorical columns
Existing_Customers_df.drop(columns=["state","gender","job_industry_category","wealth_segment"],inplace=True)


# # 3. Data Visualisation

# In[53]:


Existing_Customers_df.describe()


# In[54]:


fig, axs = plt.subplots(nrows=5, figsize=(18,50))
# Plot histograms
sns.distplot((Existing_Customers_df["past_3_years_bike_related_purchases"].dropna()), ax=axs[0])
sns.distplot((Existing_Customers_df["property_valuation"].dropna()), ax=axs[1])
sns.distplot((Existing_Customers_df["DOB"].dropna()), ax=axs[2])
sns.distplot((Existing_Customers_df["tenure"].dropna()), ax=axs[3])
sns.distplot((Existing_Customers_df["transaction_id"].dropna()), ax=axs[4])


# The DOB data is skewed to the left. All other columns are okay. However, the standard deviation is high. So we will transform all of the data.

# In[55]:


fig, axs = plt.subplots(nrows=4, figsize=(18,50))
# Plot histograms
sns.boxplot((Existing_Customers_df["past_3_years_bike_related_purchases"].dropna()), ax=axs[0])
sns.boxplot((Existing_Customers_df["property_valuation"].dropna()), ax=axs[1])
sns.boxplot((Existing_Customers_df["DOB"].dropna()), ax=axs[2])
sns.boxplot((Existing_Customers_df["tenure"].dropna()), ax=axs[3])


# There are some outliers in the DOB data. We will deal with this now. For example, there is an age of about 180 years. This is unrealistic, so we will drop all ages above 80 years.

# ## Removing Outliers

# In[56]:


# 1. Define a function to find the outliers 
def find_outliers_iqr(dataframe, column):   
    col = sorted(dataframe[column])
    q1, q3= np.percentile(col,[25,75])
    iqr = q3 - q1
    lower_bound = q1 -(1.5 * iqr)
    upper_bound = q3 +(1.5 * iqr)
    results = {"iqr": iqr, "lower_bound":lower_bound, "upper_bound":upper_bound}
    return results

# 2. Define a function to remove the outliers found
def remove_outliers_iqr(dataframe, column):
    outliers = find_outliers_iqr(dataframe, column)
    removed = dataframe[(dataframe[column] < outliers["lower_bound"]) |
                        (dataframe[column] > outliers["upper_bound"])].shape                  
        # | means OR
    dataframe = dataframe[(dataframe[column] > outliers["lower_bound"]) & 
                          (dataframe[column] < outliers["upper_bound"])]
    print("Removed:", removed[0], " outliers")
    return dataframe
Existing_Customers_df = remove_outliers_iqr(Existing_Customers_df,"DOB")


# Let us categorise the ages of the customers into age brackets

# In[57]:


Existing_Customers_df.loc[Existing_Customers_df["DOB"]<=20,"DOB"]= 0
Existing_Customers_df.loc[(Existing_Customers_df
                           ["DOB"]>=21)&(Existing_Customers_df
                                                           ["DOB"]<=30),"DOB"]= 1
Existing_Customers_df.loc[(Existing_Customers_df
                           ["DOB"]>=31)&(Existing_Customers_df
                                                           ["DOB"]<=40),"DOB"]= 2
Existing_Customers_df.loc[(Existing_Customers_df
                           ["DOB"]>=41)&(Existing_Customers_df
                                                           ["DOB"]<=50),"DOB"]= 3
Existing_Customers_df.loc[(Existing_Customers_df
                           ["DOB"]>=51)&(Existing_Customers_df
                                                           ["DOB"]<=60),"DOB"]= 4
Existing_Customers_df.loc[Existing_Customers_df["DOB"]>=61,"DOB"]= 5


# Let us drop the postcode column as we will not use it in our prediction modelling

# In[58]:


Existing_Customers_df.drop(columns=["postcode"], inplace=True)


# We will categorise the n umber of transaction made by each customer and the total profits made from each customer into three categories. These categories will represent the level of engagement or the value of the customer. We will call these categories: Low value, Intermediate value and High value customers.

# Based on the number of Transactions, customers with less than 5 transactions over the specified period are called low value customers, those with between 6 to 10 transactions are intermediate value, and those with more than 11 transactions are high value customers

# In[59]:


Existing_Customers_df = Existing_Customers_df.rename(index=str,columns={"transaction_id": "No_of_transactions"})


# In[60]:


Existing_Customers_df.loc[Existing_Customers_df["No_of_transactions"]<=5,"No_of_transactions"]= 0
Existing_Customers_df.loc[(Existing_Customers_df
                           ["No_of_transactions"]>=6)&(Existing_Customers_df
                                                       ["No_of_transactions"]<=10),"No_of_transactions"]= 1
#Existing_Customers_df.loc[(Existing_Customers_df["No_of_transactions"]>=9)&(Existing_Customers_df["No_of_transactions"]<=12),"No_of_transactions"]= 3
Existing_Customers_df.loc[Existing_Customers_df["No_of_transactions"]>=11,"No_of_transactions"]= 2


# In[61]:


Existing_Customers_df["No_of_transactions"].value_counts()


# Based on the profitability of a customer, customers with less than 5 transactions over the specified period are called low value customers, those with between 6 to 10 transactions are intermediate value, and those with more than 11 transactions are high value customers

# In[62]:


Existing_Customers_df.loc[Existing_Customers_df["Profit_per_customer"]<=2000,"Profit_per_customer"]= 0
Existing_Customers_df.loc[(Existing_Customers_df["Profit_per_customer"]>=2001)&(Existing_Customers_df
                                                           ["Profit_per_customer"]<=4000),"Profit_per_customer"]= 1
Existing_Customers_df.loc[Existing_Customers_df["Profit_per_customer"]>=4001,"Profit_per_customer"]= 2


# In[63]:


Existing_Customers_df["Profit_per_customer"].value_counts()


# In[64]:


df = Existing_Customers_df["Profit_per_customer"].value_counts()
Engagement = ['Intermediate Value', 'Low Value', 'High Value']
# Creating explode data
explode = (0.0, 0.0, 0.2)  
# Creating color parameters
colors = ( "red", "green", "beige", "orange", "beige", "grey") 
# Wedge properties
wp = { 'linewidth' : 1, 'edgecolor' : "black" } 
# Creating autocpt arguments
def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d} customers)".format(pct, absolute)  
# Creating plot
fig, ax = plt.subplots(figsize =(10, 7))
wedges, texts, autotexts = ax.pie(df, 
                                  autopct = lambda pct: func(pct, df),
                                  explode = explode, 
                                  labels = Engagement,
                                  shadow = True,
                                  colors = colors,
                                  startangle = 90,
                                  wedgeprops = wp,
                                  textprops = dict(color ="black"))  
# Adding legend
ax.legend(wedges, Engagement,
          title ="Engagement",
          loc ="center left",
          bbox_to_anchor =(1, 0, 0.5, 1)) 
plt.setp(autotexts, size = 8, weight ="bold")
ax.set_title(None) 
# show plot
plt.show()


# In[65]:


Existing_Customers_df.describe()


# # 5. Pickling

# In[66]:


import os
import pickle


# In[67]:


PICKLE_TRAIN_DIR = os.path.join("..", "processed_data", "Existing_Customers_df.pkl")


# In[68]:


pd.to_pickle(Existing_Customers_df, PICKLE_TRAIN_DIR)


# # 2. Model Development

# In[69]:


PICKLE_TRAIN_DIR = os.path.join("..", "processed_data", "Existing_Customers_df.pkl")


# In[70]:


Existing_Customers_df = pd.read_pickle(PICKLE_TRAIN_DIR)


# In[71]:


Existing_Customers_df.drop(columns=["customer_id","gender_U","gender_F","gender_M"],inplace=True)


# ## High Correlation Variables 

# **Calculating the correlation of the variables**

# In[72]:


correlation3 = Existing_Customers_df.corr()


# In[73]:


# Plot correlation
plt.figure(figsize=(19,15))
sns.heatmap(correlation3, xticklabels=correlation3.columns.values, 
            yticklabels=correlation3.columns.values, annot = True, annot_kws={'size':10})
# Axis ticks size
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show() 


# Becasue of the high correlation between The no. of transactions and the profit per customer columns as well as the DOB and tenure columns, we will drop the No. of Transactions and the Tenure columns

# In[74]:


Existing_Customers_df.drop(columns=["No_of_transactions","tenure"],inplace=True)


# ### Data Transformation

# We will now transform the skewed data such as the 'property valuation' column.
# For highly-skewed feature distributions such as 'capital-gain' and 'capital-loss', it is common practice to apply a logarithmic transformation on the data so that the very large and very small values do not negatively affect the performance of a learning algorithm. We apply this transformation ont the data and get the following:

# In[75]:


to_transform = Existing_Customers_df.drop(labels=["Profit_per_customer"],axis=1)


# In[76]:


to_transform_log = to_transform + 1
to_transform_log = np.log(to_transform_log)


# In[77]:


fig, axs = plt.subplots(figsize=(5,5))
# Plot histograms
sns.distplot((to_transform_log["property_valuation"]))


# In[78]:


y = Existing_Customers_df["Profit_per_customer"]
X = to_transform_log
#X = Existing_Customers_df.drop(labels=["Profit_per_customer"],axis=1)


# In[79]:


# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)

X_transform = pd.DataFrame(data = X)
#features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
display(X_transform.head(n = 5))


# ### Shuffle and Split Data##
# Now all categorical variables have been converted into numerical features, and all numerical features have been normalized. I will now split the data (both features and their labels) into training and test sets. 80% of the data will be used for training and 20% for testing.

# In[80]:


# Import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Split the 'X' and 'y' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transform, y, 
                                                    test_size=0.2, random_state=50)
# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# ## Model Fitting

# In[81]:


from time import time
from sklearn.metrics import fbeta_score, accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    beta = 0.5
    
    start = time() # Get start time
    learner = learner.fit(X_train, y_train)
    end = time() # Get end time
    
    results['train_time'] = end - start
        
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train)
    end = time() # Get end time
    
    results['pred_time'] = end - start
            
    results['acc_train'] = accuracy_score(y_train, predictions_train)
        
    results['acc_test'] = accuracy_score(y_test, predictions_test)
          
    # Success
    print("The accuracy results of {} are {}.".format(learner.__class__.__name__, results))
    #table_ = pd.DataFrame(results)
    # Return the results
    return results


# In[82]:


# TODO: Import the four supervised learning models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.svm import SVC

# TODO: Initialize the three models
clf_A = GaussianNB()
clf_B = RandomForestClassifier(n_estimators = 100, random_state=0)
clf_C = AdaBoostClassifier(random_state=0)
clf_D = svm.SVC(kernel = 'rbf', gamma=1, C = 1)

samples_100 = len(y_train)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C, clf_D]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_100]):
        results[clf_name][i] =         train_predict(clf, samples, X_train, y_train, X_test, y_test)


# ### Implementation: Model Tuning

# I will use the GridSearch to fine-tune the models. I would focus on the models with the best results i.e. the Random Forest model and the SVM model.

# #### Tuning the Random Forest model

# In[83]:


# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
#from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV

# TODO: Initialize the classifier
clf_1 = RandomForestClassifier(random_state=0)

# TODO: Create the parameters list you wish to tune, using a dictionary if needed.
# HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}
parameters = {"max_depth": [2, 5, 10, 20],
              "n_estimators": [2, 5, 100, 200],
              "min_samples_split": list(range(2, 5)),
              "min_samples_leaf": list(range(1, 5)),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# TODO: Make an fbeta_score scoring object using make_scorer()
#crit = accuracy_score

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj_RF = GridSearchCV(clf_1, parameters, cv=5)

# TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit_RF = grid_obj_RF.fit(X_train, y_train)

# Get the estimator
best_clf_1 = grid_fit_RF.best_estimator_

# Make predictions using the unoptimized and model
training_RF = (clf_1.fit(X_train, y_train)).predict(X_train)
best_training_RF = best_clf_1.predict(X_train)

# Make predictions using the unoptimized and model
predictions_RF = (clf_1.fit(X_train, y_train)).predict(X_test)
best_predictions_RF = best_clf_1.predict(X_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on training data: {:.4f}".format(accuracy_score(y_train, training_RF)))
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions_RF)))
print("\nOptimized Model\n------")
print("Accuracy score on training data: {:.4f}".format(accuracy_score(y_train, best_training_RF)))
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions_RF)))


# #### HyperParameter Tuning for the SVM Predictor

# In[84]:


from sklearn.utils import shuffle
#from sklearn.svm import SVC
#from sklearn.metrics import confusion_matrix,classification_report
#from sklearn.model_selection import cross_val_score, GridSearchCV

# Performing CV to tune parameters for best SVM fit 
params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},
               {'kernel': ['linear'], 'C': [1, 2, 10, 100, 1000]}]
clf_2 = svm.SVC()
grid_obj = GridSearchCV(clf_2, params_grid, cv=5)

# TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf_2 = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
training_SVM = (clf_2.fit(X_train, y_train)).predict(X_train)
best_training_SVM = best_clf_2.predict(X_train)

# Make predictions using the unoptimized and model
predictions_SVM = (clf_2.fit(X_train, y_train)).predict(X_test)
best_predictions_SVM = best_clf_2.predict(X_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on training data: {:.4f}".format(accuracy_score(y_train, training_SVM)))
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions_SVM)))

print("\nOptimized Model\n------")
print("Accuracy score on training data: {:.4f}".format(accuracy_score(y_train, best_training_SVM)))
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions_SVM)))


# In[85]:


result = {"Algorithm" : ["SVM", "RandomForest"],
        "Unoptimised Test Set Accuracy": [((accuracy_score(y_test, predictions_SVM))*100).astype(int),
                                    ((accuracy_score(y_test, predictions_RF))*100).astype(int)],
          "Optimised Test Set Accuracy":[(accuracy_score(y_test, best_predictions_SVM)*100).astype(int), 
                            (accuracy_score(y_test, best_predictions_RF)*100).astype(int)]}

result_table=pd.DataFrame(result)
result_table


# In[86]:


fig, ax = plt.subplots(figsize=(10, 8))

# Our x-axis. We basically just want a list
# of numbers from zero with a value for each
# of our jobs.
labels = result_table['Algorithm']

x = np.arange(len(labels))

# Define bar width. We need this to offset the second bar.
bar_width = 0.35

b1 = ax.bar(x, result_table['Unoptimised Test Set Accuracy'],
            width = bar_width, label = 'Unoptimised Accuracy')
# Same thing, but offset the x.
b2 = ax.bar(x+bar_width, result_table['Optimised Test Set Accuracy'],
            width = bar_width, label = 'Optimised Accuracy')

# Fix the x-axes.

ax.set_xticks(x)
ax.set_xticklabels(labels)

# Add legend.
ax.legend()

# Add axis and chart labels.
ax.set_xlabel('Job', labelpad=15)
ax.set_ylabel('# Employed', labelpad=15)
ax.set_title('Employed Workers by Gender for Select Jobs', pad=15)

fig.tight_layout()

# You can just append this to the code above.

# For each bar in the chart, add a text label.
for bar in ax.patches:
  # The text annotation for each bar should be its height.
  bar_value = bar.get_height()
  # Format the text with commas to separate thousands. You can do
  # any type of formatting here though.
  text = f'{bar_value:,}'
  # This will give the middle of each bar on the x-axis.
  text_x = bar.get_x() + bar.get_width() / 2
  # get_y() is where the bar starts so we add the height to it.
  text_y = bar.get_y() + bar_value
  # If we want the text to be the same color as the bar, we can
  # get the color like so:
  bar_color = bar.get_facecolor()
  # If you want a consistent color, you can just set it as a constant, e.g. #222222
  ax.text(text_x, text_y, text, ha='center', va='bottom', color=bar_color,
          size=12)


# It is seen that the accuracy scores of both optimised models are 41%. So we can use any of the models for our prediction. We will use the SVM model

# # Let us work on the New Customer List

# In[90]:


pd.DataFrame({"Data type":NewCustomers_df.dtypes})


# In[91]:


NewCustomers_df = NewCustomers_df[["property_valuation","past_3_years_bike_related_purchases","DOB","owns_car",
                                   "tenure","state","job_industry_category","wealth_segment"]]


# In[92]:


NewCustomers_df.isnull().sum()
pd.DataFrame({"Missing Data (%)":NewCustomers_df.isnull().sum()/len(NewCustomers_df.index)*100})


# In[93]:


NewCustomers_df["state"] = NewCustomers_df["state"].replace(to_replace =["Victoria"], value = "VIC")
NewCustomers_df["state"] = NewCustomers_df["state"].replace(to_replace =["New South Wales"], value = "NSW")


# In[94]:


NewCustomers_df = NewCustomers_df.dropna()


# In[95]:


#Define a function that converts the dates to months
def convert_date_to_years(reference_date, dataframe, column):
    """
    Input a column with timedeltas and return years
    """
    delta_time = REFERENCE_DATE - dataframe[column]
    years = (delta_time / np.timedelta64(1, "Y")).astype(int)
    
    #int returns the number of months as a whole number, not decimal
    
    return years


# In[96]:


# Create reference date as provided
REFERENCE_DATE = datetime.datetime(2021,1,1)


# In[97]:


NewCustomers_df["DOB"] = pd.to_datetime(NewCustomers_df["DOB"],format='%Y-%m-%d')


# In[98]:


NewCustomers_df["DOB"] = convert_date_to_years(REFERENCE_DATE, NewCustomers_df, "DOB")


# ### Data Visualisation

# In[99]:


# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(NewCustomers_df["past_3_years_bike_related_purchases"], bins = 10)
# Show plot
plt.show()


# In[100]:


# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(NewCustomers_df["property_valuation"], bins = 10)  
# Show plot
plt.show()


# In[101]:


# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(NewCustomers_df["DOB"], bins = 10)
plt.title("Dist. of Customers' Age")
plt.xlabel("Age")
plt.ylabel("No. of Customers") 
# Show plot
plt.show()


# In[102]:


df1 = NewCustomers_df["state"].value_counts()
states = ['NSW', 'VIC', 'QLD']

# Creating explode data
explode = (0.1, 0.1, 0.1)
  
# Creating color parameters
colors = ( "red", "green", "white", "indigo", "beige")
  
# Wedge properties
wp = { 'linewidth' : 1, 'edgecolor' : "black" }
  
# Creating autocpt arguments
def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d} customers)".format(pct, absolute)
  
# Creating plot
fig, ax = plt.subplots(figsize =(10, 7))
wedges, texts, autotexts = ax.pie(df1, 
                                  autopct = lambda pct: func(pct, df1),
                                  explode = explode, 
                                  labels = states,
                                  shadow = True,
                                  colors = colors,
                                  startangle = 90,
                                  wedgeprops = wp,
                                  textprops = dict(color ="black"))  
# Adding legend
ax.legend(wedges, states,
          title ="states",
          loc ="center left",
          bbox_to_anchor =(1, 0, 0.5, 1))
  
plt.setp(autotexts, size = 8, weight ="bold")
ax.set_title("No. of Customers by state") 
# show plot
plt.show()


# In[103]:


df2 = NewCustomers_df["job_industry_category"].value_counts()
job = ['Financial Services', 'Manufacturing', 'Health', 'Retail', 'Property', 'Entertainment',
         'IT', 'Argiculture', 'Telecommunications']
# Creating explode data
explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
# Creating color parameters
colors = ( "red", "green", "white", "orange", "beige","indigo", "blue", "grey", "brown") 
# Wedge properties
wp = { 'linewidth' : 1, 'edgecolor' : "black" }  
# Creating autocpt arguments
def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d} customers)".format(pct, absolute)  
# Creating plot
fig, ax = plt.subplots(figsize =(10, 7))
wedges, texts, autotexts = ax.pie(df2, 
                                  autopct = lambda pct: func(pct, df2),
                                  explode = explode, 
                                  labels = job,
                                  shadow = True,
                                  colors = colors,
                                  startangle = 90,
                                  wedgeprops = wp,
                                  textprops = dict(color ="black"))  
# Adding legend
ax.legend(wedges, job,
          title ="job_industry_category",
          loc ="upper left",
          bbox_to_anchor =(1, 0, 0.5, 1)) 
plt.setp(autotexts, size = 8, weight ="bold")
ax.set_title("No. of Customers by job category") 
# show plot
plt.show()


# In[104]:


df3 = NewCustomers_df["owns_car"].value_counts()
job = ['NO', 'Yes']
# Plot the data using bar() method
plt.bar(job, df3, color='g')
plt.title("No. of Customers with cars")
plt.xlabel("Owns Car")
plt.ylabel("No. of Customers") 
# Show the plot
plt.show()


# ### Feature Engineering

# In[105]:


#In the column has_gas, replace t with Tr`1ue or 1 and f with False or 0 . 
#This process is known as onehot encoding
NewCustomers_df["owns_car"] = NewCustomers_df["owns_car"].replace(["Yes", "No"],[1,0])


# In[106]:


NewCustomers_df["state"] = NewCustomers_df["state"].astype("category")


# In[107]:


# How many categories of channel sales are there?
pd.DataFrame({"Samples in category": NewCustomers_df["state"].value_counts()})
# Create 3 dummy variables for each variable
state_dummies = pd.get_dummies(NewCustomers_df["state"], prefix = "state")
# We rename columns for simplicity
state_dummies.columns = [col_name[:11] for col_name in state_dummies.columns]


# In[108]:


# Transform column into categorical data type
NewCustomers_df["job_industry_category"] = NewCustomers_df["job_industry_category"].astype("category")
# How many categories of channel sales are there?
pd.DataFrame({"Samples in category": NewCustomers_df["job_industry_category"].value_counts()})
# Create 3 dummy variables for each variable
job_industry_category_dummies = pd.get_dummies(NewCustomers_df["job_industry_category"], prefix = None)
# We rename columns for simplicity
job_industry_category_dummies.columns = [col_name[:11] for col_name in job_industry_category_dummies.columns]


# In[109]:


# Transform column into categorical data type
NewCustomers_df["wealth_segment"] = NewCustomers_df["wealth_segment"].astype("category")
# How many categories of channel sales are there?
pd.DataFrame({"Samples in category": NewCustomers_df["wealth_segment"].value_counts()})
# Create 3 dummy variables for each variable
wealth_segment_dummies = pd.get_dummies(NewCustomers_df["wealth_segment"], prefix = "wealth_segment")
# We rename columns for simplicity
wealth_segment_dummies.columns = [col_name[:11] for col_name in wealth_segment_dummies.columns]


# ### Merge the dummy variables to main dataframe

# In[110]:


NewCustomers_df = pd.merge(NewCustomers_df, state_dummies, left_index=True, right_index=True)
NewCustomers_df = pd.merge(NewCustomers_df, job_industry_category_dummies, left_index=True, right_index=True)
NewCustomers_df = pd.merge(NewCustomers_df, wealth_segment_dummies, left_index=True, right_index=True)


# In[111]:


# Let us remove the old categorical columns
NewCustomers_df.drop(columns=["state","job_industry_category","wealth_segment"],inplace=True)
NewCustomers_df


# In[112]:


fig, axs = plt.subplots(nrows=4, figsize=(18,50))
# Plot histograms
sns.distplot((NewCustomers_df["past_3_years_bike_related_purchases"].dropna()), ax=axs[0])
sns.distplot((NewCustomers_df["property_valuation"].dropna()), ax=axs[1])
sns.distplot((NewCustomers_df["DOB"].dropna()), ax=axs[2])
sns.distplot((NewCustomers_df["tenure"].dropna()), ax=axs[3])


# In[113]:


fig, axs = plt.subplots(nrows=4, figsize=(18,50))
# Plot histograms
sns.boxplot((NewCustomers_df["past_3_years_bike_related_purchases"].dropna()), ax=axs[0])
sns.boxplot((NewCustomers_df["property_valuation"].dropna()), ax=axs[1])
sns.boxplot((NewCustomers_df["DOB"].dropna()), ax=axs[2])
sns.boxplot((NewCustomers_df["tenure"].dropna()), ax=axs[3])


# In[114]:


NewCustomers_df.loc[NewCustomers_df["DOB"]<=20,"DOB"]= 0
NewCustomers_df.loc[(NewCustomers_df["DOB"]>=21)&(NewCustomers_df["DOB"]<=30),"DOB"]= 1
NewCustomers_df.loc[(NewCustomers_df["DOB"]>=31)&(NewCustomers_df["DOB"]<=40),"DOB"]= 2
NewCustomers_df.loc[(NewCustomers_df["DOB"]>=41)&(NewCustomers_df["DOB"]<=50),"DOB"]= 3
NewCustomers_df.loc[(NewCustomers_df["DOB"]>=51)&(NewCustomers_df["DOB"]<=60),"DOB"]= 4
NewCustomers_df.loc[NewCustomers_df["DOB"]>=61,"DOB"]= 5


# In[115]:


Existing_Customers_df.loc[Existing_Customers_df["DOB"]<=20,"DOB"]= 0
Existing_Customers_df.loc[(Existing_Customers_df
                           ["DOB"]>=21)&(Existing_Customers_df
                                                           ["DOB"]<=30),"DOB"]= 1
Existing_Customers_df.loc[(Existing_Customers_df
                           ["DOB"]>=31)&(Existing_Customers_df
                                                           ["DOB"]<=40),"DOB"]= 2
Existing_Customers_df.loc[(Existing_Customers_df
                           ["DOB"]>=41)&(Existing_Customers_df
                                                           ["DOB"]<=50),"DOB"]= 3
Existing_Customers_df.loc[(Existing_Customers_df
                           ["DOB"]>=51)&(Existing_Customers_df
                                                           ["DOB"]<=60),"DOB"]= 4
Existing_Customers_df.loc[Existing_Customers_df["DOB"]>=61,"DOB"]= 5


# In[116]:


# 1. Define a function to find the outliers 
def find_outliers_iqr(dataframe, column):
    
    col = sorted(dataframe[column])
    q1, q3= np.percentile(col,[25,75])
    iqr = q3 - q1
    lower_bound = q1 -(1.5 * iqr)
    upper_bound = q3 +(1.5 * iqr)
    
    results = {"iqr": iqr, "lower_bound":lower_bound, "upper_bound":upper_bound}
    return results

# 2. Define a function to remove the outliers found
def remove_outliers_iqr(dataframe, column):
   
    outliers = find_outliers_iqr(dataframe, column)
    
    removed = dataframe[(dataframe[column] < outliers["lower_bound"]) |
                        (dataframe[column] > outliers["upper_bound"])].shape
                        
        # | means OR

    dataframe = dataframe[(dataframe[column] > outliers["lower_bound"]) & 
                          (dataframe[column] < outliers["upper_bound"])]

    print("Removed:", removed[0], " outliers")
    return dataframe


# In[117]:


Existing_Customers_df = remove_outliers_iqr(Existing_Customers_df,"property_valuation")


# In[118]:


NewCustomers_df["property_valuation"] = NewCustomers_df["property_valuation"]+1
#NewCustomers_df["past_3_years_bike_related_purchases"] = NewCustomers_df["past_3_years_bike_related_purchases"]+1
#NewCustomers_df["DOB"] = NewCustomers_df["DOB"]+1
#NewCustomers_df["tenure"] = NewCustomers_df["tenure"]+1


# In[119]:


#Apply log10 transformation
NewCustomers_df["property_valuation"] = np.log10(NewCustomers_df["property_valuation"])
#NewCustomers_df["past_3_years_bike_related_purchases"] = np.log10(NewCustomers_df["past_3_years_bike_related_purchases"])
#NewCustomers_df["DOB"] = np.log10(NewCustomers_df["DOB"])
#NewCustomers_df["tenure"] = np.log10(NewCustomers_df["tenure"])


# In[120]:


correlation4 = NewCustomers_df.corr()
# Plot correlation
plt.figure(figsize=(19,15))
sns.heatmap(correlation4, xticklabels=correlation4.columns.values, 
            yticklabels=correlation4.columns.values, annot = True, annot_kws={'size':10})
# Axis ticks size
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show() 


# In[121]:


NewCustomers_df.drop(columns=["tenure"],inplace=True)


# In[122]:


NewCustomers_df


# ### Predicting the segments of the new customers

# In[123]:


# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)

NewCustomers_df_transform = pd.DataFrame(data = NewCustomers_df)
#features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
display(NewCustomers_df_transform.head(n = 5))


# In[126]:


# Make predictions with the classifier:
#pred_KNN_new = neigh.predict(NewCustomers_df)
#pred_NB_New = gnb.predict(NewCustomers_df)
pred_SVM_New = clf_B.predict(NewCustomers_df_transform)
df4 = pd.DataFrame(pred_SVM_New)
df5 = df4[0].value_counts()
df5


# In[130]:


df4 = pd.DataFrame(pred_SVM_New)
customer_segmentation = ['Mid Value','Low Value', 'High Value']
#, 'Intermediate Value', 'High Value']
df5 = df4[0].value_counts()

# Creating explode data
explode = (0.1, 0.1, 0.1)
  
# Creating color parameters
colors = ( "red", "green", "beige", "indigo", "white")
  
# Wedge properties
wp = { 'linewidth' : 1, 'edgecolor' : "black" }
  
# Creating autocpt arguments
def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d} customers)".format(pct, absolute)
  
# Creating plot
fig, ax = plt.subplots(figsize =(10, 7))
wedges, texts, autotexts = ax.pie(df5, 
                                  autopct = lambda pct: func(pct, df5),
                                  explode = explode, 
                                  labels = customer_segmentation,
                                  shadow = True,
                                  colors = colors,
                                  startangle = 90,
                                  wedgeprops = wp,
                                  textprops = dict(color ="black"))  
# Adding legend
ax.legend(wedges, customer_segmentation,
          title ="states",
          loc ="upper left",
          bbox_to_anchor =(1, 0, 0.5, 1))
  
plt.setp(autotexts, size = 8, weight ="bold")
ax.set_title("Customer Segmentation") 
# show plot
plt.show()


# In[ ]:




