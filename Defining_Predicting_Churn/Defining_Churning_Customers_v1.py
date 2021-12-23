#!/usr/bin/env python
# coding: utf-8

# #### Created By : Ashwini Kumar
# #### Dated : 13th March 2020
# #### Objective : 
#     1. To define metrics to identify disengaged customers.Which will also include customer churned in that period
#     2. Create a ML model to predict customers already churned 
#     3. Defining a stratergy to target customers as soon as we identify them
# #### Methodology :
#     We will define the disengaged customers based on three metrics
#         i. Customer who don't have single transaction in last 4 months 
#         ii.Customers whose number of transaction are decreasing very month for continuously for 3 months
#         iii.Customers whose transaction amount are decreasing very month for continuously for 3 months
#     All the customers which qualify for any above metrics is considered as Disenageged customer
#     It will be a imbalanced class problem so we need to deal with that also if we don't get a good model in first scenarios
#     Also trying to understand which variables should be used as feature importance
# #### Suggesting Next Steps : As all ideas can't be tried in limited time
# #### Suggesting Required : Suggesting what would be deployment strategy for the model
# #### Suggesting Steps to target about to churn customer with some stratergy

# In[378]:


### Import all the required packages to be used for furthar analysis
import pandas as pd
import os
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from IPython.display import display


# In[379]:


### Change the locations for current working directory
os.getcwd()


# In[380]:


### Read the file required and parse the datetime in the right format to be used further
mydateparser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")
trans = pd.read_csv('C:\\Users\\ash\\Desktop\\data_revolut\\data\\rev-transactions.csv', parse_dates=['created_date'], date_parser=mydateparser)


# In[382]:


### Print the statistics about the data read
print ("The shape of data is as follows :", trans.shape)
print ("Give the information on the data required :",trans.info())
print ("Describe the data :", trans.describe())
print ("Describe the data :",trans.describe(include ='object'))


# In[383]:


#### Extract the time related variables for datetime. Extract month, year and other variables to be summarized
trans['year'] = pd.DatetimeIndex(trans['created_date']).year
trans['quarter'] = pd.DatetimeIndex(trans['created_date']).quarter
trans['month'] = pd.DatetimeIndex(trans['created_date']).month
trans['year_quarter'] = trans['year'].astype(str) +  trans['quarter'].astype(str)
trans['month_year'] = trans['year'].astype(str) +  trans['month'].astype(str)


# In[384]:


trans.head(5)


# #### Lets only consider transaction which have following transactions so that we exclude incompleted transactions

# In[385]:


print ("Transactions with all transaction irrespective of transaction state",trans.shape)
trans_comp = trans[trans['transactions_state'] == 'COMPLETED']
print ("Transactions with only transaction state",trans_comp.shape)


# In[386]:


### Look at the distribution of transactions type after filtering out transactions state is completed
trans_comp['transactions_type'].value_counts()


# ### Lets Only consider the transactions which have customer did intentionally so we will remove folllwing transactions type
# 1. FEE
# 2. CARD_REFUND
# 3. TAX
# 4. REFUND

# In[387]:


trans_analysis= trans_comp[~trans_comp['transactions_type'].isin(['FEE','CARD_REFUND','TAX','REFUND'])]
print ("Shape of transaction data after filetring is :",trans_analysis.shape)


# ### Lets summarize the data by transaction month by user id
# ### To create following varables at month level
# ##### 1. Sum of USD Transactions Amount
# ##### 2. Number of Transactions in a Month
# ##### 3. Number of distinct cities where transactions was done
# ##### 4. Number of ditinct merchants on which transactions is done
# ##### 5. Number of distinct  types of transactions

# In[388]:


##Step 1: Group by User Id and Months as we will be used in Further analysis. 
### If we had time we will also want to add variables summary at quarter level
groupby1 = trans_analysis.groupby(['user_id','month_year'])

##Step 2: Calculate different aggregations to create variables defined above
trans_count = groupby1['transaction_id'].count()
amt_sum = groupby1['amount_usd'].sum()
dist_currency = groupby1['transactions_currency'].nunique()
dist_city = groupby1['ea_merchant_city'].nunique()
dist_country = groupby1['ea_merchant_country'].nunique()
dist_trans =  groupby1['transactions_type'].nunique()


##Step 3: Combine into a single DataFrame
trans_sum = pd.DataFrame({'trans_count':trans_count,'amt_sum':amt_sum,'dist_currency':dist_currency,'dist_city':dist_city,
                          'dist_country':dist_country,'dist_trans':dist_trans}).reset_index()


# In[389]:


### Group by always creates dataframes by hierarchical Index and columns names
### In below steps we will change them to the normal dataframes with non-hierarchical index
trans1 = trans_sum.set_index(['user_id', 'month_year'])
trans2 = trans1.unstack(level =1)


# In[390]:


### Join the levels to create alist of columns names in list and assign it to columm names
new_cols = [''.join(t) for t in t1.columns]
print ("The number of columns post aggreagtion is :",len(new_cols))
trans2.columns = new_cols
trans2 =trans2.reset_index()


# In[391]:


print ("Have alook at the data and check if its aligned with expectation i.e. Non Hierachical index")
trans2.head(4)


# #### In this data missing values means that they have no trasaction in that month. Replace it with zeroes

# In[392]:


trans2 = trans2.fillna(0)


# #### Customer at risk or disnegaged would have transaction amount going down for 3 months continuoulsy

# In[393]:


risky_avg_transaction = []
for idx,row in trans2.iterrows():
    if  (row['amt_sum20191'] > row['amt_sum20192']) & (row['amt_sum20192'] > row['amt_sum20193']) & (row['amt_sum20193'] > row['amt_sum20194']):
        risky_avg_transaction.append(idx)
        
print (len(risky_avg_transaction))

trans2['risky_by_avg_tran'] = np.where(trans2.index.isin(risky_avg_transaction),1,0)


# #### Customer at risk flag or disnegaged for customers whose transaction count is going down month by month for 201902 to 201904

# In[395]:


risky_count_transaction = []
for idx,row in trans2.iterrows():
    if  (row['trans_count20191'] > row['trans_count20192']) & (row['trans_count20192'] > row['trans_count20193']) & (row['trans_count20193'] > row['trans_count20194']):
        risky_count_transaction.append(idx)
        
print (len(risky_count_transaction))

trans2['risky_by_count_tran'] = np.where(trans2.index.isin(risky_count_transaction),1,0)


# #### Customer at risk flag or disengaged for customers who have no transaction in last 3 months but are associated for longer tahn 3 months

# In[397]:


risky_no_transaction = []
for idx,row in trans2.iterrows():
    if  (row['trans_count20191'] == 0) & (row['trans_count20192'] ==0) & (row['trans_count20193'] == 0) & (row['trans_count20194'] == 0) & (row['trans_count201812'] > 0)  :
        risky_no_transaction.append(idx)
        
print (len(risky_no_transaction))

trans2['risky_by_no_transaction'] = np.where(trans2.index.isin(risky_no_transaction),1,0)


# ### Create target varibles for model by combining three flags created before

# In[398]:



trans2['target'] = trans2.apply(lambda x: 1 if (x.risky_by_no_transaction==1) or (x.risky_by_count_tran==1) or (x.risky_by_avg_tran == 1)  else 0,axis=1)
target = trans2["target"].value_counts()
target


# ### Define the variable list to be used in model along with the target variables

# In[399]:


model_variables_list = ["user_id","trans_count20181","trans_count201810","trans_count201811","trans_count201812","trans_count20182","trans_count20183","trans_count20184","trans_count20185","trans_count20186","trans_count20187","trans_count20188","trans_count20189","amt_sum20181","amt_sum201810","amt_sum201811","amt_sum201812","amt_sum20182","amt_sum20183","amt_sum20184","amt_sum20185","amt_sum20186","amt_sum20187","amt_sum20188","amt_sum20189","dist_currency20181","dist_currency201810","dist_currency201811","dist_currency201812","dist_currency20182","dist_currency20183","dist_currency20184","dist_currency20185","dist_currency20186","dist_currency20187","dist_currency20188","dist_currency20189","dist_city20181","dist_city201810","dist_city201811","dist_city201812","dist_city20182","dist_city20183","dist_city20184","dist_city20185","dist_city20186","dist_city20187","dist_city20188","dist_city20189","dist_country20181","dist_country201810","dist_country201811","dist_country201812","dist_country20182","dist_country20183","dist_country20184","dist_country20185","dist_country20186","dist_country20187","dist_country20188","dist_country20189","dist_trans20181","dist_trans201810","dist_trans201811","dist_trans201812","dist_trans20182","dist_trans20183"
                        ,"dist_trans20184","dist_trans20185","dist_trans20186","dist_trans20187","dist_trans20188","dist_trans20189","target"]

model_df =trans2[model_variables_list]


# In[400]:


### Now model df has all the variables which we need for further analysis
### Get in the demographics variables of user


# In[401]:


users_data = pd.read_csv(('C:\\Users\\ash\\Desktop\\data_revolut\\data\\rev-users.csv'))
print ("Shape of users data is",users_data.shape)
print (users_data['user_id'].nunique())


# In[402]:


## As we have distinct user id it is safe to join it to users table
print ("Shape of model data before merge is",model_df.shape)
model_df = pd.merge(model_df,users_data,how = 'inner',on = 'user_id')
print ("Shape of model data after merge is ", model_df.shape)


# In[403]:


### Get in the device usage of users 
users_device = pd.read_csv(('C:\\Users\\ash\\Desktop\\data_revolut\\data\\rev-devices.csv'))
print ("Shape of users data is",users_device.shape)
print (users_device['user_id'].nunique())


# In[404]:


## As we have distinct user id it is safe to join it to users table
print ("Shape of model data before merge is",model_df.shape)
model_df = pd.merge(model_df,users_device,how = 'inner',on = 'user_id')
print ("Shape of model data after merge is ", model_df.shape)


# In[405]:


### Drop the variables you don't want to use right now
### referrals as it has all zero values and country and city as they would have lots of level which we dont hae time to deal with
drop_var = ['country','city','num_referrals','num_successful_referrals']
model_df.drop(drop_var,inplace = True, axis = 1)
model_df.shape


# In[406]:


### Lets do some feature engineering on other variables 
### Convert the Birth year into age groups
bins = [0,25, 35, 50, 59,60]
labels = ['less_25', '25-35', '35-50', '50-59', '60+']
model_df['agerange'] = pd.cut(2020 - model_df['birth_year'], bins, labels = labels,include_lowest = True)
model_df.drop(['birth_year'],axis = 1, inplace = True)


# In[407]:


### Lets create some features on User Create year 
### We will exlude customer which have joined after 2018 actober as we will not ahve much transaction data for them
import datetime
model_df['days_joined'] = (datetime.date(2018, 10, 31) - pd.DatetimeIndex(model_df['created_date']).date)
model_df['days_joined'] = model_df['days_joined'] / np.timedelta64(1, 'D')


# In[408]:


### Exclude all the custoemr from analysis which ahve joined after october 31
model_df = model_df[model_df['days_joined']>0]


# In[409]:


### Create variable on joining date as reqired
bins = [0,30, 90, 180,999]
labels = ['less_month', 'month_quarter', 'quarter_halfyear', 'more_halfyear']
model_df['joinedrange'] = pd.cut(model_df['days_joined'], bins, labels = labels,include_lowest = True)
model_df.drop(['days_joined','created_date'],axis = 1, inplace = True)


# In[410]:


### Our data is ready for modelling
model_df.shape


# ### Convert the categorical varible to one hot dummy encoding

# In[411]:


cat_variables_list = ['plan','brand','agerange','joinedrange','attributes_notifications_marketing_push',
'attributes_notifications_marketing_email','user_settings_crypto_unlocked']


# In[412]:


model_data = pd.get_dummies(model_df,columns = cat_variables_list)
print ("Number of variable after dummy treatments is",model_data.shape)
print ("Columns are as follow",model_data.columns)


# In[413]:


#### Create the target variable and delete unnecessary columns
target = model_data["target"]
model_data.drop(['target','user_id'],inplace=True,axis=1)
model_data.head()


# In[414]:


### divide the data in 80 & 20 and create a modelling dataset


# In[ ]:





# In[415]:


from sklearn.model_selection import train_test_split
# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(model_data, 
                                                    target, 
                                                    test_size = 0.2, 
                                                    random_state = 0,stratify = target)

# Show the results of the split
print ("Training set has  samples", X_train.shape[0])
print ("Testing set has  samples  ",X_test.shape[0])


# In[416]:


### Create a simple & dumb base line model 


# In[417]:


### Base Line Model : If our model predicted everything as Risky
TP = np.sum(target) # Counting the ones as this is the naive case. 

FP = target.count() - TP # Specific to the naive case

TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case
print (TP,FP)
# TODO: Calculate accuracy, precision and recall

accuracy = float(TP)/float(target.count())
recall = 1
precision = accuracy
beta= 0.5


# The formula for fbeta score can be written as (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
fscore = (1 + 0.25) * (precision * recall) / ((0.25 * precision) + recall)

# Print the results 
print ("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))


# In[418]:


### Building a train predictor to compare multiple models¶
### Here metrics that I am going to use to compare different models is F1 score, accuracy
### More prefrence give  to F score as it is imbalanced class problem
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer, precision_recall_curve,confusion_matrix
from sklearn import  metrics

import time
def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 

    results = {}
    
  
    start = time.time()
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time.time()
    
    results['train_time'] = end - start
    
    start = time.time()
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train)
    end = time.time()
    
    results['pred_time'] = end - start
    results['acc_train'] = accuracy_score(y_train, predictions_train)
        
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    results['f_train'] = fbeta_score(y_train, predictions_train, 0.5,average='weighted')
        
    results['f_test'] = fbeta_score(y_test, predictions_test, 0.5,average='weighted')
    matrix =  confusion_matrix(y_test, predictions_test)
    TP =  matrix[0][0]
    FP = matrix[0][1]  # Specific to the naive case

    TN = matrix[1][1] # No predicted negatives in the naive case
    FN = matrix[1][0] # No predicted negatives in the naive case

    # TODO: Calculate accuracy, precision and recall
    accuracy = metrics.accuracy_score(y_test, predictions_test)
    recall = float(TP )/  float(TP + FN)
    precision = float(TP )/ float(TP + FP)
    #print "precision: {}, recall: {}".format(recall, precision) 
    beta = 0.5
    
    # TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
    # HINT: The formula above can be written as (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    fscore = (1 + 0.25) * (precision * recall) / ((0.25 * precision) + recall)
    print ("Test fscore with beta 0.5 : ", fscore )  
    print ("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
    dataframe = pd.DataFrame(predictions_train,y_train)  
    dataframe.to_csv(learner.__class__.__name__+".txt")
    # Return the results
    return results


# ### Try out various classification model and use one of the best model for tuning

# In[420]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import warnings
warnings.filterwarnings("ignore")

# TODO: Initialize the three models
clf_A =  linear_model.LogisticRegressionCV(solver='lbfgs',random_state=40)
clf_B = KNeighborsClassifier()
clf_C = RandomForestClassifier(random_state=40)
#clf_D = RandomForestClassifier(random_state = 40)

samples_100 = len(y_train)
samples_10 = int(samples_100 * 0.1)
samples_1 = int(samples_100 * 0.01)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1,samples_10,samples_100]):
        results[clf_name][i]=train_predict(clf, samples, X_train, y_train, X_test, y_test)


# #### Random forest and K nearest are the best models. But we will use Random forest as it is faster and scalable on bigger data also
# #### We will try to use Gridsearch cv to do some hyperparamter tuning on random forest 

# In[434]:


# Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [200,400,600,800,1000]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(2, 10, num = 1)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,20,50]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,8,12,20,40]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf =  RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 2, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
# rf_random.fit(X_train, y_train)
# print (rf_random.best_params_)
# # Make predictions using the unoptimized and model
# best_random = rf_random.best_estimator_

# predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_random.predict(X_test)

# Report the before-and-afterscores

print ("\nOptimized Model\n------")
# print ("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions )))
# print ("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions , beta = 0.5,average='weighted')))
print(classification_report(y_test, predictions))


# ### Without handling imbalance class our model si predicting evrything to be 1 so it is a bad model we will tackle imbalanced class now

# Using balanced RandomForestClassifier approach:
# Weights are inversely proportional with the frequency of class observaton:
#                                             
#                                             wj=nj / knj
#  
# where  wj  is the weight to class  j ,  n  is the number of observations,  nj  is the number of observations in class  j 
# , and  k  is the total number of classes. In our case, that indicates that the minority class label (fraud) should be 
# weighted higher.

# In[443]:



from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [200,400,600,800,1000]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [4,5,6,7,8,9,10]

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,20,50]
# Minimum number of samples required at each leaf node
min_samples_leaf = [4,8,12,20,40]
# Method of selecting samples for training each tree
bootstrap = [True, False]
class_weight = ['balanced','balanced_subsample']
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'class_weight':class_weight
              }
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf =  RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 500, cv = 2, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)
print (rf_random.best_params_)
# Make predictions using the unoptimized and model
best_random = rf_random.best_estimator_

# predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_random.predict(X_test)

# Report the before-and-afterscores

print ("\nOptimized Model\n------")
# print ("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions )))
# print ("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions , beta = 0.5,average='weighted')))
print(classification_report(y_test, predictions))


# #### Class weighting did not work we will downsample data

# In[ ]:




