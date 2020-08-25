# churn-prediction

## Background

This kind of analysis are done by companies to determine the likelihood that a customer will churn. This way the company can develop different kind of strategies to diversify promotions to keep customers and also estimate a percentaje of churn which can a be a loss of income for the company. 


## Goal of the analysis

- Find the golden features of the data set

- Train different models against the same data to test performance

- Have a fist look at the SHAP library of feature analysis


## Data set columns

![](imgs/data-columns.png?raw=true "Columns of the dataset")


## Exploratory Data Analysis

- Does the gender affects on the decision to churn?

![](imgs/barplot-gender.png?raw=true "Gender Barplot")

As we can see there is no difference between genders when it comes to churn the service.

- How about the Senior Citizens, do they tend to shurn more or less?

![](imgs/barplot-scitizen.png?raw=true "Senior citizens")

![](imgs/barplot-scitizen-churn.png?raw=true "Senior citizens churn")



We can observe that although there are not many Senior Citizens inside the data set, a considerable amount of them churn their service. 

Let's dive into more details here:

![](imgs/facetgrid-scitizen.png?raw=true "Senior citizens FG")

![](imgs/facetgrid-scitizen2.png?raw=true "Senior citizens FG")

Taking a look at this charts we can start to think that the feature PaymentMethod and COntract are very powerful to this analysis.

Let's see what happend with this two features but with no Senior Citizens.

![](imgs/barplot-contract.png?raw=true "Barplot Contract Churn")

Here is a huge finding, as we can see the most part of the churn is by month-to-month contract customers. This was one of my initial theories before analyzing the data and came up to be true.

On the other hand, we are going to see the PaymentMethod

![](imgs/barplot-pmethod.png?raw=true "Barplot Contract Churn")


![](imgs/rates.png?raw=true "Contract Churn Rates")

Another interesting finding here, the electronic check was the one with most considerable churn rate. Before making any analysis I thought that people with automatic payment should churn less due to people tend to forget what they are paying this also came up to be true.

Next we want to analyze the tenure whi stands for how long the clients have been actively paying the service in terms of months

![](imgs/distplot-tenure.png?raw=true "Distplot Tenure")

![](imgs/distplot-tenure-churn.png?raw=true "Distplot Tenure Churn")

An interesting observation here, in the first distribution plot we see that the majority of the clients in this data have been clients for less than a year and also the following majority is for clients that have been paying for more than 5 years. 

And if we take a look to the second one distribution plot we can see in red the clients that did churn and in green the ones that did not. Seems obvious that long-term clients tend to churn less that short-term clients.

Let's take a look to the Dependents and Partners features.

![](imgs/facetgrid-partners-dependants.png?raw=true "Partners and dependents")

We can see that people who has not a partner or dependents tend to have a greater churn rate. My thoughts on this behaviours are that people who are not attached to anyone tend to have more freedom on their decisions so this could be the reason.

Another theory I had in mind was the fact that if the client is recieving a paper billing instead of a electronic bill it would have a greater churn rate because the paper billing arrives at your mailing and it is like a constant reminder that you are still paying the service. On the other hand the paperless bill could be an e-mail that the customer even notice.

![](imgs/barplot-paperless.png?raw=true "Paperless churn")

But it came up to be false :)

Last feature we are going to analyze is the MonthlyCharges.

![](imgs/distplot-monthlycharges.png?raw=true "Monthly Charges churn")

The first thing we noticed is that at lower charges, lower the churn rate but this could be biased by the fact that we are not counting how many services the client had payed.

![](imgs/correlation-matrix.png?raw=true "CORR MATRIX")

Taking a look at the correlation matrix of the data frame we can see the square in the middle refering to all the features that involve what kind of service the client have. We are not going to analyze any of this features and we will proceed to train different models without feature engineering since this is not the goal of the analysis.

## Creating a baseline model

Since this is a classification problem we are going to train different algorithms ans see the perfomance of each one. (Catboost, XGBoost, LightGBM and Random Forest)

The first training we are going to test the models out-of-the-box with no parameters and see the results.

### Metrics

![](imgs/metrics-baseline.png?raw=true "Metrics baselines")

### Confusion Matrix

![](imgs/confusionm-baselinemodels.png?raw=true "Baseline Confusion Matrix")


*Reading left to right the order is the following one (Catboost,XGBoost, Light GBM, RandomForest)*

### Roc-Curve

![](imgs/roccurve-baselinemodels.png?raw=true "ROC CURVE BL")

At first glance, Catboost was the best out-of-the-box performing model scoring aprox 82% of accuracy and also the best AUC score.

If you take a look to the Confusion Matrix, Catboost also have the lowest True Negative and True Positive rate. And here it depends on the strategy you want to use after this prediction. If you are trying to predict the churn rate as a way of employ some kind of price strategy on the ones that will churn to keep them as your clients you definetly have to take a look at your TN rate and your TP rate. If your TN rate is high you are wasting money in keeping clients that maybe did not want to cancel the service and if your TP rate is high you are loosing clients and you are not doing anything to keep them.


### Shap Values

##### Shap Definition

[SHAP (SHapley Additive exPlanations)](https://github.com/slundberg/shap) is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions. 

##### Results

![](imgs/shap-values.png?raw=true "SHAP VALUES")

Now looking at the plot provided by SHAP we can start validating some thoughts we made in the EDA. For example as you can see the gender was not significant to the model. On the other hand, the tenure, type of contract, InternetService and total charges were the features with more importance for the model. In this case SHAP also help us to find a feature that we did not analyze in the EDA, InternetService.


### Grid Search

We are going to make a basic grid search to see if we can get better results.

#### Parameters


####  Metrics

![](imgs/metrics-gs.png?raw=true "Metrics gs")


#### Confusion matrix

![](imgs/confusionm-gs.png?raw=true "GS Confusion Matrix")

#### Roc-Curve

![](imgs/roccurve-gs.png?raw=true "ROC CURVE GS")

After perfoming the Grid Search Catboost was the one with the slightest change in the perfomance. The grid search help the other three models to increase their performance but Catboost stills perfoming better and this analysis demonstrates the high performance of gradient boosting algorithms and also the superiority of Catboost perfoming so well without setting any parameter. 
