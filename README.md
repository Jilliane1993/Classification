# Predicting Term Deposit Sign-up

I used several machine learning algorithms to predict term deposit sign up using UCI's bank marketing data set. The goal of predicting who will sign up is to overall increase efficency of the telemarketing campaign by focusing more on potential clients likely to sign up while decreasing the amount of time spent on contacting those unlikely to create an account. A summary of the process of my modeling, model selection, and interpretation of this data can be found in [this presentation](Bank_Marketing_presentation.pdf).

## SQL, Exploration, Data Cleaning:

I first created a postgresql database to complete some initial data exploration such as looking at the point at which further contact during the telemarketing campaign ceased to yield any term deposit sign-ups and which professions had the highest amount of sign-ups. I checked for null values and droped duplicate rows as well as began some feature engineering. UCI states the observations are ordered by date from May 2008-November 2010 so I was able to add in the year for each observation. I also created new features by binning existing features such as turning individual ages into age groups. As part of my data exploration I also graphed mulitple features looking at the imbalance between the number of people who did and did not sign up for a term deposit. My postgresql database creation, data exploration, and data cleaning can be found in [this notebook](Bank_sql.ipynb)

## Dealing with class imbalance:

1. Stratified test/train split
2. Balancing Class Weights
3. SMOTE, Random oversampling, Random undersampling

Since there was slightly less than 8 times as many people that declined signing up for a term deposit than those who did the test/train splot was stratified for each model. While I primarily used balanced class weights as a method to combat class imbalance I also tested SMOTE, random oversampling, and random undersampling with several of the models.

## Models compared:

1. [Logistic Regression](Bank_sql.ipynb)
2. [Bernoulli Naive Bayes](Bank_Naive_Bayes.ipynb)
3. [Random Forest](Bank_RF.ipynb)
4. [Balanced Random Forest](Bank_RF.ipynb)
5. [XGBoost](XGBoost.ipynb)
6. [CatBoost](Catboost.ipynb)

[Direct comparison of ROC AUC model performance](ROC_AUC.ipynb)

[Main functions and pipelines used](jne_functions.py)

## CatBoost Classifier:

After tuning hyperparameters and testing different methods to combat class imabalance with the various models the CatBoost Classifier preformed the best. I used ROC AUC score, the model's confusion matrix, and f1 macro score as my metrics to gauge performance. While the CatBoost had the highest ROC AUC score, its f1 macro and accuracy scores were very similiar or even lower than some of the other models; however it had a higher percentage of true positive results while maintaining a lower rate of false positives than other models. I used a correlation heatmap to infer the directionality of feature importance for this model. 
