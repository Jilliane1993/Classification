#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 18:05:46 2021

@author: jillian
"""

def classifiers_2(model,X, y):
    """ creating a pipeline that scales numeric features, onehotencodes categoricals, transforms, and puts into pipeline with classifier model; function does test train split, conducts random search with cross validation for best hyperparameters, fits model on training data, returns best parameters"""
    from sklearn.metrics import f1_score, make_scorer
    f1 = make_scorer(f1_score , average='macro')
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.model_selection import train_test_split          
    global pipe, X_train, X_test, y_train, y_test
    X=X
    y=y
    numeric_features=['campaign','previous','emp.var.rate',
                      'cons.price.idx','cons.conf.idx','euribor3m','nr.employed']
    numeric_transformer= Pipeline(steps=[ ('ss',StandardScaler())])

    categorical_features=['marital','education','default','contact','employment','month',
                          'housing','loan','day_of_week','age_group','poutcome','year']
    categorical_transformer=OneHotEncoder(handle_unknown='error',drop='first')

    preprocessor= ColumnTransformer(
        transformers=[
            ('num',numeric_transformer,numeric_features),
            ('cat', categorical_transformer,categorical_features)
        ])

    pipe=Pipeline(steps=[('prep',preprocessor),
                        ('classifier',model)])

    X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=1, stratify=y)




def make_confusion_matrix(model, threshold=0.5):
    """creating confustion matrix"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import numpy as np
    
    y_predict = (model.predict_proba(X_test)[:, 1] >= threshold)
    deposit_confusion = confusion_matrix(y_test, y_predict)
    plt.figure(dpi=80)
    group_counts=['{0:0.0f}'.format(value) for value in deposit_confusion.flatten()]
    group_perc=['{0:.2%}'.format(value) for value in
                deposit_confusion.flatten()/np.sum(deposit_confusion)]
    labels=[f'{v1}\n{v2}' for v1,v2 in zip(group_counts,group_perc)]
    labels=np.asarray(labels).reshape(2,2)
    sns.heatmap(deposit_confusion, cmap=plt.cm.Blues, annot=labels, square=True, fmt='',
                xticklabels=['No', 'Yes'],
                yticklabels=['No', 'Yes']);
    plt.xlabel('Prediction').set_color('black')
    plt.ylabel('Actual').set_color('black')
    plt.tick_params(colors='black')
    
    
def classifiers_no_stan(model, X, y):
    """ creating a pipeline that onehotencodes categoricals, transforms, and puts into pipeline with classifier model; function does test train split, conducts random search with cross validation for best hyperparameters, fits model on training data, returns best parameters"""
    from sklearn.metrics import f1_score, make_scorer
    f1 = make_scorer(f1_score , average='macro')
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split         
    global pipe, X_train, X_test, y_train, y_test
    X=X
    y=y

    categorical_features=['marital','education', 'default','contact','employment','month',
                          'housing','loan','day_of_week','age_group','poutcome','year']
    categorical_transformer=OneHotEncoder(handle_unknown='error',drop='first')

    preprocessor= ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer,categorical_features)
        ])

    pipe=Pipeline(steps=[('prep',preprocessor),
                        ('classifier',model)])

    X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=1, stratify=y)


def classifiers_no_stan_oversample(model, X, y):
    """ creating a pipeline that onehotencodes categoricals, transforms, and puts into pipeline with classifier model; function does test train split, conducts random search with cross validation for best hyperparameters, fits model on over sampled training data, returns best parameters"""
    from sklearn.metrics import f1_score, make_scorer
    f1 = make_scorer(f1_score , average='macro')
    from sklearn.compose import ColumnTransformer
    from imblearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split           
    global pipe, X_train, X_test, y_train, y_test
    from imblearn.over_sampling import SMOTE
    X=X
    y=y

    categorical_features=['marital','education','default','contact','employment','month',
                          'housing','loan','day_of_week','age_group','poutcome','year']
    categorical_transformer=OneHotEncoder(handle_unknown='error',drop='first')

    preprocessor= ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer,categorical_features)
        ])

    pipe=Pipeline(steps=[('prep',preprocessor),
                        ('sampling',SMOTE()),
                        ('classifier',model)])

    X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=1, stratify=y)
    


def classifiers_no_stan_undersample(model, X, y):
    """ creating a pipeline that onehotencodes categoricals, transforms, and puts into pipeline with classifier model; function does test train split, conducts random search with cross validation for best hyperparameters, fits model on over sampled training data, returns best parameters"""
    from sklearn.metrics import f1_score, make_scorer
    f1 = make_scorer(f1_score , average='macro')
    from sklearn.compose import ColumnTransformer
    from imblearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split          
    global pipe, X_train, X_test, y_train, y_test
    from imblearn.under_sampling import RandomUnderSampler
    X=X
    y=y

    categorical_features=['marital','education','default','contact','employment','month',
                          'housing','loan','day_of_week','age_group','poutcome','year']
    categorical_transformer=OneHotEncoder(handle_unknown='error',drop='first')

    preprocessor= ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer,categorical_features)
        ])

    pipe=Pipeline(steps=[('prep',preprocessor),
                        ('sampling',RandomUnderSampler(sampling_strategy='majority')),
                        ('classifier',model)])

    X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=1, stratify=y)



