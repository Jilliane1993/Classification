#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 18:05:46 2021

@author: jillian
"""

def classifiers_2(model, param_grid, X, y):
    from sklearn.metrics import f1_score, make_scorer
    f1 = make_scorer(f1_score , average='macro')
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.model_selection import train_test_split, RandomizedSearchCV            
    global rand_s, X_train, X_test, y_train, y_test
    X=X
    y=y
    numeric_features=['age','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']
    numeric_transformer= Pipeline(steps=[ ('ss',StandardScaler())])

    categorical_features=['job','marital','education','default','housing','loan','month','day_of_week','age_group']
    categorical_transformer=OneHotEncoder(handle_unknown='error',drop='first')

    preprocessor= ColumnTransformer(
        transformers=[
            ('num',numeric_transformer,numeric_features),
            ('cat', categorical_transformer,categorical_features)
        ])

    pipe=Pipeline(steps=[('prep',preprocessor),
                        ('classifier',model)])

    X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=1, stratify=y)

    rand_s=RandomizedSearchCV(pipe,param_grid,cv=5, scoring=f1, n_iter=25)
    rand_s.fit(X_train,y_train)
    return rand_s.best_params_



def make_confusion_matrix(model, threshold=0.5):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    y_predict = (model.predict_proba(X_test)[:, 1] >= threshold)
    deposit_confusion = confusion_matrix(y_test, y_predict)
    plt.figure(dpi=80)
    sns.heatmap(deposit_confusion, cmap=plt.cm.Blues, annot=True, square=True, fmt='d',
                xticklabels=['no', 'yes'],
                yticklabels=['no', 'yes']);
    plt.xlabel('prediction')
    plt.ylabel('actual')
    
    



