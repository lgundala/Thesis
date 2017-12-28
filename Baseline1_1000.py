import pandas as pd
import numpy as np
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score,matthews_corrcoef
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn import linear_model, svm, gaussian_process
import sklearn.metrics as met
import xgboost as xgb
import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import *
from sklearn.metrics import accuracy_score
import random
from tabulate import tabulate
def robustSwap(x_tef, feature, rows1, rows0):
    for i in range(0,len(rows1)):
        x_tef.loc[rows1[i],feature] = x_tef.loc[rows0[i],feature];
    return x_tef;
def randomSwap(x_tef,rows1, rows0):
    j = 1;
    feature = random.randint(1,6);
    for i in range(0,len(rows1)):
        x_tef.loc[rows1[i],feature] = x_tef.loc[rows0[i],feature];
    
    j = feature + 8;
    while(j<609):
        for i in range(0,len(rows1)):
            x_tef.loc[rows1[i],j] = x_tef.loc[rows0[i],j];
        j = j+random.randint(1,6);
    j = (random.randint(1,6)-1)*5 + 610;
    while(j<640):
        for i in range(0,len(rows1)):
            x_tef.loc[rows1[i],j] = x_tef.loc[rows0[i],j];
        j = j+random.randint(1,6);
    return x_tef; 
def main():
    accuracy1 = 0;
    accuracy2 = 0;
    accuracy3 = 0;
    accuracy4 = 0;
    roc_accuracy1 = 0;
    roc_accuracy2 = 0;
    roc_accuracy3 = 0;
    roc_accuracy4 = 0;
    kappa1 = 0;
    matthews1 = 0;
    precision1_0 = 0;
    precision2_0 = 0;
    precision3_0 = 0;
    precision4_0 = 0;
    precision1_1 = 0;
    precision2_1 = 0;
    precision3_1 = 0;
    precision4_1 = 0;
    recall1_0 = 0;
    recall2_0 = 0;
    recall3_0 = 0;
    recall4_0 = 0;
    recall1_1 = 0;
    recall2_1 = 0;
    recall3_1 = 0;
    recall4_1 = 0;
    f1measure1_0 = 0;
    f1measure2_0 = 0;
    f1measure3_0 = 0;
    f1measure4_0 = 0;
    f1measure1_1 = 0;
    f1measure2_1 = 0;
    f1measure3_1 = 0;
    f1measure4_1 = 0;
    K = 10
    for i in range(0,10):
##        x_te = pd.read_csv("testing_English/test_M"+str(i)+".csv",delimiter=",", header=None, encoding="utf-8");
##        x_tr = pd.read_csv("testing_English/train_M"+str(i)+".csv",delimiter=",", header=None, encoding="utf-8");
        for j in range(0,1):
                x_te = pd.read_csv("testing_baselines/test.csv",delimiter=",", header=None, encoding="utf-8");
                x_tr = pd.read_csv("testing_baselines/train.csv",delimiter=",", header=None, encoding="utf-8");
                x_te_1 = x_te.loc[x_te[9]==1]
                x_te_0 = x_te.loc[x_te[9]==0]
                l = round((len(x_te_0.index))*0.001)
                rows = random.sample(list(x_te.loc[x_te[9]==1].index),l )
                x_te_11 = x_te_1.ix[rows];
                frames = [x_te_11, x_te_0];
                x_tef = pd.concat(frames);
                y_te = x_tef[9].values;               
                y_tr = x_tr[9].values;
                del x_tr[9];
                x = x_tr[[1]].values;
                X_tr=x[:,0:10];
                Y_tr = y_tr;
                x = x_tef[[1]].values;
                x_ter = x_tef[[1,9]];
                print(x_ter);
                feature = 1;
                x_ter_1 = x_ter.loc[x_ter[9]==1]
                x_ter_0 = x_ter.loc[x_ter[9]==0]
                l = round(len(x_ter_1)*0.1);
                print(l);
                rows1 = random.sample(list(x_ter_1.index),l);
                rows0 = random.sample(list(x_ter_0.index),l);
                del x_ter[9]
                for k in range(1,2):
                    x_ter =robustSwap(x_ter,feature, rows1, rows0);
                x = x_ter.values;
                X_te = x[:,0:10];
                Y_te = y_te;
                rf = RandomForestClassifier(n_estimators=100, criterion="entropy",random_state=4294967294)
                rf.fit(X_tr, Y_tr)
                predicted_digits = rf.predict(X_te)
                Y_p1 = rf.predict_proba(X_te)
                Y_p = predicted_digits
                accuracy1 = accuracy1+accuracy_score(Y_te, Y_p);
                print("random forests");
                print("accuracy",accuracy_score(Y_te, Y_p));
                precision1_0 = precision1_0 + precision_score(Y_te, Y_p, average='binary', pos_label=0);
                precision1_1 = precision1_1 + precision_score(Y_te, Y_p, average='binary', pos_label=1);
                recall1_0 = recall1_0 + recall_score(Y_te, Y_p, average='binary', pos_label=0);
                recall1_1 = recall1_1 + recall_score(Y_te, Y_p, average='binary', pos_label=1);       
                print("precision_0",precision_score(Y_te, Y_p, average='binary', pos_label=1));
                print("precision_1",precision_score(Y_te, Y_p, average='binary', pos_label=0));
                print("recall_0",recall_score(Y_te, Y_p, average='binary', pos_label=0));
                print("recall_1",recall_score(Y_te, Y_p, average='binary', pos_label=1));
                f1measure1_0 = f1measure1_0 + f1_score(Y_te, Y_p, average='binary', pos_label=0)
                f1measure1_1 = f1measure1_1 + f1_score(Y_te, Y_p, average='binary', pos_label=1)
                print("f1 score _0",f1_score(Y_te, Y_p, average='binary', pos_label=0));
                print("f1 score _1",f1_score(Y_te, Y_p, average='binary', pos_label=1));
                print(confusion_matrix(Y_te, Y_p));
                print("roc_auc_curve results");
                roc_accuracy1 = roc_accuracy1 + met.roc_auc_score(Y_te, Y_p1[:,1]);
                print(met.roc_auc_score(Y_te, Y_p));
                kappa1 = kappa1 + cohen_kappa_score(Y_te, Y_p);
                matthews1 = matthews1 + met.matthews_corrcoef(Y_te, Y_p);
##        sv = svm.SVC(kernel='linear', random_state = 2543484765, degree=3, cache_size = 40, class_weight = None)
##        sv.fit(X_tr, Y_tr)
##        predicted_digits = sv.predict(X_te)
##        Y_p = predicted_digits
##        accuracy2 = accuracy2+accuracy_score(Y_te, Y_p);
##        print(accuracy_score(Y_te, Y_p));
##        precision2_0 = precision2_0 + precision_score(Y_te, Y_p, average='binary', pos_label=0);
##        precision2_1 = precision2_1 + precision_score(Y_te, Y_p, average='binary', pos_label=1);
##        recall2_0 = recall2_0 + recall_score(Y_te, Y_p, average='binary', pos_label=0);
##        recall2_1 = recall2_1 + recall_score(Y_te, Y_p, average='binary', pos_label=1);       
##        print(precision_score(Y_te, Y_p, average='binary', pos_label=1));
##        print(precision_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(recall_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(recall_score(Y_te, Y_p, average='binary', pos_label=1));
##        f1measure2_0 = f1measure2_0 + f1_score(Y_te, Y_p, average='binary', pos_label=0)
##        f1measure2_1 = f1measure2_1 + f1_score(Y_te, Y_p, average='binary', pos_label=1)
##        print(f1_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(f1_score(Y_te, Y_p, average='binary', pos_label=1));
##        print(confusion_matrix(Y_te, Y_p));
##                logreg = linear_model.LogisticRegression(C=1e5)
##                logreg.fit(X_tr, Y_tr)
##                predicted_digits = logreg.predict(X_te)
##                Y_p = predicted_digits
##                accuracy3= accuracy3+accuracy_score(Y_te, Y_p);
##                print("logistics regression");
##                print("accuracy",accuracy_score(Y_te, Y_p));
##                precision3_0 = precision3_0 + precision_score(Y_te, Y_p, average='binary', pos_label=0);
##                precision3_1 = precision3_1 + precision_score(Y_te, Y_p, average='binary', pos_label=1);
##                recall3_0 = recall3_0 + recall_score(Y_te, Y_p, average='binary', pos_label=0);
##                recall3_1 = recall3_1 + recall_score(Y_te, Y_p, average='binary', pos_label=1);       
##                print("precision_0",precision_score(Y_te, Y_p, average='binary', pos_label=1));
##                print("precision_1",precision_score(Y_te, Y_p, average='binary', pos_label=0));
##                print("recall_0",recall_score(Y_te, Y_p, average='binary', pos_label=0));
##                print("recall_1",recall_score(Y_te, Y_p, average='binary', pos_label=1));
##                f1measure3_0 = f1measure3_0 + f1_score(Y_te, Y_p, average='binary', pos_label=0)
##                f1measure3_1 = f1measure3_1 + f1_score(Y_te, Y_p, average='binary', pos_label=1)
##                print("f1 score _0",f1_score(Y_te, Y_p, average='binary', pos_label=0));
##                print("f1 score _1",f1_score(Y_te, Y_p, average='binary', pos_label=1));
##                print(confusion_matrix(Y_te, Y_p));
##                print("roc_auc_curve results");
##                roc_accuracy3 = roc_accuracy3 + met.roc_auc_score(Y_te, Y_p);
##                print(met.roc_auc_score(Y_te, Y_p));
##                neigh = KNeighborsClassifier(n_neighbors=3)
##                neigh.fit(X_tr, Y_tr)
##                predicted_digits = neigh.predict(X_te)
##                Y_p = predicted_digits
##                accuracy4 = accuracy4+accuracy_score(Y_te, Y_p);
##                print("k-nearest neighbors");
##                print("accuracy",accuracy_score(Y_te, Y_p));
##                precision4_0 = precision4_0 + precision_score(Y_te, Y_p, average='binary', pos_label=0);
##                precision4_1 = precision4_1 + precision_score(Y_te, Y_p, average='binary', pos_label=1);
##                recall4_0 = recall4_0 + recall_score(Y_te, Y_p, average='binary', pos_label=0);
##                recall4_1 = recall4_1 + recall_score(Y_te, Y_p, average='binary', pos_label=1);       
##                print(precision_score(Y_te, Y_p, average='binary', pos_label=1));
##                print(precision_score(Y_te, Y_p, average='binary', pos_label=0));
##                print("recall_0",recall_score(Y_te, Y_p, average='binary', pos_label=0));
##                print("recall_1",recall_score(Y_te, Y_p, average='binary', pos_label=1));
##                f1measure4_0 = f1measure4_0 + f1_score(Y_te, Y_p, average='binary', pos_label=0)
##                f1measure4_1 = f1measure4_1 + f1_score(Y_te, Y_p, average='binary', pos_label=1)
##                print("f1 score _0",f1_score(Y_te, Y_p, average='binary', pos_label=0));
##                print("f1 score _1",f1_score(Y_te, Y_p, average='binary', pos_label=1));
##                print(confusion_matrix(Y_te, Y_p));
##                print("roc_auc_curve results");
##                roc_accuracy4 = roc_accuracy4 + met.roc_auc_score(Y_te, Y_p);
##                print(met.roc_auc_score(Y_te, Y_p));
##    print("accuracy : ",accuracy1/K);
##    print("precision: " ,precision1_0/K);
##    print("recall:",recall1_0/K);
##    print("f1 measure:",f1measure1_0/K);
##    print("precision: " ,precision1_1/K);
##    print("recall:",recall1_1/K);
##    print("f1 measure:",f1measure1_1/K);
##    print("roc_acu_curve:",roc_accuracy1/K);
####    print("accuracy : ",accuracy2/K);
####    print("precision: " ,precision2_0/K);
####    print("recall:",recall2_0/K);
####    print("f1 measure:",f1measure2_0/K);
####    print("precision: " ,precision2_1/K);
####    print("recall:",recall2_1/K);
####    print("f1 measure:",f1measure2_1/K);
##    print("accuracy : ",accuracy3/K);
##    print("precision: " ,precision3_0/K);
##    print("recall:",recall3_0/K);
##    print("f1 measure:",f1measure3_0/K);
##    print("precision: " ,precision3_1/K);
##    print("recall:",recall3_1/K);
##    print("f1 measure:",f1measure3_1/K);
##    print("roc_acu_curve:",roc_accuracy3/K);
##    print("accuracy : ",accuracy4/K);
##    print("precision: " ,precision4_0/K);
##    print("recall:",recall4_0/K);
##    print("f1 measure:",f1measure4_0/K);
##    print("precision: " ,precision4_1/K);
##    print("recall:",recall4_1/K);
##    print("f1 measure:",f1measure4_1/K);
##    print("roc_acu_curve:",roc_accuracy4/K);
    table = [
##        ["Random Forests","","","","","","","","","",""],
             ["Random Forests",accuracy1/K,roc_accuracy1/K,precision1_0/K,precision1_1/K,recall1_0/K,recall1_1/K,f1measure1_0/K,f1measure1_1/K, kappa1/K, matthews1/K]]
##             ["Logistic regression","","","","","","","","","",""],
##             ["",accuracy3/K,roc_accuracy3/K,precision3_0/K,precision3_1/K,recall3_0/K,recall3_1/K,f1measure3_0/K,f1measure3_1/K, kappa3/K, matthews3/K],
##             ["k-nearest neighbor","","","","","","","","","",""],
##             ["",accuracy4/K,roc_accuracy4/K,precision4_0/K,precision4_1/K,recall4_0/K,recall4_1/K,f1measure4_0/K,f1measure4_1/K, kappa4/K, matthews4/K]]
    print(tabulate(table, headers= ["Classifier","Accuracy","roc_auc_curve","precision_0","precision_1","recall_0","recall_1","f1 score_0","f1 score_1","kappa Score","Matthews score"]))
    f = open('Baseline1_1000_robust.txt','w')
    f.write(tabulate(table, headers= ["Classifier","Accuracy","roc_auc_curve","precision_0","precision_1","recall_0","recall_1","f1 score_0","f1 score_1","kappa Score","Matthews score"]))
    f.close()
##    print("kappa score",kappa1/K);
##    print("Matthews score",matthews/K);


##
##



if __name__=="__main__":
    main()
