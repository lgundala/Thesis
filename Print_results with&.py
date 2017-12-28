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
import matplotlib.pyplot as plt
from itertools import cycle
import random
def main():
    accuracy1 = 0;
    accuracy2 = 0;
    accuracy3 = 0;
    accuracy4 = 0;
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
        x_tr = pd.read_csv("English/test_M"+str(i)+".csv",delimiter=",", header=None, encoding="utf-8");
        x_te = pd.read_csv("English/train_M"+str(i)+".csv",delimiter=",", header=None, encoding="utf-8");
##        x_tr_1 = x_tr.loc[x_tr[8]==1]
##        x_tr_0 = x_tr.loc[x_tr[8]==0]
##        l = round((len(x_tr_1.index))*0.1)
##        rows = random.sample(list(x_tr_1),l )
##        x_tr_11 = x_tr_1.ix[rows];
##        frames = [x_tr_11, x_tr_0];
##        x_trf = pd.concat(frames);
##        x_te_1 = x_te.loc[x_te[8]==1]
##        x_te_0 = x_te.loc[x_te[8]==0]
##        l = round((len(x_te_1.index))*0.01)
##        rows = random.sample(list(x_te_1),l )
##        x_te_11 = x_te_1.ix[rows];
##        frames = [x_te_11, x_te_0];
##        x_tef = pd.concat(frames);
        y_te = x_te[3].values;
        del x_te[3]
        y_tr = x_tr[3].values;
        del x_tr[3];
        del x_te[0];
        del x_tr[0];
        x = x_tr.values;
        X_tr=x[:,1:640];
        Y_tr = y_tr;
        x = x_te.values;
        X_te = x[:,1:640];
        Y_te = y_te;
##        rf = RandomForestClassifier(n_estimators=100, criterion="entropy",random_state=0)
##        rf.fit(X_tr, Y_tr)
##        predicted_digits = rf.predict(X_te)
##        Y_p = predicted_digits
##        accuracy1 = accuracy1+accuracy_score(Y_te, Y_p);
##        print(accuracy_score(Y_te, Y_p));
##        precision1_0 = precision1_0 + precision_score(Y_te, Y_p, average='binary', pos_label=0);
##        precision1_1 = precision1_1 + precision_score(Y_te, Y_p, average='binary', pos_label=1);
##        recall1_0 = recall1_0 + recall_score(Y_te, Y_p, average='binary', pos_label=0);
##        recall1_1 = recall1_1 + recall_score(Y_te, Y_p, average='binary', pos_label=1);       
##        print(precision_score(Y_te, Y_p, average='binary', pos_label=1));
##        print(precision_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(recall_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(recall_score(Y_te, Y_p, average='binary', pos_label=1));
##        f1measure1_0 = f1measure1_0 + f1_score(Y_te, Y_p, average='binary', pos_label=0)
##        f1measure1_1 = f1measure1_1 + f1_score(Y_te, Y_p, average='binary', pos_label=1)
##        print(f1_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(f1_score(Y_te, Y_p, average='binary', pos_label=1));
##        print(confusion_matrix(Y_te, Y_p));
        sv = svm.SVC(kernel='linear', random_state = 2543484765, degree=3, cache_size = 40, class_weight = None)
        sv.fit(X_tr, Y_tr)
        predicted_digits = sv.predict(X_te)
        Y_p = predicted_digits
        accuracy2 = accuracy2+accuracy_score(Y_te, Y_p);
        print(accuracy_score(Y_te, Y_p));
        precision2_0 = precision2_0 + precision_score(Y_te, Y_p, average='binary', pos_label=0);
        precision2_1 = precision2_1 + precision_score(Y_te, Y_p, average='binary', pos_label=1);
        recall2_0 = recall2_0 + recall_score(Y_te, Y_p, average='binary', pos_label=0);
        recall2_1 = recall2_1 + recall_score(Y_te, Y_p, average='binary', pos_label=1);       
        print(precision_score(Y_te, Y_p, average='binary', pos_label=1));
        print(precision_score(Y_te, Y_p, average='binary', pos_label=0));
        print(recall_score(Y_te, Y_p, average='binary', pos_label=0));
        print(recall_score(Y_te, Y_p, average='binary', pos_label=1));
        f1measure2_0 = f1measure2_0 + f1_score(Y_te, Y_p, average='binary', pos_label=0)
        f1measure2_1 = f1measure2_1 + f1_score(Y_te, Y_p, average='binary', pos_label=1)
        print(f1_score(Y_te, Y_p, average='binary', pos_label=0));
        print(f1_score(Y_te, Y_p, average='binary', pos_label=1));
        print(confusion_matrix(Y_te, Y_p));
##        logreg = linear_model.LogisticRegression(C=1e5)
##        logreg.fit(X_tr, Y_tr)
##        predicted_digits = logreg.predict(X_te)
##        Y_p = predicted_digits
##        accuracy3= accuracy3+accuracy_score(Y_te, Y_p);
##        print(accuracy_score(Y_te, Y_p));
##        precision3_0 = precision3_0 + precision_score(Y_te, Y_p, average='binary', pos_label=0);
##        precision3_1 = precision3_1 + precision_score(Y_te, Y_p, average='binary', pos_label=1);
##        recall3_0 = recall3_0 + recall_score(Y_te, Y_p, average='binary', pos_label=0);
##        recall3_1 = recall3_1 + recall_score(Y_te, Y_p, average='binary', pos_label=1);       
##        print(precision_score(Y_te, Y_p, average='binary', pos_label=1));
##        print(precision_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(recall_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(recall_score(Y_te, Y_p, average='binary', pos_label=1));
##        f1measure3_0 = f1measure3_0 + f1_score(Y_te, Y_p, average='binary', pos_label=0)
##        f1measure3_1 = f1measure3_1 + f1_score(Y_te, Y_p, average='binary', pos_label=1)
##        print(f1_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(f1_score(Y_te, Y_p, average='binary', pos_label=1));
##        print(confusion_matrix(Y_te, Y_p));
##        neigh = KNeighborsClassifier(n_neighbors=3)
##        neigh.fit(X_tr, Y_tr)
##        predicted_digits = neigh.predict(X_te)
##        Y_p = predicted_digits
##        accuracy4 = accuracy4+accuracy_score(Y_te, Y_p);
##        print(accuracy_score(Y_te, Y_p));
##        precision4_0 = precision4_0 + precision_score(Y_te, Y_p, average='binary', pos_label=0);
##        precision4_1 = precision4_1 + precision_score(Y_te, Y_p, average='binary', pos_label=1);
##        recall4_0 = recall4_0 + recall_score(Y_te, Y_p, average='binary', pos_label=0);
##        recall4_1 = recall4_1 + recall_score(Y_te, Y_p, average='binary', pos_label=1);       
##        print(precision_score(Y_te, Y_p, average='binary', pos_label=1));
##        print(precision_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(recall_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(recall_score(Y_te, Y_p, average='binary', pos_label=1));
##        f1measure4_0 = f1measure4_0 + f1_score(Y_te, Y_p, average='binary', pos_label=0)
##        f1measure4_1 = f1measure4_1 + f1_score(Y_te, Y_p, average='binary', pos_label=1)
##        print(f1_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(f1_score(Y_te, Y_p, average='binary', pos_label=1));
##        print(confusion_matrix(Y_te, Y_p));
        
    print("accuracy : ",accuracy1/K);
    print("precision: " ,precision1_0/K);
    print("recall:",recall1_0/K);
    print("f1 measure:",f1measure1_0/K);
    print("precision: " ,precision1_1/K);
    print("recall:",recall1_1/K);
    print("f1 measure:",f1measure1_1/K);
    print("accuracy : ",accuracy2/K);
    print("precision: " ,precision2_0/K);
    print("recall:",recall2_0/K);
    print("f1 measure:",f1measure2_0/K);
    print("precision: " ,precision2_1/K);
    print("recall:",recall2_1/K);
    print("f1 measure:",f1measure2_1/K);
    print("accuracy : ",accuracy3/K);
    print("precision: " ,precision3_0/K);
    print("recall:",recall3_0/K);
    print("f1 measure:",f1measure3_0/K);
    print("precision: " ,precision3_1/K);
    print("recall:",recall3_1/K);
    print("f1 measure:",f1measure3_1/K);
    print("accuracy : ",accuracy4/K);
    print("precision: " ,precision4_0/K);
    print("recall:",recall4_0/K);
    print("f1 measure:",f1measure4_0/K);
    print("precision: " ,precision4_1/K);
    print("recall:",recall4_1/K);
    print("f1 measure:",f1measure4_1/K);






if __name__=="__main__":
    main()
