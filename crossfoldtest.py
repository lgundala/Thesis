import pandas as pd
import numpy as np
import scipy.stats
import math
import theanets
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score,matthews_corrcoef
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn import linear_model, svm, gaussian_process
import sklearn.metrics as met
import xgboost as xgb
import scipy.sparse as sp
import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from scipy.sparse import coo_matrix
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import *
from sklearn.metrics import accuracy_score
def main():
    dataset = pd.read_csv("English/allBaseFeatures.csv",delimiter=",", header=None, encoding="utf-8");
    del dataset[0]
    Y = dataset[10].values;
    del dataset[10]
    print(list(dataset.columns.values));
    x = dataset[[8]].values;
##    print(x.shape);
    X = x[:,0:5];
    print(X);
    accuracy = 0;
    precision1_0 = 0;
    precision1_1 = 0;
    recall1_0 = 0;
    recall1_1 = 0;
    f1measure1_0 = 0;
    f1measure1_1 = 0;
    K=10
    print(len(Y));
    cv = KFold(len(Y), n_folds=10,shuffle=True, random_state=22)
    for train, test in cv:
        X_tr=X[train]
        Y_tr=Y[train]
        X_te=X[test]
        Y_te=Y[test]
        rf = RandomForestClassifier(n_estimators=100, criterion="entropy",random_state=4294967294)
        rf.fit(X_tr, Y_tr)
        predicted_digits = rf.predict(X_te)
        Y_p = predicted_digits
        accuracy = accuracy+accuracy_score(Y_te, Y_p);
        print(accuracy_score(Y_te, Y_p));
        precision1_0 = precision1_0 + precision_score(Y_te, Y_p, average='binary', pos_label=0);
        precision1_1 = precision1_1 + precision_score(Y_te, Y_p, average='binary', pos_label=1);
        recall1_0 = recall1_0 + recall_score(Y_te, Y_p, average='binary', pos_label=0);
        recall1_1 = recall1_1 + recall_score(Y_te, Y_p, average='binary', pos_label=1);       
        print(precision_score(Y_te, Y_p, average='binary', pos_label=1));
        print(precision_score(Y_te, Y_p, average='binary', pos_label=0));
        print(recall_score(Y_te, Y_p, average='binary', pos_label=0));
        print(recall_score(Y_te, Y_p, average='binary', pos_label=1));
        f1measure1_0 = f1measure1_0 + f1_score(Y_te, Y_p, average='binary', pos_label=0)
        f1measure1_1 = f1measure1_1 + f1_score(Y_te, Y_p, average='binary', pos_label=1)
        print(f1_score(Y_te, Y_p, average='binary', pos_label=0));
        print(f1_score(Y_te, Y_p, average='binary', pos_label=1));
        print(confusion_matrix(Y_te, Y_p));
    print("accuracy : ",accuracy/K);
    print("precision: " ,precision1_0/K);
    print("precision: " ,precision1_1/K);
    print("recall:",recall1_0/K);
    print("recall:",recall1_1/K);
    print("f1 measure:",f1measure1_0/K);
    print("f1 measure:",f1measure1_1/K);






if __name__=="__main__":
    main()
