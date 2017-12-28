import pandas as pd
from sklearn.model_selection import StratifiedKFold
K=5
skf = StratifiedKFold(n_splits=K)
for i in range(0,5):
    df = pd.read_csv('wiki_'+str(i)+'.csv',delimiter='\t')
    X = df.copy()
    y = df['status']
    j = 0
    for train_index, test_index in skf.split(X, y):
            train = X.ix[train_index]
            test = X.ix[test_index]
            train.to_csv('fb'+str(i)+'train'+str(j)+'.csv',sep='\t',index=None)
            test.to_csv('fb'+str(i)+'test'+str(j)+'.csv',sep='\t',index=None)
            j=j+1