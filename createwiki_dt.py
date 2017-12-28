import pandas as pd
import random
from sklearn.model_selection import StratifiedKFold
K=5
cv = StratifiedKFold(n_splits=K,shuffle=True, random_state=22)
df = pd.read_csv('wiki_all.csv',delimiter='\t')
df['pref'] = 0
df['pref'] = df['prevOutdegree']*df['currIndegree']
y = df[df['status']==0]
n = df[df['status']==1]


for i in range(0,10):
    c = random.sample(list(n.index),len(y))
    newn = n.ix[c]
    frames = [y,newn]
    pf = pd.concat(frames)
    pf.to_csv('wiki'+str(i)+'.csv',sep='\t',index=None)
    X = pf.values
    Y = pf['status'].values
    j = 0
    for train, test in cv.split(X,Y):
        f_train = pf.iloc[train]
        f_test = pf.iloc[test]
        f_train.to_csv('wiki'+str(i)+'train'+str(j)+'.csv',sep='\t',index=None)
        f_test.to_csv('wiki'+str(i)+'test'+str(j)+'.csv',sep='\t',index=None)
        j = j+1
