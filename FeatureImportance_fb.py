print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

import pandas as pd


### Build a classification task using 3 informative features
##X, y = make_classification(n_samples=1000,
##                           n_features=10,
##                           n_informative=3,
##                           n_redundant=0,
##                           n_repeated=0,
##                           n_classes=2,
##                           random_state=0,
##                           shuffle=False)





# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)
df = pd.read_csv('fb_final.csv',delimiter='\t')
del df['status']
del df['time']
del df['user1']
del df['user2']
y = df['status_class'].values
del df['status_class']
X = df.values
c = list(df.columns.values)
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
np.savetxt('importances.csv',importances,delimiter='\t')
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
##plt.figure()
##plt.title("Feature importances")
##plt.bar(range(10), importances[indices[:10]],
##       color="r", yerr=std[indices[:10]], align="center")
##plt.xticks(range(10), c[:10])
##plt.xlim([-1, 10])
##plt.savefig('wiki_importances.pdf')

