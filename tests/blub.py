from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score, ShuffleSplit
import numpy as np
s = []
for i in xrange(10):
    data = np.loadtxt('sonar2.csv', delimiter=',')
    clf = AdaBoostClassifier(n_estimators=100)
    cv = ShuffleSplit(data.shape[0], 10)
    #clf = DecisionTreeClassifier(max_depth=1)

    scores = cross_val_score(clf, data[:,1:], data[:,0], cv=cv)
    s.append(1-scores.mean())
print np.mean(s)
