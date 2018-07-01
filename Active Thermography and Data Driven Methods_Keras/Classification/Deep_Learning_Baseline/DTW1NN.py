from sklearn.neighbors import KNeighborsClassifier
from util import *

fname = 'dataset'
neigh = KNeighborsClassifier(n_neighbors=1, metric=DTWDist)

start_time = time.time()

x_train, y_train = readucr(fname+'/'+fname+'_train.csv')
x_test, y_test = readucr(fname+'/'+fname+'_test.csv')

print 'Training..'
y_pred = neigh.fit(x_train, y_train).predict(x_test)
print '\n\n\n\nDone'

save_pickle(neigh, 'dtw1nn.pkl')

print "Accuracy: %.5E" % accuracy_score(y_test, y_pred)
print "Time elapsed %.2Es" % ((time.time()-start_time) / 1000.)