from sklearn import svm
from util import *

fname = 'easy' # 'dataset' # 'difficult' #

svc = svm.SVC(kernel='linear')

start_time = time.time()

x_train, y_train = readucr(fname+'/'+fname+'_train.csv')
x_test, y_test = readucr(fname+'/'+fname+'_test.csv')

print 'Training..'
y_pred = svc.fit(x_train, y_train).predict(x_test)
print 'Done'

save_pickle(svc, 'svm_easy.pkl')

print "Accuracy: %.5E" % accuracy_score(y_test, y_pred)
print "Time elapsed %.2Es" % ((time.time()-start_time) / 1000.)



# for duration in np.linspace(10.5, 15, 10):
#     print "Duration: ", duration

#     print 'Training..'
#     y_pred = svc.fit(x_train[:,:int(x_train.shape[1]*duration/MAX_TIME)], y_train).predict(x_test[:,:int(x_train.shape[1]*duration/MAX_TIME)])
#     print 'Done'

#     # save_pickle(svc, 'svm_difficult.pkl')

#     print "Accuracy: %.5E" % accuracy_score(y_test, y_pred)
#     print "Time elapsed %.2Es" % ((time.time()-start_time) / 1000.)


