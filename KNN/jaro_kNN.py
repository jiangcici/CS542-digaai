import csv
import numpy as np
import operator
from math import ceil
from tensorflow.python.platform import flags
from pyjarowinkler import distance
import sklearn.model_selection as cross_validation
import pickle
from time import time

flags.DEFINE_string("tr", 'train.csv', "Training data")
flags.DEFINE_string("te", 'test.csv', "Testing data")
flags.DEFINE_integer("cv_mode", 0, "0-not using cross validation; 1-use cross validation")
flags.DEFINE_string("metric_mode", 'wt', "'dist'-using original distance; 'wt'-using weighted kNN")
flags.DEFINE_float("Gau_h", 0.6, "0-1: value of h in Gaussian Kernel (only used when using weighted kNN)")
flags.DEFINE_integer("kNN_k", 50, "value of k")# valid when non-cross validation mode

FLAGS = flags.FLAGS
Tr = FLAGS.tr
Te = FLAGS.te
CV_mode = FLAGS.cv_mode
Metric_mode = FLAGS.metric_mode
k = FLAGS.kNN_k
h = FLAGS.Gau_h

def compute_time(time_start,time_end):
    minutes, seconds = divmod(time_end - time_start, 60)
    time_total = {'minute': minutes, 'second': seconds}
    return time_total

def Gaussian(dist, h):
    '''Gaussian Weighting function'''

    weight = 1.5 * np.exp(-(1-dist)*(1-dist)/(h*h))
    return weight

def voting(sorted_, sorted_idx, labels, k):
    '''Vote in the nearest k'''

    if Metric_mode == 'dist':
        classcount = {}
        for i in range(k):
            votelabel = labels[sorted_idx[i]] #sorted labels
            
            classcount[votelabel] = classcount.get(votelabel,0)+1 #count the number of each label
            
            maxcount = 0
        for key, value in classcount.items():  #find the class that has the most in the k elements
            if value > maxcount:
                maxcount = value
                maxindex = key    
    else:
        sum_wt_0 = 0  # sum of weight for class 0
        sum_wt_1 = 0  # sum of weight for class 1
        
        for i in range(k):
            if labels[sorted_idx[i]] == '0':
                sum_wt_0 = sum_wt_0 + sorted_[i]
            else:
                sum_wt_1 = sum_wt_1 + sorted_[i]
        
        if sum_wt_0>=sum_wt_1:
            maxindex=0
        else:
            maxindex=1
        
    return maxindex

def knn(inputs, dataset, labels, k): 
    '''Main function for doing kNN''' 

    numsamples = len(dataset)
    Distance = [] # a list of distances
    Weight = []
    for i in range(numsamples):
        dist = distance.get_jaro_distance(inputs, dataset[i], winkler=True, scaling=0.1)
        Distance.append(dist)
        if Metric_mode=='wt':
            wt = Gaussian(dist,h)
            Weight.append(wt)
    
    if Metric_mode=='dist':
        sorted_ = -1*np.sort(-1*np.array(Distance))
        sorted_idx = np.argsort(np.array(Distance))
        sorted_idx = sorted_idx[::-1]
    else:
        sorted_ = -1*np.sort(-1*np.array(Weight))
        sorted_idx = np.argsort(np.array(Weight))
        sorted_idx = sorted_idx[::-1]
    
    if CV_mode == 0:
        result = voting(sorted_, sorted_idx, labels, k)
    else:
        result = []
        for i in range(len(k)):
            maxindex = voting(sorted_, sorted_idx, labels, k[i])
            result.append(maxindex)
        result = np.array(result).reshape(1,-1)
    
    return result


def main():
    #read csv to a dictionary
    print("-------Loading Data--------")
    csvFile = open(Tr,'r') #open file
    reader = csv.reader(csvFile) #reader is still a .csv file!!
    
    #build an empty dict
    first = []
    last = []
    label = []
    for item in reader: #omit the first line because it's the header
        if reader.line_num == 1:
            continue
    #    fn = item[1]
    #    fn.replace(fn[fn.find(r'\x'):fn.find(r'\x')+3],'')
    #    print(fn)
    #    print(fn[fn.find(r'\x'):fn.find(r'x')+3])
        first.append(item[1])
        last.append(item[2])
        label.append(item[3])
    
    csvFile.close()
    X_train = first
    y_train = label
#    dataset = first
#    labels = label
    print("-------Finish Loading Training Data--------")
    
    testcsv = open(Te,'r') #open file
    testreader = csv.reader(testcsv) #reader is still a .csv file!!
    
    #build an empty dict
    t_first = []
    for item in testreader: #omit the first line because it's the header
        if testreader.line_num == 1:
            continue
        t_first.append(item[1])
    X_test = t_first # place to put test set
    
    print("-------Finish Loading Testing Data--------")
    
    print("-------Start KNN--------")
    if CV_mode == 0:
        time0=time()
        #WT = [] #weights
        result = [] # list of results
        for j in range(len(X_test)):
            test_y = knn(X_test[j], X_train, y_train, k)
            #WT.append(test_wt)
            result.append(test_y)
            print("{0}%".format(round((j + 1) * 100 / len(X_test))), end="\r") 
        time1=time()
        tt=compute_time(time0,time1)
        print ('Time for kNN: %(minute)d minute(s) %(second)d second(s)s' %  tt)
        
        print("Saving results...")
        fw = open('Testing results_k='+str(k)+'_h='+str(h)+'_'+Metric_mode+'.pkl', 'wb')
        pickle.dump(result, fw, protocol=2)
        fw.close()
        print("Finished!")
        
    else: # cross-validation mode
        K = list([30,50,100,150,200])
        five_score = []
        
        for i in range(5):
            score_k = []
            result = [] # list of results
            (X_tr, X_te, y_tr, y_te) = cross_validation.train_test_split(X_train, y_train, test_size=0.2)
            for j in range(len(X_te)):
                test_y = knn(X_te[j], X_tr, y_tr, K)# test_y is array (for each k)
                result.append(test_y)
            print('finished cv part'+str(i))
            result = np.array(result).squeeze()
            
            y_te = np.array(y_te).astype(int)
            for tt in range(len(K)):
                acc_list = result[:,tt] - y_te
                acc_list = list(acc_list)
                score_k.append(float(acc_list.count(0))/len(acc_list))
            five_score.append(np.array(score_k).reshape(1,-1))
            
        final_score = np.array(five_score).squeeze()
        
        for i in range(len(K)):
            print('results for k='+str(K[i])+'is %0.2f%% (+/- %0.2f%%)' % (final_score[:,i].mean()*100, final_score[:,i].std()*200))
            
            
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    