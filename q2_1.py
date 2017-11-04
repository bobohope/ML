'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point

        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        #print(np.shape(self.train_data))
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        #print(np.shape(train_norm))
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        distvec=self.l2_distance(test_point)
        #print(distvec)
        #print(np.shape(distvec))
        # get the index that will sort distvec in accending order
        index = np.argsort(distvec)
        # get the labels of k points that are closest to the test point
        #print(distvec)
        while True:
            labels = self.train_labels[index[0:k]]
            #print(labels)
            unique_label, count=np.unique(labels,return_counts=True)
            #check if tide
            sorted_count=np.sort(count)
            if  len(sorted_count)==1 or sorted_count[-1]>sorted_count[-2]:
                break
            k-=1
        #return prediction
        digit = unique_label[np.argmax(count)]
        #print (digit)
        return digit

def cross_validation(knn, k_range=np.arange(1,16)):
    from sklearn.model_selection import KFold
    kf=KFold(n_splits=10, shuffle=True)
    accuracy=np.zeros(len(k_range))
    for k in k_range:
        # Loop over folds
        print (k)
        good=0
        for train_index, test_index in kf.split(knn.train_data):
            X_train, X_test = knn.train_data[train_index],knn.train_data[test_index]
            y_train, y_test = knn.train_labels[train_index],knn.train_labels[test_index]
            #create new knn object
            newknn=KNearestNeighbor(X_train, y_train)
            for i in range(len(X_test)):
                predicted_label=newknn.query_knn(X_test[i],k)
                if predicted_label==y_test[i]:
                    good+=1
            #print ('for one fold:', good)
            #print ('dimension of x_test', np.shape(X_test))
        print(good)
        accuracy[k-1]=good/np.shape(knn.train_data)[0]
        # Evaluate k-NN
        # ...
    #return accuracy accross folds for different k valuse
    return accuracy
def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    pass

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    #print(np.shape(train_data))
    knn = KNearestNeighbor(train_data, train_labels)

    # Example usage:
    #predicted_label = knn.query_knn(test_data[10], 2)
    #print(predicted_label)
    print(cross_validation(knn, k_range=np.arange(1,16)))
if __name__ == '__main__':
    main()
