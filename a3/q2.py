import numpy as np

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''

    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch

class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum
    '''

    def __init__(self, lr=1, beta=0.0):
        #learning rate
        self.lr = lr
        #momentum
        self.beta = beta

    def update_params(self, params, mom, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        #params is the previous w, return the current w
        mom=-grad*self.lr+self.beta*mom
        params=params+mom
        return params, mom

class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)
        self.b = 0
        self.m = feature_count
    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss
        loss = 1-y*(np.array([self.w]).dot(X.T)+self.b)
        #print (1-y*(X.dot(np.array(self.w))))
        return np.where(loss > 0, loss, 0)

    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        y=np.logical_and(y,self.hinge_loss(X,y))*y
        #print (y)
        # Compute (sub-)gradient of SVM objective
        gradient_w = self.w - self.c/self.m*(X.T.dot(y.T)).T
        #print(np.shape(gradient_w))
        gradient_b = -self.c/self.m*np.sum(y)
        return np.append(gradient_w,gradient_b)

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1

        return None

def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets

def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''
    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x
    w = w_init
    w_history = [w_init]
    mom=0
    for i in range(steps):
        w,mom=optimizer.update_params(w,mom,func_grad(w))
        w_history.append(w)
    return w_history

def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.
    '''
    feature_count=np.shape(train_data)[1] #no feature for the bias, it is defined in SVM class !!!
    #create an svm_model
    svm_model=SVM(penalty,feature_count)
    w=np.zeros(feature_count+1)
    mom=np.zeros(feature_count+1)
    for i in range(iters):
        w, mom = optimizer.update_params(w , mom, svm_model.grad(train_data,train_targets))
    svm_model.w =
    return svm_model

if __name__ == '__main__':
    # PART 1
    # gdopt_model=GDOptimizer(lr=1,beta=0)
    # gdopt_model2=GDOptimizer(lr=1,beta=0.9)
    #
    # w_vector1=optimize_test_function(gdopt_model)
    # w_vector2=optimize_test_function(gdopt_model2)
    # #print (w_vector)
    # t = range(201)
    # plt.plot(t,w_vector1,'r--',t,w_vector2,'b-')
    # plt.show()


    # PART 2
    train_data, train_targets, test_data, test_targets=load_data()
    #print(train_targets)
    feature_count=np.shape(train_data)[1]
    svm_model=SVM(1,feature_count)
    #print(svm_model.grad(train_data,train_targets))
    print(np.shape(svm_model.grad(train_data,train_targets)))

    #print(svm_model.hinge_loss(train_data,train_targets))
