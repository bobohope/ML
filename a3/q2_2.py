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

    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # get num of data points
        n = np.shape(X)[0]
        # Implement hinge loss
        return np.squeeze(np.maximum(np.zeros(n),1-y*(np.array([self.w]).dot(X.T))))

    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        self.w[-1]=0
        n = np.shape(X)[0]
        # get rid of the gradient that the hinge_loss is less than zero
        y=np.logical_and(y,self.hinge_loss(X,y))*y
        gradient = self.w - self.c/n*(X.T.dot(y.T)).T

        #print('dimension of grad is ', np.shape(gradient))
        return np.squeeze(gradient)

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        return np.squeeze(np.sign(np.array([self.w]).dot(X.T)))

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
    #initialize the model and other variables
    feature_count=np.shape(train_data)[1]
    svm_model=SVM(penalty,feature_count)
    sampler = BatchSampler (train_data,train_targets,batchsize)
    mom=np.zeros(feature_count)
    w=np.zeros(feature_count)
    gradnorm = []
    #train the svm model
    for i in range(iters):
        # get batch
        X_batch, y_batch =sampler.get_batch()
        w, mom = optimizer.update_params(w, mom, svm_model.grad(X_batch,y_batch))
        #gradnorm.append(np.linalg.norm(svm_model.grad(train_data,train_targets)))
        svm_model.w = w
        #print(w)
    #print(gradnorm)
    return svm_model



if __name__ == '__main__':
    # # PART 1
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
    # num of data points
    n_train = np.shape(train_data)[0]
    n_test = np.shape(test_data)[0]
    # add constant one feature - no bias needed
    # x_0 = 1
    train_data = np.concatenate((np.ones((n_train,1)),train_data),axis=1)
    test_data = np.concatenate((np.ones((n_test,1)), test_data),axis=1)
    #--------------------------------
    # train the SVM model
    feature_count=np.shape(train_data)[1]
    #svm_model=SVM(1,feature_count)
    gdopt_model=GDOptimizer(lr=0.05,beta=0)
    gdopt_model2=GDOptimizer(lr=0.05,beta=0.1)
    svm_model=optimize_svm(train_data, train_targets, 1, gdopt_model, 100, 500)

    #---------------------------------------------------
    #evaluate the model
    #print(svm_model.hinge_loss(train_data,train_targets))
    #print(svm_model.hinge_loss(test_data,test_targets))
    #print(svm_model.grad(train_data,train_targets))
    print('Train loss = {}'.format(0.5*np.linalg.norm(svm_model.w)+np.mean(svm_model.hinge_loss(train_data,train_targets))))
    print('Train loss = {}'.format(0.5*np.linalg.norm(svm_model.w)+np.mean(svm_model.hinge_loss(test_data,test_targets))))

    train_pred = svm_model.classify(train_data)
    #print(train_pred)
    #print(train_targets)
    print('Train accuracy = {}'.format((train_pred == train_targets).mean()))
    test_pred = svm_model.classify(test_data)
    print('Test accuracy = {}'.format((test_pred == test_targets).mean()))
    #print(np.shape(svm_model.w))
    img=svm_model.w[1:].reshape(28,28)
    plt.imshow(img, cmap='gray')
    plt.show()
