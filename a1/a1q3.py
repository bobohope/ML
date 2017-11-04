import numpy as np
from sklearn.datasets import load_boston

BATCHES = 50

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


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    #w = np.random.randn(features)
    w = np.random.normal(0,1,features)
    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)

#TODO: implement linear regression gradient
def lin_reg_gradient(X, y, w, m):
    '''
    Compute gradient of linear regression model parameterized by w
    '''
    a=[]
    for i in range (m):
        a.append(2*np.dot(X[i].T,X[i])*w.T-2*y[i]*X[i])
    #print (a)
    return np.ones((1,m)).dot(a)/m

def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)
    batch_grad=[]
    avg_grad=[]

    variant=[]
    # Example usage
    for k in range (500):
        X_b, y_b = batch_sampler.get_batch()
        a=lin_reg_gradient(X_b, y_b, w,BATCHES)
        #print(a[0].shape)
        batch_grad.append(a[0])
    avg_grad= np.ones((1,500)).dot(batch_grad)/500
    print(avg_grad)
    #calculate true grad
    true_grad=lin_reg_gradient(X,y,w,X.shape[0])
    print(true_grad)
    #calcualte distant and cos similarity
    print(cosine_similarity(avg_grad[0],true_grad[0]))
    print(np.linalg.norm(avg_grad[0]-true_grad[0])**2,np.linalg.norm(avg_grad[0]),np.linalg.norm(true_grad[0]))
    print(np.sqrt(np.sum((avg_grad[0]-true_grad[0])**2)))
#     for m in range(400):
#         a=[]
#         for k in range (500):
#             batch_sampler_m =BatchSampler(X,y,m+1)
#             X_b, y_b = batch_sampler_m.get_batch()
#             #print(w)
#             #print(X_b,y_b)
#             #print(lin_reg_gradient(X_b, y_b, w,m))
#             a.append(lin_reg_gradient(X_b, y_b, w,m+1 )[0,2])
#         va=np.var(a)
#         print(va)
#         variant.append(va)
#     print(variant)
#     return variant


if __name__ == '__main__':
    main()
