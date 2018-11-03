'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt


def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    #print (train_data[4])
    for k in range(0,10):
        classk_data = train_data[np.where(train_labels==k)]
        means[k]=np.mean(classk_data,axis=0)
    #print(means)
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    # first get the mean
    means = compute_mean_mles(train_data, train_labels)
    for k in range(0,10):
        classk_data = train_data[np.where(train_labels==k)]
        x_minus_mean=classk_data-means[k]
        temp=0
        #print (x_minus_mean)
        #print (np.shape(classk_data))
        for i in range(np.shape(classk_data)[0]):

            temp+=np.outer(x_minus_mean[i],x_minus_mean[i])
            #print(np.outer(x_minus_mean[i],x_minus_mean[i]))
        covariances[k]=temp/np.shape(classk_data)[0]
    #print(covariances)
    #print(np.shape(covariances))
    return covariances

def plot_cov_diagonal(covariances):
    # Plot the diagonal of each covariance matrix side by side
    cov_diag_logs=[]

    for i in range(10):
        cov_diag = np.diag(covariances[i])
        #cov_diag_log=np.log(cov_diag)
        cov_diag_logs.append(np.log(cov_diag).reshape(8,8))
    all_concat = np.concatenate(cov_diag_logs,axis=1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()


def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    # digits is a matrix of dimension nx64
    d=np.shape(digits)[1]
    n=np.shape(digits)[0]
    LLmat=np.zeros((n,10))
    #print(covariances)
    covariances_prime=covariances+0.01*np.identity(d)
    #print (covariances)
    for i in range(n):
        for k in range(0,10):
            cov_det=np.linalg.det(covariances_prime[k])
            cov_inverse=np.linalg.solve(covariances_prime[k],np.identity(d))
            diff=digits[i]-means[k]
            diffa=np.array([diff])
            #print(np.shape(np.array([diff]).T))
            LL=-d/2*np.log(2*np.pi)-(1/2)*np.log(cov_det)-(1/2)*diff.T.dot(cov_inverse).dot(diff)
            #print(np.exp(LL))
            #print(LL)
            LLmat[i,k]=LL
    #print (LLmat)
    return LLmat

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    LLmat=generative_likelihood(digits,means,covariances)
    Pmat=np.exp(LLmat)
    #print(Pmat)
    #print(np.shape(np.mean(Pmat,axis=1)))
    cPmat=Pmat*0.1/(0.1*np.sum(Pmat,axis=1)[:,None])
    cLLmat=np.log(cPmat)
    #print (cLLmat)
    #print (np.exp(cLLmat))
    #print (np.exp(cLLmat)[200])
    return cLLmat

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    temp=0
    n=np.shape(digits)[0]
    for i in range(n):
        temp+=cond_likelihood[i,int(labels[i])]
        #print (temp)
    average=temp/n
    print(average)
    #print (np.exp(average))
    return average

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    predicted_label=np.argmax(cond_likelihood,axis=1)
    return predicted_label
def calculate_accuracy(digits, labels, means, covariances):
    predicted_label=classify_data(digits,means,covariances)
    temp=np.sum(predicted_label==labels)
    return temp/np.shape(digits)[0]
def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    #plot_cov_diagonal(covariances)
    #generative_likelihood(test_data, means, covariances)
    #conditional_likelihood(test_data, means, covariances)
    #avg_conditional_likelihood(test_data, test_labels, means, covariances)
    #avg_conditional_likelihood(train_data, train_labels, means, covariances)
    predicted_label=classify_data(test_data,means,covariances)
    print (predicted_label)
    print (test_labels)
    accuracy=calculate_accuracy(test_data,test_labels,means,covariances)
    print (accuracy)
    # Evaluation

if __name__ == '__main__':
    main()
