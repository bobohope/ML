'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    #binarize_data(train_data)
    eta = np.zeros((10, 64))
    for k in range (0,10):
        #print (train_labels)
        #print (np.count_nonzero(train_labels==k))
        eta[k]=(1+np.sum(train_data[np.where(train_labels==k)],axis=0))/(2+np.count_nonzero(train_labels==k))
    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    imgs=[]
    for i in range(10):
        img_i = class_images[i]
        imgs.append(img_i.reshape((8,8)))
    # Plot all means on same axis
    all_concat = np.concatenate(imgs,axis=1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))
    pr=np.zeros((10, 64))
    for  k in range(10):
        for i in range(64):
            pr[k,i]=np.random.binomial(1,eta[k,i])

    #print(pr)
    generated_data=pr
    plot_images(generated_data)

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array
    '''
    n=np.shape(bin_digits)[0]
    d=np.shape(bin_digits)[1]
    likelihood=np.zeros((n,10))
    for k in range(10):
        for i in range(n):
            for j in range(d):
                #print((eta[k,j]**bin_digits[i,j])*((1-eta[k,j])**(1-bin_digits[i,j])))
                likelihood[i,k]+=bin_digits[i,j]*np.log(eta[k,j])+(1-bin_digits[i,j])*np.log(1-eta[k,j])
    #print(likelihood)
    return likelihood

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    LLmat=generative_likelihood(bin_digits,eta)
    Pmat=np.exp(LLmat)
    #print(Pmat)
    #print(np.shape(np.mean(Pmat,axis=1)))
    cPmat=Pmat*0.1/(0.1*np.sum(Pmat,axis=1)[:,None])
    cLLmat=np.log(cPmat)
    #print (cLLmat)
    #print (np.exp(cLLmat))
    #print (np.exp(cLLmat)[200])
    return cLLmat

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    temp=0
    n=np.shape(bin_digits)[0]
    for i in range(n):
        temp+=cond_likelihood[i,int(labels[i])]
        #print (temp)
    average=temp/n
    print(average)
    #print (np.exp(average))
    return average

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute and return the most likely class
    predicted_label=np.argmax(cond_likelihood,axis=1)
    return predicted_label
def calculate_accuracy(digits, labels, eta):
    predicted_label=classify_data(digits,eta)
    temp=np.sum(predicted_label==labels)
    return temp/np.shape(digits)[0]

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)
    #print(train_labels)
    # Fit the model
    eta = compute_parameters(train_data, train_labels)
    #print(eta)
    # Evaluation
    #plot_images(eta)

    #generate_new_data(eta)
    #generative_likelihood(test_data,eta)
    avg_conditional_likelihood(test_data, test_labels,eta)
    avg_conditional_likelihood(train_data, train_labels,eta)
    accuracy=calculate_accuracy(test_data,test_labels,eta)
    print (accuracy)
if __name__ == '__main__':
    main()
