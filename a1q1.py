from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X,y,features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]
    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        #TODO: Plot feature i against y
        plt.scatter(X[:,i],y)
        plt.tight_layout()
    plt.show()


def fit_regression(X,y):
    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    #raise NotImplementedError()
    return lin_reg


def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))

    # Visualize the features
    visualize(X, y, features)

    #TODO: Split data into train and test
    trainidx=np.random.choice(506, 405, replace=False)
    testidx=[]
    for i in range (506):
        if i not in trainidx:
            testidx.append(i)
    Xtrain=X[trainidx,:]
    ytrain=y[trainidx]
    Xtest=X[testidx,:]
    ytest=y[testidx]
    # Fit regression model
    lin_reg = fit_regression(Xtrain, ytrain)
    ypredict=lin_reg.predict(Xtest)
    print (ypredict)
    plt.figure
    plt.scatter(ypredict,ytest)
    plt.show()

    # Compute fitted values, MSE, etc.
    print ('MES is ',mean_squared_error(ytest, ypredict))

if __name__ == "__main__":
    main()
