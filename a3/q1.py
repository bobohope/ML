'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # tf-idf representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def multinominal_nb(tf_idf_train, train_labels, tf_idf_test, test_labels):
    model = MultinomialNB(alpha=.01)
    model.fit(tf_idf_train, train_labels)
    #evaluate the model
    train_pred = model.predict(tf_idf_train)
    print('Multinomial Naive Bayes train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(tf_idf_test)
    print('Multinomial Naive Bayes test accuracy = {}'.format((test_pred == test_labels).mean()))
    return model
def LinSVC(bow_train,train_labels,bow_test,test_labels):
    model = LinearSVC(multi_class='ovr')
    model.fit(bow_train,train_labels)
    #evaluate the model
    train_pred = model.predict(bow_train)
    print('LinearSVC train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(bow_test)
    print('LinearSVC test accuracy = {}'.format((test_pred == test_labels).mean()))
    return model

# def knn(tf_idf_train, train_labels, tf_idf_test, test_labels):
#     model = KNeighborsClassifier(n_neighbors=15)
#     model.fit(tf_idf_train, train_labels)
#     train_pred = model.predict(tf_idf_train)
#     print('KNN train accuracy = {}'.format((train_pred == train_labels).mean()))
#     test_pred = model.predict(tf_idf_test)
#     print('KNN test accuracy = {}'.format((test_pred == test_labels).mean()))
#     return model
# def gnb(bow_train,train_labels,bow_test,test_labels):
#     model = GaussianNB()
#     model.fit(bow_train,train_labels)
#     train_pred = model.predict(bow_train)
#     print('Guassian NB train accuracy = {}'.format((train_pred == train_labels).mean()))
#     test_pred = model.predict(bow_test)
#     print('Guassian NB test accuracy = {}'.format((test_pred == test_labels).mean()))
#
#     return model

def rand_forest(tf_idf_train, train_labels, tf_idf_test, test_labels):
    from sklearn.ensemble import RandomForestClassifier
    model=RandomForestClassifier()
    model.fit(tf_idf_train, train_labels)
    #evaluate the model
    train_pred = model.predict(tf_idf_train)
    print('RF train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(tf_idf_test)
    print('RF test accuracy = {}'.format((test_pred == test_labels).mean()))
    return model
def sgd(tf_idf_train, train_labels, tf_idf_test, test_labels):
    from sklearn.linear_model import SGDClassifier
    model=SGDClassifier(max_iter=1000,tol=1e-4)
    model.fit(tf_idf_train, train_labels)
    #evaluate the model
    train_pred = model.predict(tf_idf_train)
    print('SGD train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(tf_idf_test)
    print('SGD test accuracy = {}'.format((test_pred == test_labels).mean()))
    return model
def experiment_with_data(data):
    #this function is created just to test the data set...
    from pprint import pprint
    print(np.shape(data.filenames))
    print(np.shape(data.target))
    #pprint(list(data.target_names))

def get_confusion_matrix(test_data,test_labels,model,target_names):
    print('The confusion matrix is')
    test_pred = model.predict(test_data)
    C=metrics.confusion_matrix(test_labels,test_pred)
    #print(C)
    np.fill_diagonal(C, 0)
    index=np.unravel_index(C.argmax(), C.shape)
    print(index)
    print('the group ',target_names[index[0]],' AND the group ',
    target_names[index[1]],' is the most confused.')
if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)
    tf_idf_train, tf_idf_test, feature_names = tf_idf_features(train_data, test_data)
    #bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)
    mnnb_model=multinominal_nb(tf_idf_train, train_data.target, tf_idf_test, test_data.target)
    #gnb_model=gnb(train_bow.toarray(), train_data.target, test_bow.toarray(), test_data.target)
    #LinSVC_model=LinSVC(train_bow, train_data.target, test_bow, test_data.target)
    #knn_model=knn(tf_idf_train, train_data.target, tf_idf_test, test_data.target)
    #rf_model=rand_forest(tf_idf_train, train_data.target, tf_idf_test, test_data.target)
    #sgd_model=sgd(tf_idf_train, train_data.target, tf_idf_test, test_data.target)
    get_confusion_matrix(tf_idf_test,test_data.target,mnnb_model,test_data.target_names)
    #experiment_with_data(train_data)
    #print('---------------------')
    #experiment_with_data(test_data)
