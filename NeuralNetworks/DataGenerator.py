import numpy as np
import random
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
import datetime as dt

def to_categorical(y):
    n = len(y)
    y = y.astype(np.int32)
    cat = np.zeros((n,10))

    for i in range(n):
        cat[i, y[i]] = 1

    return cat


def get_pca_data():
    try:
        return np.load("pca_minks_xtrain.npy"), np.load("pca_minks_xtest.npy"), np.load("pca_minks_ytrain.npy"), np.load("pca_minks_ytest.npy")
    except:
        t0 = dt.datetime.now()
        print("starting pca")
        df = pd.read_csv("train.csv")

        data = df.values.astype(np.float32)
        np.random.shuffle(data)

        X = data[:, 1:]
        Y = data[:, 0].astype(np.int32)

        Xtrain = X[:-1000]
        Ytrain = Y[:-1000]
        Xtest = X[-1000:]
        Ytest = Y[-1000:]
        # center the data
        mu = Xtrain.mean(axis=0)
        Xtrain = Xtrain - mu
        Xtest = Xtest - mu

        # transform the data
        pca = TruncatedSVD(algorithm='arpack', n_components=300)
        #pca = PCA(n_components=100)
        Ztrain = pca.fit_transform(Xtrain)
        Ztest = pca.transform(Xtest)

        # take first 300 cols of Z
        Ztrain = Ztrain[:, :300]
        Ztest = Ztest[:, :300]

        # normalize Z
        mu = Ztrain.mean(axis=0)
        std = Ztrain.std(axis=0)
        Ztrain = (Ztrain - mu) / std
        Ztest = (Ztest - mu) / std


        print("pca completed in {}".format(dt.datetime.now() - t0))

        print("\n\n\n")

        np.save("pca_minks_xtrain.npy", Ztrain)
        np.save("pca_minks_ytrain.npy", Ytrain)
        np.save("pca_minks_xtest.npy", Ztest)
        np.save("pca_minks_ytest.npy", Ytest)

        return Ztrain, Ztest, Ytrain, Ytest

def get_normalised_data():
    df = pd.read_csv('train.csv')
    data = df.values.astype(np.float32)
    np.random.shuffle(data)
    X = data[:, 1:]
    Y = data[:, 0]

    Xtrain = X[:-1000]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:]
    Ytest = Y[-1000:]

    # normalize the data
    mu = Xtrain.mean(axis=0)
    std = Xtrain.std(axis=0)
    np.place(std, std == 0, 1)
    Xtrain = (Xtrain - mu) / std
    Xtest = (Xtest - mu) / std

    return Xtrain, Xtest, Ytrain, Ytest


def get_minst_data(pca=False):
    if pca:
        x_train, x_test, y_train, y_test = get_pca_data()
    else:
        x_train, x_test, y_train, y_test = get_normalised_data()

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, x_test, y_train, y_test

