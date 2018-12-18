import numpy as np

RESULTS = open("/home/teo/RESULTS_TCC.txt", "a+")


def select_rows(X, rows):
    return X[rows,:]

def random_index(X, k):
    return np.random.choice(range(X.shape[0]), [k,int(X.shape[0]/k)], replace=False)

def cross_functions(X, Y, k, clf):
    index = random_index(X, k)
    acuracity = []
    for i in range(len(index)):
        to_remove = list(range(len(index)))
        to_remove.remove(i)

        X_test = X[index[i]]
        Y_test = Y[index[i]]

        X_train = X[index[to_remove].reshape(-1)]
        Y_train = Y[index[to_remove].reshape(-1)]
        
        clf.fit( X=X_train, Y=Y_train )
    
        Y_pred_train = clf.predict( X=X_train )
        Y_pred_test = clf.predict( X=X_test )
    
        acuracity_train = sum(Y_pred_train==Y_train)/Y_train.shape[0]
        acuracity_test = sum(Y_pred_test==Y_test)/Y_test.shape[0]

        acuracity.append([acuracity_train, acuracity_test, X_train.shape[0]])
    return np.array(acuracity)

def cross_validation(X,Y,clf, k, size_sample):
    train_scores = []
    test_scores=[]
    sizes = []
    for size in size_sample:
        index = np.random.choice(range(X.shape[0]), size, replace=False)
        result = cross_functions(X[index], Y[index], k, clf)
        train_scores.append(result[:,0])
        test_scores.append(result[:,1])
        sizes.append(size)
    return train_scores, test_scores, sizes

