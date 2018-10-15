# Bibliotecas pandas e numpy para manipulação de dados
import numpy as np

# Módulos do SciKit-Learn para modelagem
import sklearn.metrics as metrics
from sklearn.externals import joblib

def training_test_score( clf, X_train, y_train, X_test, y_test, metrics, **kwargs  ):
    clf.fit( X_train, y_train)
    predict = {}
    predict["y_score"] = clf.predict_proba( X_test )[:,-1]
    predict["y_pred"]  = clf.predict( X_test )
    performance = {"name":clf.__name__}
    for m in metrics:
        try:
            score = m( y_test, predict[ m.__code__.co_varnames[1] ] )
            performance[m.__name__] = score
        except ValueError as err:
            pass
    return performance
