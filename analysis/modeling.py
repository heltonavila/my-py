import pandas as pd
import numpy as np
from sklearn import tree

class tree_bins():

    def __init__(self, **kwargs):
        self.clf = tree.DecisionTreeClassifier(**kwargs)

    def fit(self, X, y):
        try:
            self.clf.fit( X, y )
        except ValueError as err:
            self.clf.fit( X.values.reshape(-1,1), y )

    def make_bins(self, X, name):
        df = pd.DataFrame()

        try:
            df[name] = self.clf.apply( X )
        except ValueError as err:
            df[name] = self.clf.apply( X.values.reshape(-1,1) )

        df_dummy = pd.get_dummies( df[name], prefix=name)       
        df = pd.concat( [df,df_dummy], axis=1 )
        return df

def mult_bins(df=None, columns=None, target=None, method=tree_bins, **kwargs):
    y = df[target]
    df_list = []
    models_bin = []
    for c in columns:
        model = method( **kwargs )
        model.fit(x , df[target] )
        df_list.append( model.make_bins( x, name= c+"_tb" ) )
        models_bin.append(model)
    return pd.concat( df_list, axis=1), models_bin
