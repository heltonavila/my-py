import pandas as pd
import numpy as np
from sklearn import tree

class tree_bins():

	def __init__(self, **kwargs):
		self.clf = tree.DecisionTreeClassifier(**kwargs)

	def fit(self, X, y):
		self.clf.fit( X, y )

	def make_bins(self, X, name):
		df = pd.DataFrame()
		df[name] = self.clf.apply( X )
		if X.shape[1] > 1:
			df_dummy = pd.get_dummies( df[name])
		else:
			df_dummy = pd.get_dummies( df[name], prefix=name)
		df = pd.concat( [df,df_dummy], axis=1 )
		return df

def mult_bins(df=None, columns=None, target=None, method=tree_bins, **kwargs):
	y = df[target]
	df_list = []
	models_bin = []
	for c in columns:
		x = df[c].values.reshape(-1,1)
		model = method( **kwargs )
		model.fit(x , df[target] )
		df_list.append( model.make_bins( x, name= c+"_tb" ) )
		models_bin.append(model)
	return pd.concat( df_list, axis=1), models_bin
