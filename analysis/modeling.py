import pandas as pd

def binariza( X ):
    Y = pd.DataFrame()
    categorias = list(X.unique())
    for i in categorias:
        Y[ X.name + " " + i ] = (X == i)
    return Y

def binariza_colunas( df ):
	colunas = list(df.columns)
	df_result = pd.DataFrame()
	for c in colunas:
		df_result = pd.concat( [df_result, binariza( df[c] )], axis=1 )
	return df_result
