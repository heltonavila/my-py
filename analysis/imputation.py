from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr

def columns_define(df, name_id="id", name_target="target", name_cat="_cat", name_bin="_bin", to_remove=[]):
    '''Funcao para definir os tipos de colunas que serão utilizadas na modelagem.
    - cat: categóricas
    - bin: binárias
    - num: numéricas
    - target: tributo alvo
    - id: identificador da linha'''
    
    column_id = name_id
    column_target = name_target
    columns_all = list( df.columns )
    columns_bin, columns_cat, columns_num = [],[],[]
    to_remove += [column_target, column_id]

    for r in to_remove:
        if r in columns_all:
            columns_all.remove(r)

    for c in columns_all:
        if name_cat in c:
            columns_cat.append(c)
        elif name_bin in c:
            columns_bin.append(c)
        else:
            columns_num.append(c)
            
    columns = {"column_id":column_id, "column_target":column_target,
               "columns_bin":columns_bin, "columns_cat":columns_cat, "columns_num":columns_num}
    
    return columns

def calc_miss_rate(df, all_vars=False):
    '''Calcula a taxa de missing por variável.
    - df: dataframe pandas para analisar a taxa de missing.
    - all_vars: Se deve ser retornado uma série com todas as variáveis (True) ou apenas as que possuem missing (False).'''

    missing_rate = 1 - df.count() / df.shape[0]
    
    if all_vars:
        return missing_rate
    
    else:
        missing_rate = missing_rate[missing_rate>0]
        missing_rate.sort_values(inplace=True)
        return missing_rate

def define_var_type(var_imput, columns):
    '''Verifica qual o tipo da variáevl que sera utilizada na análise: Binária, nominal ou numérica.
    - Variável a ser testada
    - columns: Dicionário de colunas com chaves do tipo das variáveis, e valores correspondente à lista de variáveis do tipo chave.
    Retorna-se o tipo da variável.'''
    for c in columns:
        if var_imput in columns[c]:
            return c

def imput_min_miss_rate(df):
    '''Treina um modelo de random forest para a variável que possui menor taxa de missing do dataframe, removendo as linhas de missing para tal treinamento.
    Pós treino do modelo, a método aplica o modelo nas linhas que há missing para a variável identificada.
    Retorna-se um dataframe com os valores imputados para a variável e o modelo que foi utilizado (já treinado) bem como as variáveis utilizadas.'''
    df = df.copy()

    columns = columns_define(df)

    missing_rate = calc_miss_rate(df)
    if len(list(missing_rate.index))==0:
        return df

    imput_var = missing_rate.index[0]
    imput_type = define_var_type(imput_var, columns)
    vars_remove = list(missing_rate.index[1:]) if len(missing_rate) > 1 else []

    columns = columns_define(df, to_remove=vars_remove+[imput_var])
    var_cov = []
    for i in ["columns_bin", "columns_cat", "columns_num"]:
        var_cov+= columns[i]

    df_imput = df[var_cov+[imput_var]].copy()
    df_imput.dropna(inplace=True)

    model = rfr if "_num" in imput_type else rfc
    imput_model = model(n_estimators=100, max_depth=10, min_samples_split=2,
                  min_samples_leaf=1, max_features='auto', max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  n_jobs=-1)

    imput_model.fit( df_imput[var_cov], df_imput[imput_var] )
    df = imput_new( df=df, model=imput_model, target=imput_var, covars=var_cov )
    
    dct_model = {"model":imput_model, "target":imput_var, "covars":var_cov}

    return df, dct_model

def imput_new( df, model, target, covars ):
    '''Realiza a imputação dos dados faltantes com base em um modelo estatístico ou de machine learning.
    - df: dataframe em que será aplicado a imputação;
    - model: Modelo a ser utilizado que ja foi ajustado/treinado
    - target: Variável a ser imputada;
    - covars: lista de variáveis que foram utilizadas na modelagem como variáveis de entrada (covariáveis);
    '''
    df = df.copy()
    na_index =  list( df[ df[target].isnull() ].index )
    if len(na_index)==0:
        return df
    values_imput =  model.predict( df.loc[ na_index , covars] )
    df.loc[na_index, target] = values_imput
    return df
    
def imput_all_missing(df):
    '''Método recursivo para imputação de dados em todas as variáveis que possuem missing no dataframe.'''
    df = df.copy()
    models = []
    miss_rate = calc_miss_rate(df, True)
    
    while miss_rate.sum() > 0:
        df, model = imput_min_miss_rate(df)
        models.append(model)
        miss_rate = calc_miss_rate(df, True)
    
    return df, models