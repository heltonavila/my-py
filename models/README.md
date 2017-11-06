# WeBayes

Modelo Naive-Bayes para features com distribuição Bernoulli e Normal em Python.

## Requisitos: numpy

## Utilização:
```python
from webayes import webayes
from sklearn import datasets
iris = datasets.load_iris()
clf = webayes()

# AJUSTE DO MODELO APENAS COM AS VARIÁVEIS NORMAIS
clf.fit( XB =None , XN=iris.data, Y=iris.target )

# PREDIÇÃO APENAS COM AS VARIÁVEIS NORMAIS
y_pred = clf.predict( XN=iris.data )
```

## License

[![CC0](https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

The MIT License.

-

Copyright (c) 2015-2016 [Teodoro B. Calvo](https://github.com/TeoCalvo/).
