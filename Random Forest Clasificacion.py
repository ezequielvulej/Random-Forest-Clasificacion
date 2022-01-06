# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 09:13:43 2022

@author: evule
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import multiprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


#Tratamiento de datos:
wine=load_wine(return_X_y=False)
datos=np.column_stack((wine.data, wine.target))
datos=pd.DataFrame(datos, columns=np.append(wine.feature_names,'target'))
datos['objetivo']=np.where(datos['target']>0,1,0) #Trabajo con 2 clases
datos=datos.drop(['target'],axis=1)


train_features, test_features, train_labels, test_labels = train_test_split(datos.drop(columns='objetivo'), datos['objetivo'], test_size = 0.25, random_state = 42)
#print(datos.dtypes)
#Como todas las variables son numericas, no existen categoricas que requieran ser convertidas en dummies

#Busqueda de hiperparametros por cross-validation:
grillaParametros = {'n_estimators': [10], #numero de arboles a incluir en el modelo
                    'max_features': [5, 4, 3, 2], #cantidad de predictores en cada particion
                    'max_depth'   : [None, 1, 2], #maxima profundidad
                    'criterion'   : ['gini', 'entropy']
                    }

grilla = GridSearchCV(
         estimator  = RandomForestClassifier(random_state = 42),
         param_grid = grillaParametros,
         scoring    = 'accuracy',
         n_jobs     = multiprocessing.cpu_count() - 1,
         refit      = True, #Reentrenar automaticamente con los parametros optimos
         verbose    = 0,
         return_train_score = True
         )

grilla.fit(X = train_features, y = train_labels)


#Resultados:
resultados=pd.DataFrame(grilla.cv_results_)
print(resultados)

#Parametros optimos:
print(grilla.best_params_, ":", grilla.best_score_, grilla.scoring)
modelo=grilla.best_estimator_

#Evaluacion del modelo:
predicciones=modelo.predict(test_features)
#print(predicciones[:5])

matrizConfusion = confusion_matrix(y_true = test_labels, y_pred = predicciones)
accuracy = accuracy_score(y_true = test_labels ,y_pred = predicciones,normalize = True)

print("Matriz de confusion:")
print(matrizConfusion)
print("Accuracy: {}%".format(100*accuracy))
print(classification_report(y_true = test_labels ,y_pred = predicciones))
print(accuracy*100)

#Predicciones:
predicciones=modelo.predict_proba(X=test_features) #De acuerdo al n_estimators
#print(predicciones[:10,:])
#Por default, si estas probabilidades son mayores que 0.5, se predice 1 (lo puedo modificar con np.where)


#Importancia de predictores:
importancia=pd.DataFrame({'predictor':train_features.columns,'importancia':modelo.feature_importances_}) #Por gini importance (o mean decrease impurity)
print(importancia.sort_values(by=['importancia'], ascending=False))


