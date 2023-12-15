#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 20:08:13 2023

@author: alvarogonzalezgonzalez
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame

sectores: DataFrame = pd.read_csv("ocupados_sectores.csv", delimiter=";")


# Análisis exploratorio de los datos para identificar necesidades de preprocesamiento
data_clean=sectores
# Revisión de valores faltantes
missing_values = data_clean.isnull().sum()
# Revisión de balance de clases para la variable objetivo 'Eco_Sector'
class_distribution = data_clean['Eco_Sector'].value_counts()

print(missing_values)
print(class_distribution)
print(data_clean.shape)

# Eliminando las filas con valores faltantes en la columna 'Total'
data_clean = data_clean.dropna(subset=['Total'])

# categorías innecesarias para la clasificación
data_clean = data_clean.query("Eco_Sector != 'Total'")
data_clean = data_clean.query("Gender != 'Both'")

# Revisión de balance de clases para la variable objetivo 'Eco_Sector'
class_distribution = data_clean['Eco_Sector'].value_counts()


# Verificando si se han eliminado correctamente las filas con valores faltantes
remaining_missing_values = data_clean.isnull().sum()
remaining_missing_values, class_distribution, data_clean.shape

print()

print(remaining_missing_values, class_distribution, data_clean.shape)



from sklearn.preprocessing import LabelEncoder
data_vis = data_clean.copy()
# Creando codificadores de etiquetas (si no existen)
label_encoders = {}
for column in ['Gender', 'Eco_Sector', 'Age', 'Period']:
    label_encoders[column] = LabelEncoder()
    data_clean[column] = label_encoders[column].fit_transform(data_clean[column])

label_encoder = LabelEncoder()
data_clean['Eco_Sector'] = label_encoder.fit_transform(data_vis['Eco_Sector'])
print(dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))



from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Normalizando la variable 'Total'
scaler = StandardScaler()
data_clean['Total'] = scaler.fit_transform(data_clean[['Total']])

# Dividiendo los datos nuevamente en características (X) y etiqueta objetivo (y)
# También realizaremos una nueva división de los datos en conjuntos de entrenamiento y prueba
X = data_clean.drop('Eco_Sector', axis=1)
y = data_clean['Eco_Sector']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creando y entrenando el modelo de árbol de decisión con los datos normalizados
decision_tree = DecisionTreeClassifier(max_depth=10, min_samples_split=21,random_state=42)
decision_tree.fit(X_train, y_train)




# Evaluando el modelo con los datos de prueba
y_pred = decision_tree.predict(X_test)
report = classification_report(y_test, y_pred)


# Imprimir el informe de clasificación de manera más visual
print("Informe de Clasificación:")
print(report)

# Calcula la matriz de confusión
confusion = confusion_matrix(y_test, y_pred)

print("Matriz de Confusión:")
print(confusion)

# Obtener la importancia de las características
feature_importances = decision_tree.feature_importances_
feature_names = X.columns

# Crear un DataFrame para visualizar la importancia de cada característica

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)
print(importance_df)
print()
print(dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))


from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

# Definir los tamaños de entrenamiento y obtener las puntuaciones de aprendizaje
train_sizes, train_scores, test_scores = learning_curve(
    decision_tree,
    X, y,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy',
    n_jobs=-1
)

# Calcular la media y la desviación estándar para los puntajes de entrenamiento y prueba
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Crear el gráfico
plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.15)
plt.plot(train_sizes, test_mean, label='Cross-validation score', color='green', marker='o')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='green', alpha=0.15)

# Añadir títulos y etiquetas
plt.title('Learning Curve')
plt.xlabel('Training Data Size')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Mostrar el gráfico
plt.show()

from sklearn.tree import export_graphviz
import graphviz

# Asumiendo que 'label_encoders' es un diccionario de LabelEncoders y 'Eco_Sector' es la columna objetivo
class_names_str = label_encoders['Eco_Sector'].classes_.astype(str)

# Luego, puedes usar 'class_names_str' como el argumento de 'class_names' en export_graphviz
dot_data = export_graphviz(
    decision_tree,
    out_file=None,
    feature_names=X.columns,
    class_names=class_names_str,
    filled=True,
    rounded=True,
    special_characters=True
)


# Usar Graphviz para visualizar el árbol
graph = graphviz.Source(dot_data)
graph.render("decision_tree")  # Guarda el árbol en el archivo "decision_tree.pdf"





