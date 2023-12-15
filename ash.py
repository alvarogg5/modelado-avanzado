#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 19:58:38 2023

@author: alvarogonzalezgonzalez
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame

labour_cost: DataFrame = pd.read_csv("coste_salarial_por_hora_efectiva_2.csv", delimiter=";")

labour_cost["Year"] = labour_cost["Period"].str[:4]
labour_cost["Cost"] = pd.to_numeric(labour_cost["Cost"])

# Revisando la estructura de los datos más detalladamente para identificar posibles problemas
labour_cost.info()

# También verificaremos si hay valores faltantes o atípicos que puedan estar afectando el análisis
labour_cost.describe()



data = labour_cost['Cost']

# Define la función ASH avanzada
def advanced_ash(data, bin_width, m):
    t0 = np.min(data) - bin_width * (m // 2)  # Ajuste para incluir bins vacíos en los extremos
    nbin = int((np.max(data) - t0) / bin_width) + 1 + (m - 1)
    bins = np.linspace(t0, nbin * bin_width + t0, nbin + 1)
    hist, _ = np.histogram(data, bins=bins)
    ash = np.zeros(nbin)

    # Calcula el ASH con histogramas desplazados
    for i in range(m):
        ash += np.roll(hist, i - m // 2)

    # Normaliza el ASH para sumar 1 como una densidad
    ash /= (ash.sum() * bin_width)

    # Calcula los centros de los bins
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Usa solo la parte del ASH que corresponde a los datos
    valid_range = (bin_centers >= np.min(data)) & (bin_centers <= np.max(data))
    bin_centers = bin_centers[valid_range]
    ash = ash[valid_range]

    return bin_centers, ash

# Conjunto de anchos de banda y valores de m para probar
bin_widths = [0.05, 0.5]
ms = [10, 50]

# Crea la figura y los ejes
plt.figure(figsize=(10, 6))

# Grafica el histograma una vez
sns.histplot(data, bins=30, kde=False, color='gray', alpha=0.2, stat='density')

# Grafica cada combinación de ASH
for bin_width in bin_widths:
    for m in ms:
        bin_centers, ash_values = advanced_ash(data, bin_width, m)
        plt.plot(bin_centers, ash_values, label=f'ASH (bin_width={bin_width}, m={m})')

# Añade títulos y etiquetas
plt.title('Ajuste ASH con diferentes paràmetros')
plt.xlabel("Coste Salarial por Hora (€)")
plt.ylabel("Densidad")
plt.legend()

# Muestra el gráfico
plt.show()
