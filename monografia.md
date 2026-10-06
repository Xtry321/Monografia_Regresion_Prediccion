# Regresión y predicción
## Integrantes:
- Araujo Champi José Eduardo
- Melgarejo Guzmán Renzo Gustavo
# Resumen

# Indice
- [Resumen](#resumen)
- [Introducción](#introducción)
- [Desarrollo](#desarrollo)
    - [Regresión Lineal Simple](#regresión-lineal-simple)
    - [Regresión Lineal Multiple](#regresión-lineal-múltiple)
    - [Predicción Usando Regresión](#predicción-usando-regresión)
    - [Variables Categóricas en Regresión](#variables-categóricas-en-regresión)
    - [Interpretación de la ecuación de regresión](#interpretación-de-la-ecuación-de-regresión)
    - [Diagnóstico de regresión](#diagnóstico-de-regresión)
    - [Regresión polinómica y splines](#regresión-polinómica-y-splines)
- [Conclusiónes](#conclusiones)
- [Referencias Bibliográficas](#referencias-bibliográficas)

# Introducción
En la actualidad, uno de los objetivos centrales tanto de la estadística como de la ciencia de datos es comprender como se relacionan entre sí las variables de un fenómeno, y poder anticipar resultados futuros. La regresión es una de las herramientas más utilizadas para lograr este propósito, ya que, permite modelar la relación entre una variable de interés y uno o más factores que podrían explicarla o predecirla, desde el precio de una vivienda en función de sus características, hasta el rendimiento de un estudiante en función de sus horas de estudio.

Aunque la regresión es una técnica que ya fue desarrollada dentro de la estadística clásica, su relevancia se ha visto renovada con el auge de la ciencia de datos y el aprendizaje automático. Por eso, comprender sus fundamentos de como se realiza, interpreta y sus limitaciones es esencial para cualquier persona que desee trabajar con datos de forma rigurosa.

La presente monografía tiene como objetivo compilar y explicar de manera clara los principales conceptos de regresión y predicción, complementando con ejemplos desarrollados en el lenguaje de programación python.

Para lograr este objetivo, el trabajo desarrollará la regresión lineal simple, presentando su ecuación fundamental y el método de mínimos cuadrados, al igual que la regresión lineal múltiple incluyendo la evaluación y selección de modelos con varios predictores. Se examina la predicción mediante regresión, enfocando los intervalos de confianza y de predicción. Luego, se analiza el tratamiento de las variables categóricas en un modelo de regresión, seguido de los principales aspectos para la interpretación de las ecuaciones de regresión. Además, se presentan las herramientas utilizadas para evaluar la calidad de un modelo. 

Finalmente, estos conceptos permiten comprender la regresión como un conjunto de herramientas orientadas no solo a describir y explicar las relaciones que existen entre las variables, sino también a desarrollar modelos que sean capaces de realizar predicciones de manera fundamentada.
# Desarrollo
## Regresión Lineal Simple
La regresión lineal simple busca modelar la relación entre dos variables numéricas. Donde existe una variable independiente o predictora ($X$) y una variable dependiente o de respuesta ($Y$), ajustando una línea recta que describa esa relación de la forma más precisa posible. Illowsky & Dean (2013) señalan que este tipo de análisis es útil cuando se quiere determinar si dos variables numéricas están relacionadas y, en el caso, cuál es la naturaleza y fuerza de esa relación.

Además, Bruce, et al. (2020) indican que, a pesar de que la correlación y la regresión están relacionadas, no son lo mismo, ya que la correlación mide qué tan fuerte es la asociación entre dos variables, la regresión permite cuantificar exactamente como cambia una variable en relación de la otra.
### La ecuación de regresión lineal
La ecuación de regresión tiene la siguiente representación:
$$Y=b_O+b_1X$$
Donde:
- Y: es la variable respuesta o variable dependiente
- X: es la variable predictora o variable independiente
- b0: es el intercepto o valor que toma Y cuando el valor de X es 0.
- b1: es la pendiente o coeficiente de regresión. Este representa cuanto cambia Y por cada unidad que cambia X.

Diez et al. (2019) coincide en definir la recta de regresión como una que permite predecir el valor de la variable respuesta a partir de la variable explicativa, enfatizando que esta ecuación describe una relación estimada y no determinística. Debido a que siempre existirá una diferencia entre lo observado y lo predicho, la cual está representada por el término error ($e_i$):
$$Y_i=b_0+b_1X_i+e_i$$
### Valores ajustados y residuos
A partir del modelo se obtienen los valores ajustados ($\hat{Y_i}$), que son las predicciones del modelo para cada observación, y los residuos ($\hat{e_i}$), que representan la distancia vertical entre el valor real y el valor ajustado:
$$\hat{e_i}=Y_i-\hat{Y_i}$$
El análisis de los residuos es importante para evaluar si el modelo lineal es adecuado: si los residuos se distribuyen de forma aleatoria alrededor de cero, sin ningún patrón visible, el modelo lineal es el apropiado. Pero si esta muestra una tendencia como una curva, esto indica que la relación entre estas variables no es realmente lineal y que el modelo se debería ajustar de otra manera.
### Método de mínimos cuadrados
Este es el criterio más utilizado para calcular el valor de $b_0$ y $b_1$, con el objetivo de que se encuentre los valores que minimicen la suma de residuos al cuadrado (RSS):
$$RSS=\sum_{i=1}^{n} (\hat{e_i})^2$$
$$RSS=\sum_{i=1}^{n} (Y_i-\hat{Y}_i)^2$$
$$RSS=\sum_{i=1}^{n} (Y_i-\hat{b}_0-\hat{b}_iXi)^2$$
Aquí se eleva cada residuo al cuadrado, en lugar de utilizar su valor absoluto, por dos razones principales:
- Evista que residuos positivos y negativos se cancelen entre sí al sumarlos.
- Permite obtener una función diferenciable.

Esta última razón hace posible que se pueda obtener analíticamente los valores que minimizan el error total.

Una limitación de este método es que al elevarse al cuadrado los valores de los residuos, afecta desproporcionadamente en el cálculo los valores atípicos (outliers), por lo que un dato que sea muy extremo puede alterar significativamente la línea de regresión estimada.
### Predicción versus explicación
La regresión se puede utilizar para dos propósitos distintos:
- Explicar una relación, donde se enfoca en el coeficiente $b_1$ para comprender la relación entre estas variables.
- Predecir valores futuros, enfocandose en los valores ajustados $\hat{Y}$.

Además, esta relación lineal no es suficiente para interpretar causalidad entre las dos variables. La evidencia de causa efecto debe ser sustentada con conocimiento sobre el fenómeno y no solo en la ecuación matemática.
### Ejemplo práctico en Python
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.DataFrame({
    'Exposure': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    'PEFR': [450, 430, 420, 400, 390, 370, 360, 340, 320, 310, 300]
})

predictors = ['Exposure']
outcome = 'PEFR'

model = LinearRegression()
model.fit(data[predictors], data[outcome])

print(f'Intercepto (b0): {model.intercept_:.3f}')
print(f'Coeficiente Exposure (b1): {model.coef_[0]:.3f}')

fitted = model.predict(data[predictors])
residuals = data[outcome] - fitted

rmse = np.sqrt(mean_squared_error(data[outcome], fitted))
r2 = r2_score(data[outcome], fitted)

print(f'RMSE: {rmse:.2f}')
print(f'R²: {r2:.4f}')

import matplotlib.pyplot as plt

plt.scatter(data['Exposure'], data['PEFR'], color='steelblue', label='Datos observados')
plt.plot(data['Exposure'], fitted, color='red', label='Línea de regresión')
plt.xlabel('Exposición (años)')
plt.ylabel('PEFR')
plt.title('Regresión lineal simple: Exposición vs. PEFR')
plt.legend()
plt.show()
```

## Regresión Lineal Múltiple
Cuando el fenómeno que se quiere predecir depende de más de una variable, la regresión lineal simple no es suficiente. La regresión lineal múltiple extiende el modelo simple incorporando varias variables predictoras simultáneamente, permitiendo así explicar un resultado a partir de un conjunto más amplio de características. Diez et al. (2019) plantea este modelo como una generalización natural, ya que en lugar de una sola variable explicativa, se incluyen k variables predictoras, cada una con su propio coeficiente, bajo el supuesto de que la relación de cada una con la variable respuesta se mantiene lineal.

La ecuación general se representa de la siguiente manera:
$$Y=b_0+b_1X_1+b_2X_2+...+b_pX_p+e$$
Cada coeficiente $b_j$ representa el cambio esperado en $Y$ por cada unidad de cambio en $X_j$, manteniendo constantes todas las demás variables del modelo. Esta condición es clave para interpretar correctamente los coeficientes, ya que en la práctica los predictores rara vez son completamente independientes entre sí.
### Evaluación del modelo: MRSE, RSE y $R^2$
Al igual que en los modelos de regresión lineal simple, es necesario evaluar qué tan bien se ajusta el modelo a los datos. Donde las métricas más relevantes son:
- RMSE (Root Mean Square Error): Este es la raíz cuadrada del promedio de los errores al cuadrado. Es muy utilizada para comparar modelos de regresión.  
Según Kosourova (2025), este valor muestra por término medio, lo lejos que están las predicciones de los valores reales. Por lo que, un RMSE menor sugiere menores errores medios de predicción, resultando en predicciones más precisas.  
$$RMSE=\sqrt{\frac{\sum_{i=1}^{n} (y_i-\hat{y}_i)^2}{n}}$$
- RSE (Residual Standard Error): Su función es similar a la de RMSE, pero este está ajustado por los grados de libertad. Por lo que se infiere que también un menor valor de RSE indica que el modelo puede realizar predicciónes más precisas. Donde p es la cantidad de variables predictoras o variables independientes.
$$RSE=\sqrt{\frac{\sum_{i=1}^{n} (y_i-\hat{y}_i)^2}{n-p-1}}$$
- $R^2$: El coeficiente de correlación mide la proporción de la varianza de $Y$ explicada por el modelo, con valores entre 0 y 1.
$$R^2=1-\frac{\sum_{i=1}^{n} (y_i-\hat{y}_i)^2}{\sum_{i=1}^{n} (y_i-\bar{y}_i)^2}$$
### Validación cruzada
Las métricas anteriores se calculan sobre los mismos datos usados para ajustar el modelo, por lo que se conocen como métricas “in-sample”. Pero para evaluar de una manera más realista qué tan bien generalizará el modelo a datos nuevos, se recurre a la validación cruzada.

La validación cruzada amplía la idea de una muestra de reserva “holdout sample” a múltiples muestras de reserva secuenciales. El algoritmo para la validación cruzada básica de k particiones “k-fold” es el siguiente:
1. Reservar 1/k de los datos como muestra de reserva.
2. Entrenar el modelo con los datos restantes.
3. Evaluar el modelo sobre la muestra de reserva de 1/k y registrar las métricas de evaluación del modelo necesarias.
4. Reincorporar la primera fracción de 1/k de los datos y reservar la siguiente fracción de 1/k.
5. Repetir los pasos 2 y 3.
6. Repetir el proceso hasta que cada registro haya formado parte de la muestra de reserva.
7. Calcular el promedio o combinar de otro modo las métricas de evaluación del modelo.
### Selección de modelos y regresión por pasos
En casos de que se cuente con muchas variables candidatas, no es necesario incluir todas para crear el mejor modelo. Bruce et al. (2020) explica que es preferible modelos más simples cunado el ajuste es comparable. Para eso existen métricas como el AIC, este penaliza la cantidad de variables incluidas.

La fórmula se representa de la siguiente manera:
$$AIC=2P+nlog(RSS/n)$$
- P: Cantidad de parámetros del modelo.
- n: Tamaño de la muestra
- RSS: Suma de los residuos al cuadrado

Los métodos automáticos para seleccionar variables incluyen:
- Selección hacia adelante (forward selection): Parte de un modelo vacío y se agregan variables una por una.
- Eliminación hacia atrás (backward elimination): Parte del modelo completo y se van eliminando variables poco significativas.
- Selección por pasos (stepwise): Combina ambos enfoques, agregando y quitando variables según el aporte al modelo.

Además, un punto importante que es señalado por Bruce et al. (2020) es que el $R^2$ siempre aumenta al añadir más variables predictoras, incluso si estas no aportan información relevante al modelo. Por lo cual, recomiendan usar el $R^2$ ajustado, que penaliza la inclusión de predictores poco útiles:
$$R_{adj}^2=1-(1-R^2)\frac{n-1}{n-p-1}$$
Esto ayuda a identificar si una variable predictora aporta mejora o no, para descartarla del modelo.
### Ejemplo práctico en Python
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
cantidad_casas = 200

datos = pd.DataFrame({
    'metros_cuadrados': np.random.normal(180, 45, cantidad_casas),
    'tamanio_terreno': np.random.normal(750, 280, cantidad_casas),
    'banios': np.random.randint(1, 4, cantidad_casas),
    'habitaciones': np.random.randint(2, 6, cantidad_casas),
    'grado_construccion': np.random.randint(5, 12, cantidad_casas)
})

datos['precio_venta'] = (
    50000
    + datos['metros_cuadrados'] * 1500
    + datos['grado_construccion'] * 20000
    - datos['habitaciones'] * 5000
    + np.random.normal(0, 30000, cantidad_casas)
)

predictores = ['metros_cuadrados', 'tamanio_terreno', 'banios', 'habitaciones', 'grado_construccion']
respuesta = 'precio_venta'

modelo = LinearRegression()
modelo.fit(datos[predictores], datos[respuesta])

print(f'Intercepto: {modelo.intercept_:.3f}')
print('Coeficientes:')
for nombre, coeficiente in zip(predictores, modelo.coef_):
    print(f'  {nombre}: {coeficiente:.3f}')

valores_ajustados = modelo.predict(datos[predictores])
residuos = datos[respuesta] - valores_ajustados

rmse = np.sqrt(mean_squared_error(datos[respuesta], valores_ajustados))
r2 = r2_score(datos[respuesta], valores_ajustados)

print(f'RMSE: {rmse:.2f}')
print(f'R²: {r2:.4f}')

plt.figure(figsize=(6, 6))
plt.scatter(datos[respuesta], valores_ajustados, color='steelblue', alpha=0.6)

minimo = min(datos[respuesta].min(), valores_ajustados.min())
maximo = max(datos[respuesta].max(), valores_ajustados.max())
plt.plot([minimo, maximo], [minimo, maximo], color='red', linestyle='--', 
         label='Predicción perfecta')

plt.xlabel('Precio real')
plt.ylabel('Precio predicho')
plt.title('Valores reales vs. valores predichos')
plt.legend()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(valores_ajustados, residuos, color='seagreen', alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')

plt.xlabel('Valores predichos')
plt.ylabel('Residuos')
plt.title('Gráfico de residuos')
plt.show()
```
## Predicción Usando Regresión
El uso de la regresión en el contexto de ciencia de datos es usado principalmente para la predicción. Usar el modelo ajustado para estimar el valor de la variable respuesta en observaciones nuevas, cuyo valor aún no se conoce. Esto difiere con el uso tradicional de la regresión en la estadística clásica, má orientado a explicar relaciones entre las variables dentro de los datos ya observados.
### Los peligros de la extrapolación
La extrapolación es utilizar el modelo de regresión para poder predecir valores que se encuentren fuera del rango de los valores predictores que se utilizaron para realizar el entrenamiento del modelo. Esta es una práctica riesgosa, ya que no existe garantía de que la relación lineal observada se mantenga fuera de los límites de los datos.
### Intervalos de confianza y predicción:
Al realizar una predicción con un modelo de regresión, no basta con dar un único valor puntual: es necesario cuantificar la incertidumbre que puede estar asociado a esa predicción. Existen dos tipos de intervalos que responden a distintas preguntas:
- Intervalo de confianza: Cuantifica la incertidumbre alrededor de un estadístico calculado a partir de múltiples valores.
- Intervalor de predicción: Cuantifica la incertidumbre alrededor de un valor individual específico.

Bruce et al. (2020) explica que el intervalo de predicción va a ser siempre más ancho que el intervalo de confianza para el mismo nivel de confianza, ya que la incertidumbre sobre la ubicación en la línea de regresión se suma la variabilidad propia de cada observación individual. La incertidumbre de una predicción individual proviene de dos fuentes:
- La incertidumbre sobre los propios coeficientes del modelo, que fueron estimados a partir de una muestra y no son los valores poblacionales “verdaderos”.
- La variabilidad inherente a cada observación individual, es decir, el hecho de que incluso conociendo exactamente la ecuación poblacional, dos casos con los mismos valores X pueden tener resultados Y distintos.

Algo importante a resaltar es que, el intervalo se estrecha cuando el valor de X para el cual se está prediciendo está cerca de la media de los datos observados (x̄), y se ensancha cuanto más se aleja de ella.
### El enfoque bootstrap para construir intervalos
Se propone una alternativa computacional para construir estos intervalos, basada en el remuestreo bootstrap. Donde el procedimiento general es:
1. Tomar una muestra bootstrap de los datos originales.
2. Ajustar la regresión sobre esa muestra y genera una predicción para el nuevo caso.
3. Tomar un residuo al azar del modelo original y sumarlo a la predicción, para incorporar la variabilidad individual.
4. Repetir el proceso muchas veces.
5. Calcular los percentiles 2.5 y 97.5 de los resultados obtenidos, para construir un intervalo del 95%.
## Variables categóricas en regresión
## Interpretación de la ecuación de regresión
## Diagnóstico de regresión
## Regresión polinómica y splines
# Conclusiones
# Referencias Bibliográficas
Bruce, P., Bruce, A., & Gedeck, P. (2020). *Practical statistics for data scientists: 50+ essential concepts using R and Python* (2nd ed.). O'Reilly Media.

Diez, D., Barr, C., & Çetinkaya-Rundel, M. (2019). *OpenIntro Statistics (4th ed.).* https://www.openintro.org/book/os/

Illowsky, B., & Dean, S. (2023). *Introductory statistics 2e. OpenStax.* https://openstax.org/books/introductory-statistics-2e/

Kosourova, E. (2025, 19 de junio). *Explicación del RMSE: Guía para la precisión de la predicción de regresión.* DataCamp. https://www.datacamp.com/es/tutorial/rmse
