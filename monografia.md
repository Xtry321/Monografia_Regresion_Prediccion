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
Este es el criterio más utilizado para calcular el valor de $b_0$ y $b_1$. Las ecuaciones que se utilizan para hallar estos valores son:
$$b_1=\frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sum(x_i-\bar{x})^2}$$
$$b_0=\bar{y}-b_1\bar{x}$$

Este se realiza con el objetivo de que se encuentre los valores que minimicen la suma de residuos al cuadrado (RSS):
$$RSS=\sum_{i=1}^{n} (\hat{e_i})^2$$
$$RSS=\sum_{i=1}^{n} (Y_i-\hat{Y}_i)^2$$
$$RSS=\sum_{i=1}^{n} (Y_i-\hat{b}_0-\hat{b}_iXi)^2$$
Aquí se eleva cada residuo al cuadrado, en lugar de utilizar su valor absoluto, por dos razones principales:
- Evita que residuos positivos y negativos se cancelen entre sí al sumarlos.
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

# Exposición al polvo de algodon
datos = pd.DataFrame({
    'anios_exposicion': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    'capacidad_pulmonar': [450, 430, 420, 400, 390, 370, 360, 340, 320, 310, 300]
})

predictor = ['anios_exposicion']
respuesta = 'capacidad_pulmonar'

modelo = LinearRegression()

modelo.fit(datos[predictor], datos[respuesta])

print(f'Intercepto (b0): {modelo.intercept_:.3f}')
print(f'Coeficiente de años de exposición (b1): {modelo.coef_[0]:.3f}')

valores_ajustados = modelo.predict(datos[predictor])

residuos = datos[respuesta] - valores_ajustados

rmse = np.sqrt(mean_squared_error(datos[respuesta], valores_ajustados))
r2 = r2_score(datos[respuesta], valores_ajustados)

print(f'RMSE: {rmse:.2f}')
print(f'R²: {r2:.4f}')

import matplotlib.pyplot as plt

plt.scatter(datos['anios_exposicion'], datos['capacidad_pulmonar'], color='steelblue', label='Datos observados')
plt.plot(datos['anios_exposicion'], valores_ajustados, color='red', label='Línea de regresión')
plt.xlabel('Años de exposición')
plt.ylabel('Capacidad pulmonar (PEFR)')
plt.title('Regresión lineal simple: Exposición vs. Capacidad pulmonar')
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

<img src="https://upload.wikimedia.org/wikipedia/commons/1/18/Esquema_castell%C3%A0.jpg?utm_source=es.wikipedia.org&utm_campaign=index&utm_content=original" width="700">

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
La regresión requiere entradas numéricas, por lo que las variables categóricas (o factores), que toman un número limitado de valores discretos, deben recodificarse antes de incluirse en el modelo. Bruce et al. (2020) señalan que el enfoque más común es convertir la variable en un conjunto de variables binarias llamadas variables *dummy*. Un caso especial es la variable binaria (sí/no), también llamada variable indicadora.
### Representación mediante variables dummy
Supongamos una variable `tipo_propiedad` con tres niveles: *Multiplex*, *Unifamiliar* y *Casa adosada*. Se crea una columna binaria por cada nivel, que vale 1 si el registro pertenece a ese nivel y 0 en caso contrario. En aprendizaje automático esta representación se conoce como *one hot encoding* y es la forma estándar de tratar factores en modelos como árboles de decisión o vecinos más cercanos.

En regresión lineal, sin embargo, un factor con $P$ niveles se representa con solo $P-1$ columnas. El modelo incluye un intercepto, por lo que una vez definidas $P-1$ variables binarias, el valor de la última queda determinado por las demás y resulta redundante. Incluir las $P$ columnas genera **multicolinealidad**, lo que impide obtener una solución única para los coeficientes. El nivel omitido se denomina **nivel de referencia**, y los coeficientes de los demás niveles se interpretan como la diferencia respecto a él. Esta forma de codificación se conoce como *reference coding* o *treatment coding*.

Existen otras codificaciones, como la codificación por desviación (*sum contrasts*), que compara cada nivel con la media global, o la codificación polinómica, apropiada para factores ordenados. Bruce et al. (2020) indican que, salvo con factores ordenados, un científico de datos rara vez necesitará algo distinto a la codificación de referencia o al *one hot encoding*.
### Factores con muchos niveles
Algunas variables categóricas generan una cantidad enorme de dummies. Por ejemplo, en Estados Unidos existen unos 43,000 códigos postales. En estos casos es recomendable explorar si las categorías aportan información útil y decidir si se mantienen todas o se consolidan. Un enfoque sencillo es agrupar los niveles según otra variable, como el precio. Un enfoque mejor, según Bruce et al. (2020), es formar los grupos a partir de los **residuos de un modelo inicial**:
1. Ajustar un modelo preliminar sin la variable categórica.
2. Calcular la mediana de los residuos para cada nivel de la variable.
3. Ordenar los niveles según esa mediana y dividirlos en un número reducido de grupos (por ejemplo, cinco).

De esta forma, niveles que el modelo inicial subestima o sobreestima de manera similar quedan en el mismo grupo, y la nueva variable (por ejemplo, `ZipGroup`) captura el efecto de la ubicación con pocos coeficientes.
### Factores ordenados
Algunas variables categóricas reflejan un orden natural, como la calificación de un préstamo (A, B, C, ...) o el grado de construcción de una vivienda (1 = cabaña, 13 = mansión). Estas se denominan variables categóricas ordenadas y, con frecuencia, pueden convertirse en valores numéricos y usarse tal cual. Tratarlas como numéricas conserva la información del orden, que se perdería al convertirlas en un factor sin orden.
### Ejemplo práctico en Python
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

np.random.seed(42)
n = 300

tipos = ['Multiplex', 'Unifamiliar', 'Casa adosada']
efecto_tipo = {'Multiplex': 0, 'Unifamiliar': -25000, 'Casa adosada': -60000}

datos = pd.DataFrame({
    'metros_cuadrados': np.random.normal(180, 45, n),
    'tipo_propiedad': np.random.choice(tipos, n, p=[0.1, 0.6, 0.3])
})

datos['precio_venta'] = (
    80000
    + datos['metros_cuadrados'] * 1500
    + datos['tipo_propiedad'].map(efecto_tipo)
    + np.random.normal(0, 25000, n)
)

# Se define el orden de las categorías para que "Multiplex" sea la referencia
datos['tipo_propiedad'] = pd.Categorical(datos['tipo_propiedad'], categories=tipos)

# drop_first=True elimina la primera categoría (Multiplex) y evita la multicolinealidad
X = pd.get_dummies(datos[['metros_cuadrados', 'tipo_propiedad']], drop_first=True, dtype=int)
y = datos['precio_venta']

modelo = LinearRegression()
modelo.fit(X, y)

print(f'Intercepto: {modelo.intercept_:.3f}')
for nombre, coef in zip(X.columns, modelo.coef_):
    print(f'  {nombre}: {coef:.3f}')
```
Los coeficientes de `tipo_propiedad_Unifamiliar` y `tipo_propiedad_Casa adosada` indican cuánto menos (o más) vale una vivienda de ese tipo respecto a un *Multiplex* de igual tamaño.

Para consolidar un factor con muchos niveles a partir de los residuos de un modelo inicial:
```python
# Supongamos un factor con 20 códigos postales
datos['codigo_postal'] = np.random.choice(np.arange(98001, 98021), n)

# Residuos del modelo inicial (sin la variable categórica)
modelo_inicial = LinearRegression().fit(datos[['metros_cuadrados']], y)
datos['residuo'] = y - modelo_inicial.predict(datos[['metros_cuadrados']])

# Mediana del residuo y cantidad de registros por código postal
zonas = (datos.groupby('codigo_postal')['residuo']
              .agg(mediana_residuo='median', cantidad='count')
              .sort_values('mediana_residuo'))

# División en 5 grupos según la mediana del residuo
zonas['grupo_zona'] = pd.qcut(zonas['mediana_residuo'], 5, labels=False)

datos = datos.join(zonas['grupo_zona'], on='codigo_postal')
datos['grupo_zona'] = datos['grupo_zona'].astype('category')
```

---
## Interpretación de la ecuación de regresión
Aunque en ciencia de datos el uso principal de la regresión es predecir, en ocasiones también interesa comprender la naturaleza de la relación entre los predictores y la respuesta. Esto requiere cuidado, porque los coeficientes pueden ser engañosos en presencia de ciertos problemas.
### Predictores correlacionados
En regresión múltiple es frecuente que los predictores estén correlacionados entre sí. Bruce et al. (2020) ilustran esto con un modelo de precios de viviendas en el que el coeficiente de `Bedrooms` (número de habitaciones) resulta **negativo**, lo que sugeriría que agregar una habitación reduce el valor de la casa. La explicación es que las casas más grandes tienden a tener más habitaciones, y es el **tamaño** el que impulsa el precio. Al comparar dos casas del mismo tamaño, tener más habitaciones (y por tanto más pequeñas) resulta menos deseable. Cuando se retiran del modelo las variables de tamaño, el coeficiente de las habitaciones se vuelve positivo, pero esto ocurre porque la variable pasa a actuar como sustituto (*proxy*) del tamaño.

Por lo tanto, la correlación entre predictores dificulta la interpretación del signo y la magnitud de cada coeficiente, y además puede inflar los errores estándar de las estimaciones.
### Multicolinealidad
La multicolinealidad es el caso extremo de la correlación entre predictores: existe redundancia entre ellos. Hay multicolinealidad perfecta cuando un predictor puede expresarse como combinación lineal de otros. Suele presentarse cuando:
- Una variable se incluye varias veces por error.
- Se crean $P$ variables dummy en lugar de $P-1$ a partir de un factor.
- Dos variables están casi perfectamente correlacionadas.

Con multicolinealidad perfecta, la regresión no tiene una solución bien definida, por lo que deben eliminarse variables hasta que desaparezca. Algunos paquetes de software manejan automáticamente ciertos casos, pero cuando la multicolinealidad no es perfecta el software puede devolver una solución cuyos resultados son **inestables**. En métodos no lineales como árboles, agrupamiento o vecinos más cercanos, este problema es menos grave.
### Variables de confusión
Con predictores correlacionados el problema es de *comisión*: se incluyen variables distintas con una relación predictiva similar con la respuesta. Con las **variables de confusión** (*confounding variables*) el problema es de *omisión*: falta en el modelo una variable importante, y la interpretación ingenua de los coeficientes puede llevar a conclusiones inválidas.

En el ejemplo de King County, el modelo original no incluía la ubicación, un predictor muy importante del precio. Por eso, los coeficientes de `SqFtLot`, `Bathrooms` y `Bedrooms` eran todos negativos. Al incorporar la variable `ZipGroup`, los coeficientes de `SqFtLot` y `Bathrooms` pasaron a ser positivos, y una vivienda en el grupo de zona más caro resultó valer casi \$340,000 más. El coeficiente de `Bedrooms` siguió siendo negativo, un fenómeno conocido en el mercado inmobiliario.
### Interacciones y efectos principales
Los **efectos principales** son los predictores que aparecen de forma individual en la ecuación. Usar solo efectos principales supone que la relación entre un predictor y la respuesta es independiente de los demás predictores, lo cual con frecuencia no se cumple. Una **interacción** es una relación interdependiente entre dos o más predictores y la respuesta. Por ejemplo, un metro cuadrado adicional no aporta el mismo valor en una zona económica que en una zona exclusiva.

En el modelo de King County, la interacción entre el tamaño y la zona mostró que, para viviendas de la zona más barata, cada pie cuadrado adicional añadía unos \$118 al precio, mientras que en la zona más cara añadía unos \$342, casi tres veces más.

Cuando hay muchas variables, decidir qué interacciones incluir es difícil. Bruce et al. (2020) mencionan varios enfoques: usar conocimiento previo del problema, aplicar regresión por pasos, usar regresión penalizada, o, lo más común, emplear modelos de árbol y sus derivados (*random forest* y *gradient boosting*), que buscan automáticamente las interacciones relevantes.
### Ejemplo práctico en Python
```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)
n = 500

datos = pd.DataFrame({'metros_cuadrados': np.random.normal(180, 45, n)})

# El número de habitaciones está correlacionado con el tamaño
datos['habitaciones'] = np.clip(
    np.round(datos['metros_cuadrados'] / 45 + np.random.normal(0, 0.7, n)), 1, 8
)
datos['zona'] = np.random.choice(['A', 'B', 'C'], n)

# Efecto de la zona sobre el precio por metro cuadrado (interacción)
pendiente = datos['zona'].map({'A': 1000, 'B': 1800, 'C': 3000})

datos['precio_venta'] = (
    50000
    + datos['metros_cuadrados'] * pendiente
    - datos['habitaciones'] * 8000
    + np.random.normal(0, 30000, n)
)

# 1) Con el tamaño incluido: el coeficiente de habitaciones es negativo
m1 = smf.ols('precio_venta ~ metros_cuadrados + habitaciones', data=datos).fit()
print(m1.params)

# 2) Sin el tamaño: las habitaciones actúan como proxy y su coeficiente es positivo
m2 = smf.ols('precio_venta ~ habitaciones', data=datos).fit()
print(m2.params)

# 3) Modelo con interacción entre tamaño y zona
m3 = smf.ols('precio_venta ~ metros_cuadrados * C(zona) + habitaciones', data=datos).fit()
print(m3.summary())
```
En el modelo 3, el coeficiente de `metros_cuadrados` es la pendiente para la zona de referencia (A), y los términos `metros_cuadrados:C(zona)[T.B]` y `metros_cuadrados:C(zona)[T.C]` indican cuánto aumenta esa pendiente en las otras zonas.

---

## Diagnóstico de regresión
En un contexto explicativo se aplican, además de las métricas de evaluación, varios pasos para verificar qué tan bien se ajusta el modelo a los datos, basados principalmente en el análisis de los residuos. Bruce et al. (2020) señalan que estos pasos no miden directamente la precisión predictiva, pero aportan información útil también en un contexto predictivo.
### Valores atípicos (outliers)
En regresión, un valor atípico es un registro cuyo valor real de $y$ está lejos del valor predicho. Se detecta mediante el **residuo estandarizado**, es decir, el residuo dividido por el error estándar de los residuos, que se interpreta como el número de errores estándar que el registro se aleja de la línea de regresión. No existe una teoría estadística que separe los atípicos de los no atípicos, solo reglas prácticas; en el ejemplo del libro, se revisan los residuos estandarizados de mayor magnitud (por ejemplo, mayores a 2.5 en valor absoluto).

En el ejemplo de las ventas en el código postal 98105, el mayor residuo correspondía a una casa vendida en \$119,748, muy por debajo de lo que el modelo esperaba (un error de más de cuatro errores estándar). Al revisar la escritura de la venta se comprobó que solo se había transferido el 25% de la propiedad, por lo que el registro era anómalo y debía excluirse. Los atípicos también pueden deberse a errores de digitación o a unidades mal registradas.

En problemas con muchos datos, los atípicos generalmente no afectan de forma importante el ajuste. Sin embargo, son el centro de la **detección de anomalías**, donde encontrarlos es el objetivo, por ejemplo para identificar fraudes o acciones accidentales.
### Valores influyentes
Una observación es **influyente** cuando su ausencia modificaría de forma significativa la ecuación de regresión, aunque no tenga un residuo grande. Se dice entonces que tiene alto **apalancamiento** (*leverage*). Las principales métricas son:
- **Valor hat (*hat-value*)**: medida común de apalancamiento. Valores superiores a $2(P+1)/n$ indican un registro de alto apalancamiento.
- **Distancia de Cook**: define la influencia como una combinación del apalancamiento y el tamaño del residuo. Como regla práctica, una observación tiene alta influencia si su distancia de Cook supera $4/(n-P-1)$.

Un **gráfico de influencia** (o de burbujas) combina los residuos estandarizados, los valores hat y la distancia de Cook en una sola figura. En el ejemplo de Bruce et al. (2020), al eliminar los puntos con distancia de Cook mayor a 0.08, el coeficiente de `Bathrooms` cambió de forma drástica (de 2,282 a −16,132). Con todo, identificar observaciones influyentes es útil para ajustar modelos solo en conjuntos de datos pequeños, ya que con muchos registros es poco probable que uno solo tenga un peso determinante.
### Heterocedasticidad, no normalidad y errores correlacionados
La distribución de los residuos es relevante sobre todo para la validez de la inferencia estadística formal (pruebas de hipótesis y valores $p$). Para que esta sea válida, se supone que los residuos son normales, tienen la misma varianza y son independientes. Sin embargo, el método de mínimos cuadrados es insesgado, y en muchos casos óptimo, bajo una amplia variedad de supuestos distribucionales, por lo que en ciencia de datos no suele ser una gran preocupación.
- **Heterocedasticidad**: ausencia de varianza constante de los residuos a lo largo del rango de valores predichos. Se analiza graficando el valor absoluto de los residuos frente a los valores predichos y superponiendo un suavizador (por ejemplo, *loess*). Puede indicar que el modelo está incompleto y que algo queda sin explicar en cierto rango de valores. En el ejemplo de Bruce et al. (2020), la varianza de los residuos aumentaba para las viviendas más caras.
- **No normalidad**: los residuos pueden tener colas más largas que una distribución normal. Esto afecta principalmente a los intervalos de confianza basados en fórmulas, pero rara vez es un problema para la precisión predictiva.
- **Errores correlacionados**: se presentan sobre todo en datos recogidos a lo largo del tiempo o del espacio. El estadístico de Durbin-Watson permite detectar autocorrelación significativa en series de tiempo.

Aunque una regresión viole algún supuesto distribucional, lo que más interesa en ciencia de datos es la precisión predictiva; la revisión de la heterocedasticidad puede revelar señal en los datos que el modelo no ha capturado.
### Gráficos de residuos parciales y no linealidad
Los **gráficos de residuos parciales** permiten visualizar qué tan bien el ajuste estimado explica la relación entre un predictor y la respuesta, aislando esa relación y teniendo en cuenta todos los demás predictores. El residuo parcial para el predictor $X_i$ es el residuo ordinario más el término de regresión asociado a ese predictor:
$$\text{Residuo parcial}=\text{Residuo}+\hat{b}_iX_i$$
En el gráfico se coloca $X_i$ en el eje horizontal y los residuos parciales en el vertical, superponiendo la línea de regresión y un suavizador. Si ambos no coinciden, hay indicios de no linealidad. En el ejemplo de Bruce et al. (2020), la relación entre el tamaño de la vivienda y el precio resultó no lineal: la recta subestimaba el precio de las casas muy pequeñas y sobreestimaba el de las de tamaño intermedio, lo cual tiene sentido, pues agregar 500 pies cuadrados a una casa pequeña representa un cambio mucho mayor que a una casa grande.
### Ejemplo práctico en Python
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

np.random.seed(42)
n = 200

datos = pd.DataFrame({
    'metros_cuadrados': np.random.uniform(50, 400, n),
    'grado_construccion': np.random.randint(5, 12, n)
})

# Relación no lineal con el tamaño y varianza creciente (heterocedasticidad)
datos['precio_venta'] = (
    20000
    + 800 * datos['metros_cuadrados']
    + 4 * datos['metros_cuadrados'] ** 2
    + 15000 * datos['grado_construccion']
    + np.random.normal(0, 1, n) * datos['metros_cuadrados'] * 150
)

# Se introduce un valor atípico (por ejemplo, una venta parcial)
datos.loc[0, 'precio_venta'] = 30000

X = sm.add_constant(datos[['metros_cuadrados', 'grado_construccion']])
resultado = sm.OLS(datos['precio_venta'], X).fit()

influencia = OLSInfluence(resultado)
res_estandarizados = influencia.resid_studentized_internal
hat = influencia.hat_matrix_diag
cook = influencia.cooks_distance[0]

# 1) Valor atípico: residuo estandarizado de mayor magnitud
idx = np.abs(res_estandarizados).idxmax()
print('Registro más atípico:', idx, '| residuo estandarizado:', round(res_estandarizados[idx], 2))

# 2) Observaciones influyentes según las reglas prácticas
P = 2
umbral_hat = 2 * (P + 1) / n
umbral_cook = 4 / (n - P - 1)
print('Registros con alto apalancamiento:', np.where(hat > umbral_hat)[0])
print('Registros con alta distancia de Cook:', np.where(cook > umbral_cook)[0])

# 3) Gráfico de influencia
fig, ax = plt.subplots(figsize=(6, 6))
ax.axhline(-2.5, linestyle='--', color='C1')
ax.axhline(2.5, linestyle='--', color='C1')
ax.scatter(hat, res_estandarizados, s=1000 * np.sqrt(cook), alpha=0.5)
ax.set_xlabel('Valores hat')
ax.set_ylabel('Residuos estandarizados')
ax.set_title('Gráfico de influencia')
plt.show()

# 4) Heterocedasticidad: |residuo| vs. valores predichos
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(resultado.fittedvalues, np.abs(resultado.resid), alpha=0.4)
ax.set_xlabel('Valores predichos')
ax.set_ylabel('|Residuo|')
ax.set_title('Valor absoluto de los residuos vs. valores predichos')
plt.show()

# 5) Gráfico de residuos parciales para el tamaño de la vivienda
sm.graphics.plot_ccpr(resultado, 'metros_cuadrados')
plt.show()
```

---

## Regresión polinómica y splines
La relación entre la respuesta y un predictor no siempre es lineal. La respuesta a la dosis de un medicamento, por ejemplo, rara vez se duplica al duplicar la dosis, y la demanda de un producto se satura a partir de cierto gasto en publicidad. Bruce et al. (2020) presentan varias formas de extender la regresión para capturar estos efectos no lineales.

Conviene aclarar que, en el sentido estadístico estricto, la *regresión no lineal* se refiere a modelos que no pueden ajustarse por mínimos cuadrados y requieren optimización numérica. Los polinomios y splines siguen siendo modelos lineales en los coeficientes, por lo que se ajustan con las mismas herramientas.
### Regresión polinómica
La regresión polinómica añade términos polinómicos (cuadrados, cubos, etc.) a la ecuación de regresión. Por ejemplo, una regresión cuadrática entre la respuesta $Y$ y el predictor $X$ tiene la forma:
$$Y=b_0+b_1X+b_2X^2+e$$
Ahora hay dos coeficientes asociados al predictor: uno para el término lineal y otro para el cuadrático. En el ejemplo de King County, el gráfico de residuos parciales mostró que el ajuste cuadrático seguía más de cerca el suavizador de los residuos parciales que una recta. No obstante, añadir términos de grado superior (cúbico, cuártico) suele introducir una "ondulación" indeseable en la ecuación.
### Splines
Una alternativa, a menudo superior, son los *splines*. Su nombre proviene de las tiras flexibles de madera que los dibujantes, sobre todo en la construcción de barcos y aviones, doblaban con pesos (llamados "patos") para trazar curvas suaves. Matemáticamente, un spline es una serie de polinomios por tramos, unidos de forma suave en puntos fijos del predictor llamados **nudos** (*knots*).

Para ajustar un spline deben especificarse dos parámetros: el grado de los polinomios (por ejemplo, 3 para un spline cúbico) y la ubicación de los nudos (por ejemplo, en los cuartiles del predictor). A diferencia de un término lineal, los coeficientes de un spline **no son interpretables**, por lo que conviene usar la visualización (el gráfico de residuos parciales) para evaluar su ajuste.

Bruce et al. (2020) advierten que un mejor ajuste visual no implica necesariamente un mejor modelo. En su ejemplo, el spline sugería que las viviendas muy pequeñas (menos de 1,000 pies cuadrados) valían más que otras ligeramente más grandes, algo sin sentido económico, posiblemente un efecto de una variable de confusión.
### Modelos aditivos generalizados (GAM)
Los polinomios pueden no ser lo bastante flexibles y los splines exigen definir los nudos de antemano. Los **modelos aditivos generalizados** (*Generalized Additive Models*, GAM) son una técnica flexible que ajusta automáticamente una regresión con splines, seleccionando los "mejores" nudos. En R se usa el término `s(variable)` del paquete `mgcv`, y en Python el paquete `pyGAM`. En este último, el valor predeterminado de `n_splines` (20) puede producir sobreajuste, por lo que a menudo se elige un valor menor.
### Ejemplo práctico en Python
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

np.random.seed(42)
n = 300

datos = pd.DataFrame({
    'metros_cuadrados': np.random.uniform(50, 400, n),
    'banios': np.random.randint(1, 4, n)
})

# Relación no lineal: el precio crece rápido al inicio y luego se desacelera
datos['precio_venta'] = (
    100000
    + 9000 * np.sqrt(datos['metros_cuadrados'] - 40) * 10
    + 10000 * datos['banios']
    + np.random.normal(0, 25000, n)
)

# 1) Modelo lineal (referencia)
m_lineal = smf.ols('precio_venta ~ metros_cuadrados + banios', data=datos).fit()

# 2) Regresión polinómica (término cuadrático)
m_poly = smf.ols('precio_venta ~ metros_cuadrados + I(metros_cuadrados**2) + banios',
                 data=datos).fit()

# 3) Spline cúbico con 6 grados de libertad (3 nudos internos)
m_spline = smf.ols('precio_venta ~ bs(metros_cuadrados, df=6, degree=3) + banios',
                   data=datos).fit()

print(f'R² lineal:     {m_lineal.rsquared:.4f}')
print(f'R² polinómico: {m_poly.rsquared:.4f}')
print(f'R² spline:     {m_spline.rsquared:.4f}')

# Comparación visual (se fija banios en su valor típico)
malla = pd.DataFrame({
    'metros_cuadrados': np.linspace(50, 400, 200),
    'banios': 2
})

plt.figure(figsize=(7, 5))
plt.scatter(datos['metros_cuadrados'], datos['precio_venta'], alpha=0.3, label='Datos')
plt.plot(malla['metros_cuadrados'], m_lineal.predict(malla), label='Lineal')
plt.plot(malla['metros_cuadrados'], m_poly.predict(malla), label='Polinómico (grado 2)')
plt.plot(malla['metros_cuadrados'], m_spline.predict(malla), label='Spline cúbico')
plt.xlabel('Metros cuadrados')
plt.ylabel('Precio de venta')
plt.legend()
plt.show()
```
Con `pyGAM` (requiere `pip install pygam`), el ajuste automático de un GAM se realiza así:
```python
from pygam import LinearGAM, s, l

X = datos[['metros_cuadrados', 'banios']].values
y = datos['precio_venta']

# s(0): término spline para metros_cuadrados | l(1): término lineal para baños
gam = LinearGAM(s(0, n_splines=12) + l(1))
gam.gridsearch(X, y)
gam.summary()
```

---

# Conclusiones
- La regresión lineal simple y múltiple permite modelar la relación entre una variable respuesta y uno o más predictores. Los coeficientes se estiman por mínimos cuadrados, método eficiente pero sensible a los valores atípicos.
- La regresión sirve tanto para **explicar** relaciones como para **predecir**. Sin embargo, una ecuación de regresión por sí sola no demuestra causalidad; esa conclusión debe apoyarse en el conocimiento del fenómeno.
- Evaluar un modelo solo con métricas *in-sample* ($R^2$, valores $p$) puede ser engañoso. Por eso se recomienda usar métricas como el RMSE sobre datos de reserva o mediante validación cruzada, y criterios que penalicen la complejidad (AIC, $R^2$ ajustado) para seleccionar variables.
- Al predecir hay que evitar la extrapolación fuera del rango de los datos y cuantificar la incertidumbre con intervalos de predicción, que son más amplios que los de confianza porque incluyen la variabilidad de cada observación individual.
- Las variables categóricas deben codificarse con $P-1$ variables dummy para evitar la multicolinealidad, y los factores con muchos niveles pueden consolidarse, por ejemplo, a partir de los residuos de un modelo inicial.
- Los coeficientes deben interpretarse con cuidado: la correlación entre predictores, la multicolinealidad y la omisión de variables de confusión pueden alterar su signo y magnitud. Las interacciones permiten capturar que el efecto de un predictor depende de otro.
- Los diagnósticos (residuos estandarizados, valores hat, distancia de Cook, gráficos de heterocedasticidad y de residuos parciales) ayudan a detectar datos anómalos, observaciones influyentes y no linealidad. En ciencia de datos prima la precisión predictiva sobre el cumplimiento estricto de los supuestos distribucionales.
- Cuando la relación no es lineal, los polinomios, los splines y los GAM ofrecen alternativas flexibles. Un mejor ajuste visual no garantiza un mejor modelo si carece de sentido en el contexto del problema.
- En conjunto, la regresión es un conjunto de herramientas que permite tanto describir relaciones entre variables como construir modelos predictivos fundamentados, siempre que se evalúen y diagnostiquen de forma rigurosa.
# Referencias Bibliográficas
Bruce, P., Bruce, A., & Gedeck, P. (2020). *Practical statistics for data scientists: 50+ essential concepts using R and Python* (2nd ed.). O'Reilly Media.

Diez, D., Barr, C., & Çetinkaya-Rundel, M. (2019). *OpenIntro Statistics (4th ed.).* https://www.openintro.org/book/os/

Illowsky, B., & Dean, S. (2023). *Introductory statistics 2e. OpenStax.* https://openstax.org/books/introductory-statistics-2e/

Kosourova, E. (2025, 19 de junio). *Explicación del RMSE: Guía para la precisión de la predicción de regresión.* DataCamp. https://www.datacamp.com/es/tutorial/rmse
