#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

# Importación de librerías necesarias para el proyecto
# Librerías de scikit-learn para procesamiento de datos y modelos de machine learning
from sklearn.pipeline import Pipeline  # Para crear pipelines de procesamiento y modelado
from sklearn.compose import ColumnTransformer  # Para aplicar diferentes transformaciones a diferentes columnas
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler  # Para codificación one-hot y escalado de datos
from sklearn.feature_selection import SelectKBest, f_regression  # Para selección de características más relevantes
from sklearn.model_selection import GridSearchCV  # Para búsqueda de hiperparámetros con validación cruzada
from sklearn.linear_model import LinearRegression  # Modelo de regresión lineal
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error  # Métricas de evaluación

# Librerías estándar de Python
import pickle  # Para serialización de objetos Python
import zipfile  # Para manejo de archivos comprimidos ZIP
import gzip  # Para compresión de archivos
import json  # Para manejo de archivos JSON
import os  # Para operaciones del sistema operativo
import pandas as pd  # Para manipulación y análisis de datos


def limpiar_datos(df):
    """
    Función para preprocesar los datos del dataset de vehículos usados.
    
    Esta función realiza las siguientes operaciones:
    1. Crea una copia del DataFrame para evitar modificar el original
    2. Calcula la edad del vehículo basándose en el año de fabricación
    3. Elimina columnas innecesarias (Year y Car_Name)
    4. Elimina filas with valores faltantes
    
    Parámetros:
    df (DataFrame): DataFrame con los datos originales del vehículo
    
    Retorna:
    DataFrame: DataFrame limpio y procesado
    """
    # Crear una copia del DataFrame para evitar modificar el original
    df = df.copy()
    
    # Calcular la edad del vehículo: año actual (2021) - año de fabricación
    # Esto convierte el año en una característica más útil (edad)
    df['Age'] = 2021 - df['Year']
    
    # Eliminar columnas que no son útiles para el modelo:
    # - 'Year': ya no es necesaria porque se convirtió en 'Age'
    # - 'Car_Name': es texto descriptivo que no aporta valor predictivo
    df = df.drop(columns=['Year', 'Car_Name'])
    
    # Eliminar filas con valores faltantes (NaN) para evitar errores en el entrenamiento
    df = df.dropna()

    return df

def modelo():
    """
    Función que crea el pipeline de machine learning para la predicción de precios.
    
    El pipeline incluye los siguientes pasos:
    1. Preprocesamiento de variables categóricas (One-Hot Encoding)
    2. Escalado de variables numéricas (Min-Max Scaling)
    3. Selección de las mejores características (SelectKBest)
    4. Modelo de regresión lineal
    
    Retorna:
    Pipeline: Pipeline completo listo para entrenamiento
    """
    
    # Definir las columnas categóricas que necesitan codificación one-hot
    # Estas variables contienen categorías (texto) que deben convertirse a números
    categoricas = ["Fuel_Type", "Selling_type", "Transmission"]  
    
    # Definir las columnas numéricas que necesitan escalado
    # Estas variables ya son numéricas pero tienen diferentes rangos
    numericas = [
        "Selling_Price", "Driven_kms", "Age", "Owner"
    ]

    # Crear el preprocesador que aplica transformaciones diferentes a diferentes tipos de columnas
    preprocesador = ColumnTransformer(
        transformers=[
            # Para variables categóricas: aplicar One-Hot Encoding
            # Esto convierte categorías como 'Petrol', 'Diesel' en columnas binarias (0 o 1)
            ('cat', OneHotEncoder(handle_unknown='ignore'), categoricas),
            
            # Para variables numéricas: aplicar escalado Min-Max al rango [0,1]
            # Esto normaliza todas las variables numéricas al mismo rango
            ('scaler', MinMaxScaler(), numericas)
        ],
        remainder='passthrough'  # Mantener otras columnas sin cambios
    )

    # Crear el selector de características que elegirá las K mejores variables
    # Usa f_regression para evaluar la importancia de cada característica
    seleccionar_mejores = SelectKBest(score_func=f_regression)

    # Construir el pipeline completo con todos los pasos en secuencia
    pipeline = Pipeline(steps=[
        # Paso 1: Preprocesar los datos (codificación y escalado)
        ('preprocesador', preprocesador),
        
        # Paso 2: Seleccionar las mejores características
        ("seleccionar_mejores", seleccionar_mejores),
        
        # Paso 3: Aplicar el modelo de regresión lineal
        ('clasificador', LinearRegression())
    ])

    return pipeline

def hiperparametros(modelo, n_divisiones, x_entrenamiento, y_entrenamiento, puntuacion):
    """
    Función para optimizar los hiperparámetros del modelo usando validación cruzada.
    
    Esta función usa GridSearchCV para encontrar el mejor valor del parámetro 'k'
    (número de mejores características a seleccionar) que maximice el rendimiento del modelo.
    
    Parámetros:
    modelo (Pipeline): Pipeline del modelo a optimizar
    n_divisiones (int): Número de divisiones para la validación cruzada
    x_entrenamiento (DataFrame): Variables independientes de entrenamiento
    y_entrenamiento (Series): Variable dependiente de entrenamiento
    puntuacion (str): Métrica de evaluación para la optimización
    
    Retorna:
    GridSearchCV: Modelo optimizado con los mejores hiperparámetros
    """
    
    # Crear el objeto GridSearchCV para búsqueda de hiperparámetros
    estimador = GridSearchCV(
        estimator=modelo,  # El pipeline que queremos optimizar
        
        # Definir el rango de hiperparámetros a probar
        # Probamos valores de k desde 1 hasta 12 características
        param_grid = {
            "seleccionar_mejores__k": range(1, 13),  # '__' conecta el nombre del paso con el parámetro
        },
        
        cv=n_divisiones,  # Número de divisiones para validación cruzada (10 en este caso)
        refit=True,  # Reentrenar el modelo con los mejores parámetros en todo el dataset
        scoring=puntuacion  # Métrica de evaluación (error medio absoluto negativo)
    )
    
    # Entrenar el modelo probando todos los valores de k y seleccionar el mejor
    # Esto puede tomar tiempo porque prueba 12 valores diferentes con validación cruzada
    estimador.fit(x_entrenamiento, y_entrenamiento)

    return estimador

def metricas(modelo, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba):
    """
    Función para calcular métricas de evaluación del modelo en conjuntos de entrenamiento y prueba.
    
    Calcula tres métricas importantes:
    - R² (coeficiente de determinación): mide qué tan bien el modelo explica la variabilidad
    - MSE (error cuadrático medio): penaliza más los errores grandes
    - MAD (error absoluto mediano): métrica robusta menos sensible a valores atípicos
    
    Parámetros:
    modelo (GridSearchCV): Modelo entrenado y optimizado
    x_entrenamiento (DataFrame): Variables independientes de entrenamiento
    y_entrenamiento (Series): Variable dependiente de entrenamiento
    x_prueba (DataFrame): Variables independientes de prueba
    y_prueba (Series): Variable dependiente de prueba
    
    Retorna:
    tuple: (métricas_entrenamiento, métricas_prueba) - diccionarios con las métricas
    """

    # Realizar predicciones en el conjunto de entrenamiento
    # Esto nos dice qué tan bien el modelo se ajusta a los datos que ya conoce
    y_entrenamiento_pred = modelo.predict(x_entrenamiento)
    
    # Realizar predicciones en el conjunto de prueba
    # Esto nos dice qué tan bien el modelo generaliza a datos nuevos
    y_prueba_pred = modelo.predict(x_prueba)

    # Calcular métricas para el conjunto de entrenamiento
    metricas_entrenamiento = {
        'type': 'metrics',  # Identificador del tipo de registro
        'dataset': 'train',  # Especifica que son métricas de entrenamiento
        
        # R²: valor entre 0 y 1, donde 1 es perfecto
        'r2': r2_score(y_entrenamiento, y_entrenamiento_pred),
        
        # MSE: error cuadrático medio, menor es mejor
        'mse': mean_squared_error(y_entrenamiento, y_entrenamiento_pred),
        
        # MAD: error absoluto mediano, menor es mejor
        'mad': median_absolute_error(y_entrenamiento, y_entrenamiento_pred)
    }

    # Calcular métricas para el conjunto de prueba
    metricas_prueba = {
        'type': 'metrics',  # Identificador del tipo de registro
        'dataset': 'test',  # Especifica que son métricas de prueba
        
        # R²: si es mucho menor que en entrenamiento, puede haber sobreajuste
        'r2': r2_score(y_prueba, y_prueba_pred),
        
        # MSE: si es mucho mayor que en entrenamiento, hay sobreajuste
        'mse': mean_squared_error(y_prueba, y_prueba_pred),
        
        # MAD: métrica más robusta para evaluar el rendimiento general
        'mad': median_absolute_error(y_prueba, y_prueba_pred)
    }

    return metricas_entrenamiento, metricas_prueba

def guardar_modelo(modelo):
    """
    Función para guardar el modelo entrenado en un archivo comprimido.
    
    El modelo se guarda usando pickle (serialización) y se comprime con gzip
    para reducir el tamaño del archivo. Esto permite reutilizar el modelo
    posteriormente sin necesidad de reentrenarlo.
    
    Parámetros:
    modelo (GridSearchCV): Modelo entrenado y optimizado que se va a guardar
    """
    
    # Crear el directorio 'files/models' si no existe
    # exist_ok=True evita errores si el directorio ya existe
    os.makedirs('files/models', exist_ok=True)

    # Guardar el modelo en un archivo comprimido (.gz)
    # 'wb' significa escribir en modo binario
    with gzip.open('files/models/model.pkl.gz', 'wb') as f:
        # pickle.dump serializa el objeto modelo y lo guarda en el archivo
        # Esto preserva toda la información del modelo incluyendo hiperparámetros optimizados
        pickle.dump(modelo, f)

def guardar_metricas(metricas):
    """
    Función para guardar las métricas de evaluación en un archivo JSON.
    
    Las métricas se guardan línea por línea en formato JSON, donde cada línea
    contiene un diccionario con las métricas de un conjunto de datos
    (entrenamiento o prueba).
    
    Parámetros:
    metricas (list): Lista de diccionarios con las métricas a guardar
    """
    
    # Crear el directorio 'files/output' si no existe
    os.makedirs('files/output', exist_ok=True)

    # Abrir el archivo en modo escritura de texto
    with open("files/output/metrics.json", "w") as f:
        # Iterar sobre cada diccionario de métricas
        for metrica in metricas:
            # Convertir el diccionario a formato JSON (string)
            json_line = json.dumps(metrica)
            # Escribir la línea JSON al archivo con salto de línea
            # Esto crea un archivo con formato JSON Lines (cada línea es un JSON válido)
            f.write(json_line + "\n")

# === CARGA DE DATOS ===
# Los datos están almacenados en archivos ZIP que necesitan ser extraídos y leídos

# Cargar el conjunto de datos de prueba
# Abrir el archivo ZIP que contiene los datos de prueba
with zipfile.ZipFile('files/input/test_data.csv.zip', 'r') as zip:
    # Extraer y leer el archivo CSV desde dentro del ZIP
    with zip.open('test_data.csv') as f:
        # Leer el CSV y crear un DataFrame de pandas
        df_Prueba = pd.read_csv(f)

# Cargar el conjunto de datos de entrenamiento
# Mismo proceso que arriba pero para los datos de entrenamiento
with zipfile.ZipFile('files/input/train_data.csv.zip', 'r') as zip:
    with zip.open('train_data.csv') as f:
        df_Entrenamiento = pd.read_csv(f)


# === EJECUCIÓN PRINCIPAL DEL PROGRAMA ===
# Este bloque se ejecuta solo cuando el archivo se ejecuta directamente (no cuando se importa)

if __name__ == '__main__':
    # PASO 1: PREPROCESAMIENTO DE DATOS
    # Limpiar y preparar los datos para el entrenamiento
    print("Limpiando datos...")
    df_Prueba = limpiar_datos(df_Prueba)
    df_Entrenamiento = limpiar_datos(df_Entrenamiento)

    # PASO 2: DIVISIÓN DE DATOS EN CARACTERÍSTICAS (X) Y VARIABLE OBJETIVO (Y)
    # Separar las variables independientes (características) de la variable dependiente (precio)
    
    # Para el conjunto de entrenamiento:
    # x_entrenamiento: todas las columnas excepto 'Present_Price' (lo que usamos para predecir)
    # y_entrenamiento: solo la columna 'Present_Price' (lo que queremos predecir)
    x_entrenamiento, y_entrenamiento = df_Entrenamiento.drop('Present_Price', axis=1), df_Entrenamiento['Present_Price']
    
    # Para el conjunto de prueba: mismo proceso
    x_prueba, y_prueba = df_Prueba.drop('Present_Price', axis=1), df_Prueba['Present_Price']

    # PASO 3: CREACIÓN DEL MODELO
    # Crear el pipeline de machine learning con todos los pasos de procesamiento
    print("Protegeme señor con tu espiritu : Creando modelo...")
    pipeline_modelo = modelo()

    # PASO 4: OPTIMIZACIÓN DE HIPERPARÁMETROS
    # Usar validación cruzada para encontrar el mejor valor de 'k' (número de características)
    # Se usan 10 divisiones (folds) y se optimiza el error medio absoluto
    print("Ya casi : Optimizando hiperparámetros...")
    pipeline_modelo = hiperparametros(pipeline_modelo, 10, x_entrenamiento, y_entrenamiento, 'neg_mean_absolute_error')

    # PASO 5: GUARDAR EL MODELO ENTRENADO
    # Guardar el modelo optimizado para uso futuro
    print("Mk Ya : Guardando modelo...")
    guardar_modelo(pipeline_modelo)

    # PASO 6: EVALUACIÓN DEL MODELO
    # Calcular métricas de rendimiento en los conjuntos de entrenamiento y prueba
    print("Calculando métricas...")
    metricas_entrenamiento, metricas_prueba = metricas(pipeline_modelo, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba)

    # PASO 7: GUARDAR MÉTRICAS
    # Guardar las métricas en un archivo JSON para análisis posterior
    print("Guardando métricas...")
    guardar_metricas([metricas_entrenamiento, metricas_prueba])
    
    print("Somos unos lokitos!")