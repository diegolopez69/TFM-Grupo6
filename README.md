# TFM: Optimización del Proceso de Selección de Patentes

## Descripción del Proyecto

Este proyecto se centra en el análisis de datos de patentes farmacéuticas con el objetivo de identificar aquellas con mayor probabilidad de ser revocadas en el futuro. Utilizando técnicas de minería de datos, machine learning y análisis exploratorio de datos (EDA), se desarrollan modelos predictivos para optimizar el proceso de toma de decisiones en Sandoz.

## Estructura del Proyecto

El proyecto se divide en una serie de notebooks, cada uno con un propósito específico dentro del flujo de trabajo:

1. **1_Retrieve Application Numbers.ipynb**:
   - Obtención de los números de aplicación de las patentes a partir de una base de datos proporcionada.
   - Función principal: recolectar identificadores únicos para su posterior análisis.

2. **2_Bajar_HTMLS.ipynb**:
   - Descarga de documentos HTML de las patentes usando los números de aplicación obtenidos previamente.
   - Uso de técnicas de scraping para automatizar la obtención de datos.

3. **3_Webscrapping.ipynb**:
   - Procesamiento de los documentos HTML descargados para extraer información relevante.
   - Parsing y limpieza de datos, transformándolos en un formato adecuado para su análisis.

4. **4_Preprocesado.ipynb**:
   - Preprocesamiento de los datos extraídos, incluyendo la eliminación de duplicados y normalización.
   - Preparación de los datos para la fase de análisis exploratorio y modelado.

5. **5_EDA.ipynb**:
   - Análisis exploratorio de datos (EDA) para identificar patrones y relaciones entre las variables.
   - Visualización de distribuciones, correlaciones y tendencias en los datos.

6. **6_Modelos.ipynb**:
   - Desarrollo de modelos predictivos utilizando técnicas de clasificación y regresión.
   - Evaluación de modelos mediante métricas como AUC, ROC y matrices de confusión.

7. **8_llm.ipynb**:
   - Implementación de un modelo de lenguaje (LLM) para la consulta y análisis de datos textuales.
   - Uso de embeddings y modelos pre-entrenados para responder a preguntas específicas sobre las patentes.

8. **Modelo_Complejo.ipynb**:
   - Desarrollo de un modelo más complejo utilizando técnicas avanzadas como SHAP para interpretar la importancia de las variables.
   - Optimización y ajuste de hiperparámetros para mejorar la precisión y generalización del modelo.

## Instalación y Requerimientos

Para ejecutar el proyecto, asegúrate de tener instaladas las siguientes librerías:

- Python 3.8 o superior
- Jupyter Notebook
- Pandas
- Numpy
- Scikit-learn
- Matplotlib
- Seaborn
- SHAP
- Llama-index y otras dependencias (ver archivo `8_llm.ipynb` para detalles)

Puedes instalar todas las dependencias necesarias ejecutando el siguiente comando:

```bash
pip install -r requirements.txt
```

## Ejemplo de Ejecución

1. **Paso 1**: Clonar el repositorio y navegar al directorio del proyecto.
   ```bash
   git clone https://github.com/tuusuario/tfm-patentes.git
   cd tfm-patentes
   ```

2. **Paso 2**: Ejecutar cada notebook en el orden indicado para reproducir el flujo completo de análisis:
   - 1_Retrieve Application Numbers.ipynb
- 2_Bajar_HTMLS.ipynb
- 3_Webscrapping.ipynb
- 4_Preprocesado.ipynb
- 5_EDA.ipynb
- 6_Modelos.ipynb
- 8_llm.ipynb (opcional, para consultas avanzadas)
- Modelo_Complejo.ipynb

3. **Paso 3**: Verificar los resultados en cada etapa para asegurarse de que el flujo se ejecute correctamente.
