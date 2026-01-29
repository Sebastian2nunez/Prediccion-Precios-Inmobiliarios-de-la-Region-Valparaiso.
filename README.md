# 🏠 Predictor de Precios Inmobiliarios - Región Valparaíso

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![ML](https://img.shields.io/badge/ML-Gradient%20Boosting-green.svg)
![Status](https://img.shields.io/badge/Status-Completo-success.svg)

> Sistema end-to-end de Machine Learning que predice precios de propiedades inmobiliarias utilizando Gradient Boosting, alcanzando R² = 0.67 en datos reales de la Región de Valparaíso.

## 📊 Resultados Clave

- ✅ **961 propiedades** analizadas (Región de Valparaíso)
- ✅ **R² = 0.67** en conjunto de prueba
- ✅ **MAE = 2,696 UF** (error promedio absoluto)
- ✅ **6 algoritmos** comparados (Linear Regression, Ridge, Lasso, Random Forest, XGBoost, Gradient Boosting)
- ✅ **Optimización de hiperparámetros** con RandomizedSearchCV (40 combinaciones)

## 🎯 Problema & Solución

**Problema:**  
El mercado inmobiliario de la Región de Valparaíso carece de herramientas de valoración objetiva, dificultando a compradores e inversionistas identificar oportunidades.

**Solución:**  
Sistema automatizado que:
1. Recolecta datos de propiedades mediante web scraping
2. Realiza análisis exploratorio exhaustivo (EDA)
3. Limpia y procesa datos con transformaciones específicas del dominio
4. Entrena y compara múltiples modelos de Machine Learning
5. Predice valores de mercado con precisión del 67%

## 🛠️ Stack Técnico

**Data Collection:**
- Python 3.9+
- BeautifulSoup4
- Requests
- Selenium

**Data Processing & Analysis:**
- Pandas
- NumPy
- Scikit-learn

**Machine Learning:**
- Gradient Boosting Regressor (modelo final)
- Random Forest, XGBoost
- Linear Regression, Ridge, Lasso
- RandomizedSearchCV para optimización
- Cross-validation (5-fold)

**Visualization:**
- Matplotlib
- Seaborn

**Tools:**
- Jupyter Notebook
- Git
- Joblib

## 📁 Estructura del Proyecto

```
├── notebooks/
│   ├── 01_Scraping_icasas.ipynb        # Web scraping de icasas.cl
│   ├── 02_EDA_pre_limpieza.ipynb       # Análisis exploratorio (51 celdas)
│   ├── 03_Limpieza.ipynb               # Limpieza y transformación de datos
│   └── 04_ML.ipynb                     # Modelado, evaluación y optimización
├── data/
│   ├── raw/                            # Datos scrapeados (995 propiedades)
│   └── processed/                      # Datos limpios (961 propiedades)
├── models/
│   └── modelo_final_valparaiso.pkl     # Gradient Boosting optimizado
├── requirements.txt
└── README.md
```

## 🚀 Instalación y Uso

```bash
# Clonar repositorio
git clone https://github.com/[tu-usuario]/predictor-precios-valparaiso
cd predictor-precios-valparaiso

# Crear ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar notebooks en orden
jupyter notebook
```

## 📈 Metodología

### 1. Recolección de Datos (Web Scraping)
- Fuente: [icasas.cl](https://www.icasas.cl)
- Respeto a robots.txt
- 995 propiedades iniciales
- Variables: ubicación, precio, área, características, amenities

### 2. Análisis Exploratorio (EDA)
- **51 celdas** de análisis detallado
- Identificación de patrones por comuna
- Análisis de distribuciones de precios
- Detección de valores atípicos
- Correlaciones entre variables

### 3. Limpieza y Preprocesamiento
- **Corrección de separadores decimales:** Conversión de formato europeo a estándar
- **Normalización de áreas:** Consolidación de `Area m2`, `Área útil`, `m2 terreno` en:
  - `Area Construida` (máximo entre las tres)
  - `Area Terreno` (mínimo entre las tres)
- **Manejo de Año de Construcción:** Corrección de valores incoherentes (10-20.3 → 2010-2020.3)
- **Flags categóricos:** Creación de variables `Casa` y `Terreno` basadas en lógica de negocio
- **Eliminación de duplicados:** Reducción de 995 → 961 propiedades únicas
- **Limpieza de amenities:** Conversión de columnas binarias (True/False)

### 4. Feature Engineering
- Variables derivadas: edad del inmueble, ratios, densidad
- Encoding de variables categóricas (OneHotEncoder)
- Manejo de valores faltantes (imputación con medianas)
- Split estratificado por rangos de precio para balancear distribuciones

### 5. Modelado y Evaluación

**Modelos probados:**

| Modelo | CV R² (5-fold) | Test R² | Test MAE (UF) |
|--------|----------------|---------|---------------|
| Linear Regression | 0.46 | 0.51 | 3417 |
| Ridge | 0.46 | 0.52 | 3404 |
| Lasso | 0.46 | 0.51 | 3418 |
| Random Forest | 0.58 | 0.64 | 2786 |
| XGBoost | 0.52 | 0.61 | 2856 |
| **Gradient Boosting** | **0.62** | **0.66** | **2696** |

**Modelo Final: Gradient Boosting Optimizado**
- Hiperparámetros optimizados con RandomizedSearchCV
- 40 combinaciones evaluadas
- Configuración óptima:
  - `n_estimators`: 700
  - `max_depth`: 4
  - `learning_rate`: 0.03
  - `subsample`: 0.8
  - `max_features`: 'sqrt'
  - `min_samples_split`: 5

**Validación:**
- Cross-validation 5-fold: R² = 0.62
- Split estratificado train/test (80/20)
- Test R² = 0.67 (modelo generaliza correctamente)

## 📊 Interpretación de Resultados

**R² = 0.67** significa que el modelo explica el **67% de la variabilidad** en los precios de propiedades.

**MAE = 2,696 UF** representa un error promedio de aproximadamente **$102 millones CLP** (considerando UF ≈ $38,000).

**Contexto:**
- El 33% de variabilidad no explicada corresponde a factores no capturados en el dataset (ubicación exacta, estado de conservación, vista, proximidad a servicios).
- Para un proyecto basado en web scraping sin datos premium (GPS, fotos, tasaciones profesionales), estos resultados son sólidos.

## 🎓 Aprendizajes Clave

1. **Importancia del preprocesamiento:** La corrección de separadores decimales y normalización de áreas fue crítica para la calidad del modelo.

2. **Split estratificado:** Implementar estratificación por rangos de precio eliminó el sesgo de distribución entre train/test.

3. **Gradient Boosting vs Random Forest:** GB superó a RF en este dataset, probablemente por la capacidad de GB de corregir errores iterativamente.

4. **Hyperparameter tuning:** La optimización mejoró el R² de 0.46 (baseline) a 0.67 (optimizado), un incremento del 46%.

5. **Feature engineering específico del dominio:** Variables como `Casa`/`Terreno` basadas en lógica inmobiliaria mejoraron el modelo.

## 🔮 Próximos Pasos

- [ ] Ampliar a otras regiones (Santiago, Concepción)
- [ ] Implementar sistema de detección de oportunidades (propiedades infravaloradas)
- [ ] Dashboard interactivo con Streamlit
- [ ] Automatización de scraping y reentrenamiento

## 📫 Contacto

**[Sebastián Núñez]**
- 📧 Email: [snunez445@gmail.com]
- 💼 LinkedIn: [www.linkedin.com/in/sebastián-mauricio-núñez-pérez-de-arce-98612534a]
---
