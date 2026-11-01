Moneyball

Hola quiero realizar un proyecto basado en la estutura de probabilidades y estadisticas, matemáticas e informática, basado en la liga de béisbol del 2002 utilizada en la pelicula Moneyball. Me puedes dar un contexto completo dentro de un prompt para recrear el sistema aplicado, supuestamente, por los Oaklands Atletics, para todos los equipos, y un desgloce tal cual lo muestra en la pelicula, con los datos de cada fecha , y como la muestra de datos comienza a ser mas sólida. Asimismo, compartir la estructura de código y demás arquitectura que pueda recrear la utilizada para poder conocer, en números las métricas y cómo los números cuentan una historia.

---------------------------------

La filosofía central de "Moneyball" no trata de comprar jugadores, sino de comprar victorias a través de la producción de carreras, y para fabricar carreras, el jugador necesita evitar el out. El sistema de los Oakland Athletics de 2002, fundamentado en la sabermetría de Bill James, es en el fondo un modelo de arbitraje financiero aplicado al deporte: explotar las ineficiencias del mercado descartando las métricas sobrevaloradas por los ojeadores tradicionales (promedio de bateo, bases robadas, apariencia atlética) a favor de datos duros que tienen una correlación matemática directa con las victorias.

## El Prompt de Contexto Base

Para recrear este entorno en un modelo de lenguaje o como documento fundacional de tu proyecto, utiliza este prompt general que establece las reglas del sistema para todos los equipos:

> **Rol y Objetivo:** Eres el Gerente General y Director de Análisis Cuantitativo de un equipo de la MLB en 2002. Tu mandato es implementar un sistema de evaluación sabermétrico para los 30 equipos, identificando ineficiencias del mercado en la asignación de salarios.
> **Reglas del Sistema:**
> 1. Ignora la evaluación cualitativa de los ojeadores tradicionales (edad, "swing bonito", velocidad pura, lesiones pasadas que no afecten mecánicas actuales).
> 2. El objetivo único de la ofensiva es no hacer out. El recurso más escaso en el béisbol son los 27 outs por juego.
> 3. Construye modelos de regresión para proyectar Carreras Creadas (RC). Debes estructurar un equipo que matemáticamente asegure 814 carreras en la temporada, lo que estadísticamente equivale a 95 victorias y la clasificación a playoffs.
> 4. Evalúa a los jugadores estrictamente por su Porcentaje de Embasado (OBP), Slugging (SLG) y OPS.
> 
> 
> **Salida Esperada:** Para cada análisis, desglosa el valor del jugador en "Costo por Carrera Creada" frente a su salario, explicando cómo la varianza estadística dicta que su rendimiento a largo plazo corregirá cualquier mala racha a corto plazo.

## La Cronología de Datos: La Ley de los Grandes Números

En la película y en la vida real, el sistema es cuestionado a principios de temporada porque los datos necesitan tiempo para consolidarse. El béisbol es un juego de alta varianza diaria, pero de extrema precisión a largo plazo (162 juegos).

| Mes | Juegos | Estado de la Muestra (Apariciones al Plato - PA) | Lo que cuentan los números |
| --- | --- | --- | --- |
| **Abril** | 1 - 25 | **Ruido Estadístico (< 100 PA):** Alta volatilidad. | Un jugador malo puede batear .400 por pura suerte, y uno bueno puede batear .150. El modelo ignora los resultados y confía en el pronóstico histórico. |
| **Mayo - Jun** | 26 - 75 | **Estabilización de Disciplina (120 - 150 PA):** | Las tasas de Ponches (K%) y Bases por Bolas (BB%) convergen. El sistema ya sabe si un jugador tiene paciencia en el plato, independientemente de si hace contacto. |
| **Julio** | 76 - 100 | **El Mercado de Arbitraje (300 PA):** | El Slugging (poder real aislado de la suerte) se estabiliza. Aquí Oakland buscaba intercambios: compraban jugadores con alto OBP que estaban en "malas rachas" de suerte, pagando centavos por ellos. |
| **Ago - Oct** | 101 - 162 | **Convergencia Final (460+ PA):** | El Porcentaje de Embasado (OBP) alcanza su valor predictivo real. La varianza desaparece. Las carreras creadas colectivas del equipo coinciden matemáticamente con el total de victorias predicho. |

## Las Métricas Clave y su Historia

El sistema descartó el Promedio de Bateo (AVG) porque asume incorrectamente que todos los hits valen lo mismo y que una base por bolas no tiene valor.

* **OBP (On-Base Percentage):** Cuenta la historia de la supervivencia. Responde a la pregunta: *¿Qué porcentaje de las veces este jugador se niega a regalarle un out al oponente?* Incluye hits, bases por bolas y golpes.
* **SLG (Slugging Percentage):** Cuenta la historia del daño. Mide el número de bases totales que un jugador alcanza por turno al bate.
* **Expectativa Pitagórica de Victorias:** La fórmula matemática que Oakland usaba para saber cuántas victorias merecían realmente, calculada como `(Carreras Anotadas^2) / (Carreras Anotadas^2 + Carreras Permitidas^2)`. Si el equipo anotaba más carreras de las permitidas pero perdía juegos, el modelo sabía que era simple mala suerte de secuencia y que, eventualmente, el equipo empezaría a ganar.

## Arquitectura y Estructura de Código

Para recrear esto, necesitarás un flujo de datos (ETL) que procese bases de datos históricas como *Retrosheet* o la *Lahman Baseball Database*, y un entorno analítico en Python.

**Arquitectura recomendada:**

1. **Ingesta de Datos:** SQL o Pandas para unir tablas de `Salaries` y `Batting`.
2. **Motor Analítico:** Numpy/Pandas para calcular métricas de Sabermetría.
3. **Modelo Predictivo:** Scikit-learn (Regresión Lineal Múltiple) para predecir Carreras del Equipo basándose en el OBP y SLG colectivo.
4. **Visualización:** Streamlit o Dash para construir el "Dashboard del Gerente General".

**Estructura base del código (Python):**

```python
import pandas as pd
import numpy as np

# 1. Carga de datos de bateo y salarios del 2002
batting = pd.read_csv('Batting.csv')
salaries = pd.read_csv('Salaries.csv')

# Filtrar para el año 2002
batting_2002 = batting[batting['yearID'] == 2002].copy()
salaries_2002 = salaries[salaries['yearID'] == 2002]

# 2. Calcular los componentes de Moneyball
# Sencillos (1B) = Hits totales menos extrabases
batting_2002['1B'] = batting_2002['H'] - batting_2002['2B'] - batting_2002['3B'] - batting_2002['HR']

# OBP = (Hits + Bases por Bolas + Golpes) / (Turnos + Bases por Bolas + Golpes + Elevados de Sacrificio)
batting_2002['OBP'] = (batting_2002['H'] + batting_2002['BB'] + batting_2002['HBP']) / \
                      (batting_2002['AB'] + batting_2002['BB'] + batting_2002['HBP'] + batting_2002['SF'])

# SLG = Bases Totales / Turnos al Bate
batting_2002['SLG'] = (batting_2002['1B'] + 2*batting_2002['2B'] + 3*batting_2002['3B'] + 4*batting_2002['HR']) / batting_2002['AB']
batting_2002['OPS'] = batting_2002['OBP'] + batting_2002['SLG']

# 3. Fórmula de Carreras Creadas (Bill James)
batting_2002['Runs_Created'] = ((batting_2002['H'] + batting_2002['BB']) * \
                               (batting_2002['1B'] + 2*batting_2002['2B'] + 3*batting_2002['3B'] + 4*batting_2002['HR'])) / \
                               (batting_2002['AB'] + batting_2002['BB'])

# 4. Unir con Salarios para buscar Ineficiencias de Mercado
stats_with_salary = pd.merge(batting_2002, salaries_2002, on=['playerID', 'yearID'])

# Filtrar jugadores subvalorados: Muestra sólida (> 300 Turnos), Alto OBP (> 0.350), Salario bajo (< $1.5M)
undervalued_targets = stats_with_salary[
    (stats_with_salary['AB'] > 300) & 
    (stats_with_salary['OBP'] > 0.350) & 
    (stats_with_salary['salary'] < 1500000)
].sort_values(by='Runs_Created', ascending=False)

print(undervalued_targets[['playerID', 'OBP', 'SLG', 'Runs_Created', 'salary']].head(10))

```

-------------------------------------

 quiero que me guíes para obtener e integrar la base de datos real Lahman para ejecutar este modelo! Tambien  quiero desarrollar una app basada en datos, como temática "Moneyball" , y que sea flexibles para tomar datos de forma gratuita y poder ser aplicada a toda la linea del tiempo.
Favor adjuntar recursos, tales como repos, bugs, y soluciones a esos posibles bugs.

-----------------------------------

Para construir una aplicación escalable de sabermetría que consuma datos gratuitos de toda la historia del béisbol, necesitas una arquitectura basada en extracción continua y normalización de bases de datos de código abierto.

### Fuentes de Datos y Recursos Principales

El estándar de la industria de código abierto para esto se divide en bases históricas consolidadas y librerías de extracción (scraping) para datos modernos.

1. **Librería Principal (Python): `pybaseball**`
* **Repositorio:** [github.com/jldbc/pybaseball](https://github.com/jldbc/pybaseball)
* **Función:** Es la herramienta definitiva. Extrae datos a nivel de lanzamiento (Statcast), estadísticas de temporada (Baseball Reference, FanGraphs) y permite descargar la base de datos histórica completa de Lahman con un solo comando `pybaseball.lahman.download_lahman()`.


2. **Base de Datos Histórica: The Lahman Baseball Database**
* **Recurso:** Disponible en formato CSV o SQLite (seanlahman.com). Contiene tablas completas de pitcheo, bateo y fildeo desde 1871 hasta la temporada pasada.
* **Librería alternativa:** `pylahman` (disponible vía `pip install pylahman`), un wrapper directo a pandas.


3. **Mapeo de Entidades: Chadwick Baseball Bureau Register**
* **Repositorio:** [github.com/chadwickbureau/register](https://github.com/chadwickbureau/register)
* **Función:** El registro maestro para vincular a los jugadores. Cada proveedor (MLB, Yahoo, FanGraphs, Retrosheet, Lahman) usa un ID diferente para el mismo jugador. Este repositorio es el diccionario de traducción.



---

### Bugs Frecuentes en el Desarrollo y Soluciones

Al procesar la línea de tiempo completa del béisbol con Pandas o SQL, te encontrarás con fallas matemáticas y de integridad de datos.

| Bug / Problema | Causa Raíz | Solución Estructural |
| --- | --- | --- |
| **`ZeroDivisionError` o `NaN` en métricas de ratio** | Jugadores (usualmente lanzadores de relevo o novatos) con 0 Turnos al Bate (AB) o 0 Apariciones al Plato (PA). Al calcular SLG o AVG, divides entre cero. | Aplicar filtros de muestra estandarizados antes del cálculo (`df = df[df['AB'] > 0]`) o usar `np.where(df['AB'] > 0, formula, 0)`. Para modelos predictivos, exigir un mínimo de 130 AB (estándar MLB para novatos). |
| **Fallo en la fórmula OBP histórica** | Los Elevados de Sacrificio (SF) no se registraron oficialmente como una estadística separada sino hasta 1954. Las fórmulas modernas de OBP fallan al intentar sumar `df['SF']` en años previos porque la columna es nula. | Rellenar valores nulos basándose en el año cronológico: `df['SF'] = df['SF'].fillna(0)`. |
| **Pérdida de datos en cruces (Joins)** | Intentar hacer un `MERGE` entre estadísticas de Statcast (datos modernos) y Lahman (históricos) usando nombres completos en lugar de IDs. "Ken Griffey Jr." vs "Ken Griffey". | **Nunca usar strings de nombres como llave primaria.** Usar siempre la base de datos de Chadwick para convertir el `key_mlbam` (ID moderno) a `key_retro` o `key_lahman` antes de cruzar los DataFrames. |
| **Duplicidad de registros por temporada** | Un jugador es traspasado a mitad de año. Lahman crea una fila para su tiempo en el Equipo A y otra para el Equipo B en el mismo `yearID`. Al calcular líderes, los números se diluyen. | Agrupar por jugador y año antes de calcular métricas finales: `df.groupby(['playerID', 'yearID']).sum()`. |

### Estructura de Arquitectura Recomendada

Para la aplicación, separa el motor de datos del frontend.

* **Backend / ETL (Python):** Usa `pybaseball` y `pandas` para descargar y limpiar los datos. Tira los DataFrames procesados a una base de datos local SQLite o PostgreSQL (si usas pgvector en tus despliegues).
* **API (Opcional):** Construye endpoints con `FastAPI` si planeas conectar múltiples interfaces.
* **Frontend (Python):** `Streamlit` o `Dash`. Permiten construir dashboards interactivos con deslizadores de tiempo (para moverte de 1900 a 2023) y gráficos de dispersión (Scatter plots) relacionando Salario vs. OPS.

¿Iniciamos con el código backend?