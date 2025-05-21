# ----- *** Cargar ficheros y modulos necesarios *** -----
import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chisquare, shapiro, ttest_ind
import altair as alt 
import plotly.express as px 
import plotly.figure_factory as ff
import plotly.graph_objects as go



# -------------------------------------------------------- Fin de carga de modulos

# ----- *** Metodos personalizados *** -----

# ----- *** Función para obtener información de las columnas *** -----
def info_columnas(df_filtrado, columnas, n=5):
    """
    Entrega información clave de columnas: tipo de dato, nulos, únicos y ejemplos.
    """
    start = time.time()
    registros = []

    for col in columnas:
        registros.append({
            'Columna': col,
            'Tipo de dato': str(df_filtrado[col].dtype),
            'Valores nulos': int(df_filtrado[col].isnull().sum()),
            'Valores únicos': int(df_filtrado[col].nunique()),
        })

    resultado = pd.DataFrame(registros)
    end = time.time()
    print(f"[pandas] Tiempo de análisis: {end - start:.4f} segundos")
    return resultado
# -------------------------------------------------------- Fin de función para obtener información de las columnas

# ----- *** Funcion para realizar la prueba Chi-cuadrado a una variable especifica para determinar si la distribucion es uniforme *** -----
def chi_cuadrado(df_filtrado, variable):
    """
        Definicion: Realiza la prueba Chi-cuadrado para evaluar si una variable categórica tiene una distribución uniforme.
        Parámetros:
        - df_filtrado: DataFrame de pandas.
        - variable: nombre de la variable a analizar.
        Devuelve el estadístico Chi-cuadrado y el p-valor.
    """
    start = time.time()
    frecuencias = df_filtrado[variable].value_counts().values
    n = frecuencias.sum()
    k = len(frecuencias)
    esperadas = [n / k] * k

    # Prueba Chi-cuadrado
    stat, p_value = chisquare(frecuencias, f_exp=esperadas)

    end = time.time()
    print(f"[scipy] Tiempo de análisis: {end - start:.4f} segundos")

    return stat, p_value
# -------------------------------------------------------- Fin de función Chi-cuadrado


# ----- *** Comparar el gasto entre tipos de clientes usando t-test *** -----
def comparar_gasto_ttest(df_filtrado, dato_1, dato_2):
    """
    Realiza una prueba t de Student para comparar el gasto entre dos tipos de clientes.
    Parámetros:
    - df_filtrado: DataFrame de pandas.
    - dato_1: columna categórica.
    - dato_2: columna numérica a comparar.
    Retorna un diccionario con el estadístico t, p-valor y conclusión.
    """
    tipos = df_filtrado[dato_1].unique()
    if len(tipos) != 2:
        raise ValueError("La columna de tipo de cliente debe tener exactamente dos categorías.")
    grupo1 = df_filtrado[df_filtrado[dato_1] == tipos[0]][dato_2]
    grupo2 = df_filtrado[df_filtrado[dato_1] == tipos[1]][dato_2]
    t_stat, p_val = ttest_ind(grupo1, grupo2, equal_var=False)
    conclusion = "Sí hay diferencia significativa" if p_val < 0.05 else "No hay diferencia significativa"
    return {
        'Tipo 1': tipos[0],
        'Tipo 2': tipos[1],
        't-stat': t_stat,
        'p-valor': p_val,
        'Conclusión': conclusion
    }
# -------------------------------------------------------- Fin de comparación de gasto entre tipos de clientes


# ---- *** Crear grafico de ventas por dia *** -----
def ventas_por_dia(df_filtrado):
    """
        Devuelve un DataFrame con las ventas totales por día y el promedio diario.
        Parámetros:
        - df_filtrado: DataFrame de pandas.
        Retorna:
        - ventas_diarias: DataFrame con columnas ['Fecha', 'Total']
        - promedio_diario: float
    """
    # Si existe columna 'DateTime', úsala; si no, crea a partir de 'Date'
    if 'DateTime' in df_filtrado.columns:
        fechas = pd.to_datetime(df_filtrado['DateTime']).dt.date
    else:
        fechas = pd.to_datetime(df_filtrado['Date']).dt.date

    df_filtrado['Total'] = df_filtrado['Total'].astype(float)
    ventas_diarias = df_filtrado.groupby(fechas)['Total'].sum().reset_index()
    ventas_diarias.columns = ['Fecha', 'Total']
    promedio_diario = ventas_diarias['Total'].mean()
    return ventas_diarias, promedio_diario
# -------------------------------------------------------- Fin de gráfico de ventas por día

# ----- *** Crear grafico de ventas por hora *** -----
def ventas_por_hora(df_filtrado):
    """
    Devuelve un DataFrame con las ventas totales por hora y el promedio horario.
    Parámetros:
    - df_filtrado: DataFrame de pandas.
    Retorna:
    - ventas_horarias: DataFrame con columnas ['Hora', 'Total']
    - promedio_horario: float
    """
    df_filtrado = df_filtrado.copy()

    # Intentar usar DateTime o construirla a partir de Date y Time
    if 'DateTime' not in df_filtrado.columns:
        if 'Date' in df_filtrado.columns and 'Time' in df_filtrado.columns:
            df_filtrado['DateTime'] = pd.to_datetime(df_filtrado['Date'] + ' ' + df_filtrado['Time'])
        else:
            raise ValueError("No se encontró una columna 'DateTime' ni 'Date' y 'Time' para construirla.")

    df_filtrado['DateTime'] = pd.to_datetime(df_filtrado['DateTime'], errors='coerce')
    df_filtrado['Total'] = pd.to_numeric(df_filtrado['Total'], errors='coerce')

    df_filtrado = df_filtrado.dropna(subset=['DateTime', 'Total'])  # eliminar valores erróneos

    df_filtrado['Hora'] = df_filtrado['DateTime'].dt.hour.astype(int)

    ventas_horarias = df_filtrado.groupby('Hora')['Total'].sum().reset_index()
    promedio_horario = ventas_horarias['Total'].mean()
    return ventas_horarias, promedio_horario

# -------------------------------------------------------- Fin de gráfico de ventas por hora

# ----- *** Funcion que grafica ventas por producto por dia *** -----
def ventas_por_producto(df_filtrado, producto):
    """
        Definicion: Genera un gráfico de líneas que muestra las ventas totales por día para un producto específico.
        Parámetros:
        - df_filtrado: DataFrame de pandas.
        - producto: nombre del producto a analizar.
    """
    df_filtrado['DateTime'] = pd.to_datetime(df_filtrado['DateTime'])
    df_filtrado['Total'] = df_filtrado['Total'].astype(float)
    ventas_producto = df_filtrado[df_filtrado['Product line'] == producto].groupby(df_filtrado['DateTime'].dt.date)['Total'].sum().reset_index()
    promedio_producto = ventas_producto['Total'].mean()
    plt.figure(figsize=(20, 6))
    sns.lineplot(data=ventas_producto, x='DateTime', y='Total', label=f'Ventas de {producto}')
    plt.axhline(promedio_producto, color='red', linestyle='--', label=f'Promedio diario: {promedio_producto:.2f}')
    plt.title(f'Ventas Totales por Día para {producto}')
    plt.xlabel('Fecha')
    plt.ylabel('Ventas Totales')
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()
    plt.show()
# -------------------------------------------------------- Fin de gráfico de ventas por producto

# ----- *** Funcion que grafica ventas por producto por hora *** -----
def ventas_por_producto_hora(df_filtrado, producto):
    """
        Definicion: Genera un gráfico de líneas que muestra las ventas totales por hora para un producto específico.
        Parámetros:
        - df_filtrado: DataFrame de pandas.
        - producto: nombre del producto a analizar.
    """
    df_filtrado['DateTime'] = pd.to_datetime(df_filtrado['DateTime'])
    df_filtrado['Total'] = df_filtrado['Total'].astype(float)
    ventas_producto_hora = df_filtrado[df_filtrado['Product line'] == producto].groupby(df_filtrado['DateTime'].dt.hour)['Total'].sum().reset_index()
    promedio_producto_hora = ventas_producto_hora['Total'].mean()
    plt.figure(figsize=(20, 6))
    sns.lineplot(data=ventas_producto_hora, x='DateTime', y='Total', label=f'Ventas de {producto}')
    plt.axhline(promedio_producto_hora, color='red', linestyle='--', label=f'Promedio por hora: {promedio_producto_hora:.2f}')
    plt.title(f'Ventas Totales por Hora para {producto}')
    plt.xlabel('Hora')
    plt.ylabel('Ventas Totales')
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()
    plt.show()
# -------------------------------------------------------- Fin de gráfico de ventas por producto por hora


# -----------------------------------------------------------------------------------------------------

# ----- *** Configuración de la aplicación Streamlit *** -----
st.set_page_config(
    page_title="📊 Dashboard Analítico de Ventas de Supermercado Grupo 41",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
# -------------------------------------------------------- Fin de configuración

# ----- *** Banner de la aplicación *** -----
st.markdown("""
<div style='padding-top:10px'>
    <h1 style='margin-bottom:0;'>📊 Dashboard Analítico de Ventas de Supermercado</h1>
    <p style='font-size:18px; color:gray; margin-top:4px; height:auto;'>
        Plataforma interactiva para explorar, visualizar y analizar los datos de ventas de una tienda.<br>
        Permite identificar tendencias, comparar productos, analizar el comportamiento de los clientes y evaluar métodos de pago.
    </p>
</div>
""", unsafe_allow_html=True)
# -------------------------------------------------------- Fin de banner

# ----- *** Banner de sidebar de la aplicación *** -----
st.sidebar.image('https://upload.wikimedia.org/wikipedia/commons/a/a4/Universidad-autonoma-de-chile.png', width=400 )  # Ajusta el tamaño de la imagen según sea necesario
st.sidebar.header("Menú de Navegación")
st.sidebar.markdown("""
<div style='background-color:#f0f0f0; border-radius:8px; padding:16px; margin-bottom:16px; font-size:15px;'>
<b>Integrantes del grupo:</b><br>
- Stephania Cañole<br>
- Vitto De la Fuente<br>
- Sergio Gutierrez<br>
- Nicolas Morales<br>
- Diego Nalli<br>
<br>
<i>Trabajo realizado para el módulo <b>Visualización de Información</b> en el área de <b>Big Data</b>.</i>
</div>
""", unsafe_allow_html=True)
st.sidebar.info("Aquí aparecerán todos los filtros e información de las columnas una vez que se cargue el fichero de datos de la evaluación.")
# ------------------------------------------------------ Fin de banner de sidebar


# ----- *** Carga de archivo *** -----
st.markdown("### 📂 Carga tu archivo de ventas")
# Boton deslizante de la cantidad de filas a mostrar del dataframe
filas = st.slider(
    label="Cantidad de filas a mostrar",
    min_value=1,
    max_value=50,
    value=7,
    step=1,
    help="Selecciona la cantidad de filas a mostrar del DataFrame."
)
archivo = st.file_uploader(
    label="Selecciona un archivo CSV o Excel",
    type=["csv", "xlsx"],
    help="El archivo debe contener los datos de ventas de la evaluación."
)

# ---- *** Procesar el archivo si se carga correctamente *** -----
if archivo is not None:
    try:
        if archivo.name.endswith('.csv'):
            start = time.time()
            df = pd.read_csv(archivo)
            end = time.time()
        else:
            start = time.time()
            df = pd.read_excel(archivo)
            end = time.time()
        # ---- *** Configuración de la tabla de datos *** -----
        config = {
            "Tax 5%": st.column_config.ProgressColumn(
                help="Impuesto del 5% aplicado a las ventas",
                format="%d",
                min_value=0,
                max_value=100,
            ),
            "Invoice ID": st.column_config.TextColumn(help="ID de la factura"),
            "Branch": st.column_config.TextColumn(help="Sucursal de la tienda"),
            "City": st.column_config.TextColumn(help="Ciudad de la tienda"),
            "Customer type": st.column_config.TextColumn(help="Tipo de cliente (Miembro o No miembro)"),
            "Gender": st.column_config.TextColumn(help="Género del cliente"),
            "Product line": st.column_config.TextColumn(help="Línea de producto"),
            "Unit price": st.column_config.NumberColumn(help="Precio unitario del producto"),
            "Quantity": st.column_config.NumberColumn(help="Cantidad de productos vendidos"),
            "Total": st.column_config.NumberColumn(help="Total de la venta"),
            "Date": st.column_config.TextColumn(help="Fecha y de venta"),
            "Time": st.column_config.TextColumn(help="Hora de la venta"),
            "Payment": st.column_config.TextColumn(help="Método de pago utilizado"),
            "cogs": st.column_config.NumberColumn(help="Costo de bienes vendidos"),
            "gross margin percentage": st.column_config.NumberColumn(help="Porcentaje de margen bruto"),
            "gross income": st.column_config.NumberColumn(help="Ingreso bruto de la venta"),
            "Rating": st.column_config.NumberColumn(help="Calificación del cliente")
        }

        # ---- *** Mostrar el DataFrame con Streamlit *** -----
        st.dataframe(df.head(filas),column_config=config, use_container_width=True, height=300)
        
        st.success(f"✅ Archivo procesado correctamente en {end - start:.4f} segundos.")

    except Exception as e:
        st.error(f"❌ Error al leer el archivo: {e}")
# --------------------------------------------------------------- Fin de carga de archivo

# ---- *** Filtros e informacion de sidebar *** -----
# --- *** Control de botones y filtros del sidebar *** ---

if 'df' in locals():

    # ---- *** Muestra detalle de columnas del dataframe *** -----
    st.sidebar.markdown("### 📘 Variables del dataset")
    if "mostrar_variables" not in st.session_state:
        st.session_state.mostrar_variables = False
    if st.sidebar.button(label="Mostrar/Ocultar Variables", help="Muestra el detalle de las variables del dataset", type="primary"):
        st.session_state.mostrar_variables = not st.session_state.mostrar_variables
    if st.session_state.mostrar_variables:
        st.sidebar.markdown("""
| Columna         | Definición                                                                                            |
| --------------- | ----------------------------------------------------------------------------------------------------- |
| `Invoice ID`    | Identificador único de cada factura                  |   
| `Branch`        | Sucursal donde se realizó la compra.                    |
| `City`          | Ciudad donde está ubicada la sucursal                      |
| `Customer type` | Tipo de cliente                                     |
| `Gender`        | Género del cliente                                                      |
| `Product line`  | Categoría del producto comprado                                           |
| `Payment`       | Método de pago usado                                            |
| `Unit price`              | Precio unitario del producto.                                            |
| `Quantity`                | Cantidad de productos comprados.                                         |
| `Tax 5%`                  | Impuesto aplicado.                                                       |
| `Total`                   | Total pagado por la compra, impuesto incluido.                           |
| `cogs`                    | Costo por producto (cantidad × precio).                                  |
| `gross margin percentage` | Porcentaje de margen bruto (4.76%) en todo el dataset.                   |
| `gross income`            | Ingreso bruto (equivale al impuesto).                                    |
| `Rating`                  | Puntuación de satisfacción del cliente               |
| `Date`    | Fecha en la que se realizó la compra.   |
| `Time`    | Hora en la que se realizó la compra.    |
""")
    # ------------------------------------------------------ Fin de detalle de columnas del dataframe

    # ----- *** Filtros de fecha general en el sidebar **** ----
    st.sidebar.markdown("### 📅 Filtro de Fecha")
    min_fecha = pd.to_datetime(df['Date']).min()
    max_fecha = pd.to_datetime(df['Date']).max()
    fecha_inicio, fecha_fin = st.sidebar.date_input(
        "Selecciona el rango de fechas",
        value=(min_fecha, max_fecha),
        min_value=min_fecha,
        max_value=max_fecha
    )
    # ------------------------------------------------------- Fin de filtros de fecha


    # Filtrar el DataFrame por el rango de fechas seleccionado
    df_filtrado = df[
        (pd.to_datetime(df['Date']) >= pd.to_datetime(fecha_inicio)) &
        (pd.to_datetime(df['Date']) <= pd.to_datetime(fecha_fin))
    ]

    # ----- *** Filtro de Product line *** -----
    st.sidebar.markdown("### 📦 Filtro de Líneas de Producto")
    product_lines = df_filtrado['Product line'].unique()
    selected_lines = st.sidebar.multiselect(
        "Selecciona las líneas de producto a comparar",
        options=product_lines,
        default=list(product_lines)
    )
    df_productos = df_filtrado[df_filtrado['Product line'].isin(selected_lines)]


    # Filtro de tipo de cliente (en el sidebar) - Multiselect
    st.sidebar.markdown("### 👤 Filtro de Tipo de Cliente")
    tipos_cliente = df_productos['Customer type'].unique()
    selected_tipos = st.sidebar.multiselect(
        "Selecciona el/los tipo(s) de cliente a mostrar",
        options=tipos_cliente,
        default=list(tipos_cliente)
    )

    # Filtrar el DataFrame según selección
    df_cliente = df_productos[df_productos['Customer type'].isin(selected_tipos)]
    # ------------------------------------------------------ Fin de filtro de Product line

    # --- Filtro de tipo de Payment en el sidebar ---
    st.sidebar.markdown("### 💳 Filtro de Método de Pago")
    metodos_pago_sidebar = df_cliente['Payment'].unique()
    selected_payments_sidebar = st.sidebar.multiselect(
        "Selecciona el/los método(s) de pago a mostrar",
        options=metodos_pago_sidebar,
        default=list(metodos_pago_sidebar)
    )

    # Filtrar el DataFrame según selección de Payment
    df_cliente = df_cliente[df_cliente['Payment'].isin(selected_payments_sidebar)]
    # ------------------------------------------------------ Fin de filtro de Payment

    # FIltro de sucursal en el sidebar
    st.sidebar.markdown("### 🏢 Filtro de Sucursal (Branch)")
    branches = df_cliente['Branch'].unique()
    selected_branches = st.sidebar.multiselect(
        "Selecciona la(s) sucursal(es) a mostrar",
        options=branches,
        default=list(branches)
    )
    # Filtrar el DataFrame según selección de Branch
    df_cliente = df_cliente[df_cliente['Branch'].isin(selected_branches)]

    # ----- *** KPIs resumen en 4 columnas *** -----
    st.markdown("## 📈 Métricas Clave (KPIs)")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        total_ingresos = df_cliente['Total'].sum()
        st.metric(label="Ingresos Totales", value=f"${total_ingresos:,.2f}")

    with kpi2:
        ingreso_bruto = df_cliente['gross income'].sum()
        st.metric(label="Ingreso Bruto Total", value=f"${ingreso_bruto:,.2f}")

    with kpi3:
        rating_promedio = df_cliente['Rating'].mean()
        st.markdown("**Calificación Promedio**")
        st.markdown(f"<h2 style='margin-top:-10px;'>{rating_promedio:.2f} ⭐</h2>", unsafe_allow_html=True)

    with kpi4:
        num_transacciones = len(df_cliente)
        st.metric(label="Número de Transacciones", value=f"{num_transacciones:,}")



# ------------------------------------------------------ Fin de estado de variables 



# ----- *** Seccion de análisis de datos *** -----
st.markdown("### 📊 Análisis de Datos") 
if 'df' not in locals():
    st.warning("⚠️ Los datos no han sido cargados. Para acceder al análisis, primero carga un archivo CSV o Excel.")
else:
    # ----- *** Analisis preliminar de los datos *** -----
    col1, col2 = st.columns(2)
    with col1:
        # --- Mostrar la cantidad de registros ---
        st.markdown("#### 📊 Estadísticas Descriptivas")
        st.write("Esta sección muestra estadísticas descriptivas del DataFrame, incluyendo la cantidad de registros, columnas y tipos de datos.")
        st.dataframe(df_cliente.describe(), height=320, use_container_width=True)
    with col2:
        st.write("#### 📊 Información del DataFrame")
        st.write("Esta sección muestra información del DataFrame, incluyendo la cantidad de registros unicos, columnas y tipos de datos.")
        st.dataframe(info_columnas(df_cliente, df_cliente.columns, n=5), height=320, use_container_width=True)
    # ------------------------------------------------- Fin de análisis preliminar de los datos

    # ----- *** Selección de Variables Relevantes para el Análisis *** -----
    with st.container():
        st.markdown("""<h3 style="margin-top:0;">Selección de Variables Relevantes para el Análisis</h3>""", unsafe_allow_html=True)            
        st.markdown("""
        <div style="background-color: #f9f9f9; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); padding: 24px; margin-bottom: 24px;">
            <p>Para realizar un análisis del comportamiento de ventas y clientes se seleccionan las siguientes variables:</p>
            <ul>
                <li><b>Variables Categóricas:</b>
                    <ul>
                        <li><code>Branch</code>, <code>City</code>: Permiten analizar geograficamente (posibles analisis futuros para categorizacion o clusterizacion).</li>
                        <li><code>Customer type</code>: Diferencia entre clientes fidelizados y ocasionales (posibles analisis futuros para categorizacion o clusterizacion).</li>
                        <li><code>Gender</code>: Permite explorar posibles diferencias de comportamiento por género (variables de diferenciacion Z)</li>
                        <li><code>Product line</code>: Permitira identificar productos mas vendidos y su impacto en los ingresos</li>
                        <li><code>Payment</code>: Entender las preferencias de pago de los clientes.</li>
                    </ul>
                </li>
                <li><b>Variables Numéricas:</b>
                    <ul>
                        <li><code>Unit price</code>, <code>Quantity</code>, <code>Total</code>: Fundamentales para analizar el volumen y valor de las ventas.</li>
                        <li><code>gross income</code>: Permiten evaluar la rentabilidad y el margen de ganancia.</li>
                        <li><code>Rating</code>: Refleja la satisfacción del cliente (KPI de satisfaccion)</li>
                    </ul>
                </li>
                <li><b>Variables Temporales:</b>
                    <ul>
                        <li><code>Date</code>, <code>Time</code>: Indispensables para analizar tendencias, patrones temporales y horarios de mayor actividad.</li>
                    </ul>
                </li>
            </ul>
            <p>Las siguientes variables por considerarse poco relevantes para el análisis solicitado:</p>
            <ul>
                <li><b><code>Invoice ID</code></b>: Es un id único de cada registro, es muy útil para registro interno, pero no aporta valor para el analisis</li>
                <li><b><code>Tax 5%</code></b> y <b><code>gross margin percentage</code></b>: Son variables constantes en el dataset.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    # -------------------------------------------------------- Fin de selección de variables relevantes

    # ----- *** Grafico de frecuencias de Payment por Fecha y hora *** -----
    # ----- *** Division de dos columnas *** -----
    col1_sec1, col2_sec1 = st.columns(2)
    with col1_sec1:
        # ----- *** Ventas por día *** -----
        st.markdown("#### 📈 Grafico de ventas por Día")
        st.write("Esta sección muestra un gráfico estático de ventas por día y el promedio diario.")
        ventas_diarias, promedio_diario = ventas_por_dia(df_cliente)
        fig, ax = plt.subplots(figsize=(20, 6))
        sns.lineplot(data=ventas_diarias, x='Fecha', y='Total', ax=ax)
        ax.axhline(promedio_diario, color='red', linestyle='--', label=f'Promedio diario: {promedio_diario:.2f}')
        ax.set_title('Ventas Totales por Día')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Ventas Totales')
        ax.legend()
        ax.grid()
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.write("Promedio diario de ventas:", round(promedio_diario, 2))
    # ----------------------------------------------- Fin de gráfico de ventas por día

    with col2_sec1:
        st.markdown("#### 📈 Gráfico de ventas por Hora")
        st.write("Esta sección muestra un gráfico estático de ventas por hora y el promedio por hora.")

        ventas_horarias, promedio_horario = ventas_por_hora(df_cliente)
        
        fig, ax = plt.subplots(figsize=(20, 6))
        sns.lineplot(data=ventas_horarias, x='Hora', y='Total', ax=ax, marker='o')
        ax.axhline(promedio_horario, color='red', linestyle='--', label=f'Promedio por hora: {promedio_horario:.2f}')
        ax.set_title('Ventas Totales por Hora')
        ax.set_xlabel('Hora')
        ax.set_ylabel('Ventas Totales')
        ax.legend()
        ax.grid()
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        st.write("Promedio por hora de ventas:", round(promedio_horario, 2))

    # ------------------------------------------------- Fin de gráfico de ventas por hora
    # ------------------------------------------------- Fin de frecuencia de Payment por Fecha y hora

    # ---- *** Grafico dinamico de frecuencias de ventas por Fecha *** -----
    with st.container(border=True):
        # Agrupar ventas por fecha
        ventas_fecha = (
            df_cliente
            .groupby(pd.to_datetime(df_cliente['Date']).dt.date)['Total']
            .sum()
            .reset_index()
            .rename(columns={'Date': 'Fecha', 'Total': 'Ventas'})
        )
        tab1, tab2 = st.tabs(["Gráfico", "Datos"])
        with tab1:
            st.markdown("#### 📈 Gráfico dinamico de Frecuencia de Ventas por Fecha")
            st.line_chart(
                data=ventas_fecha.set_index('Fecha')['Ventas'],
                height=250
            )
        with tab2:
            st.dataframe(ventas_fecha, height=250, use_container_width=True)
    # -------------------------------------------------------------------- Fin grafico de frecuencias de ventas por fecha
    
    # ---- *** Grafico dinamico de frecuencias de ventas por Hora *** -----
    with st.container(border=True):
        # Agrupar ventas por hora
        if 'DateTime' not in df_cliente.columns:
            if 'Date' in df_cliente.columns and 'Time' in df_cliente.columns:
                df_cliente['DateTime'] = pd.to_datetime(df_cliente['Date'] + ' ' + df_cliente['Time'])
            else:
                st.warning("No se encontró una columna 'DateTime' ni 'Date' y 'Time' para construirla.")
        df_cliente['Hora'] = pd.to_datetime(df_cliente['DateTime']).dt.hour
        ventas_hora = (
            df_cliente
            .groupby('Hora')['Total']
            .sum()
            .reset_index()
            .rename(columns={'Total': 'Ventas'})
        )
        tab1, tab2 = st.tabs(["Gráfico", "Datos"])
        with tab1:
            st.markdown("#### 📈 Gráfico dinamico de Frecuencia de Ventas por Hora")
            st.line_chart(
                data=ventas_hora.set_index('Hora')['Ventas'],
                height=250
            )
        with tab2:
            st.dataframe(ventas_hora, height=250, use_container_width=True)
    # -------------------------------------------------------------------- Fin grafico de frecuencias de ventas por hora


    # ---- *** Gráfico de rango de ventas diarias (máximo-mínimo) con filtro de fechas *** ----
    with st.container(border=True):
        st.markdown("#### 📊 Rango de Ventas Diarias: Maximos, Minimos y Rango")

        # Asegurarse de que DateTime existe y es datetime
        if 'DateTime' not in df_cliente.columns:
            if 'Date' in df_cliente.columns and 'Time' in df_cliente.columns:
                df_cliente['DateTime'] = pd.to_datetime(df_cliente['Date'] + ' ' + df_cliente['Time'])
            else:
                st.warning("No se encontró una columna 'DateTime' ni 'Date' y 'Time' para construirla.")

        df_cliente['DateTime'] = pd.to_datetime(df_cliente['DateTime'], errors='coerce')
        df_cliente['Total'] = pd.to_numeric(df_cliente['Total'], errors='coerce')

        # Agrupar por fecha
        ventas_por_dia = (
            df_cliente
            .groupby(df_cliente['DateTime'].dt.date)['Total']
            .agg(['min', 'max'])
            .reset_index()
        )
        ventas_por_dia.columns = ['Fecha', 'Venta Mínima', 'Venta Máxima']
        ventas_por_dia['Rango Venta'] = ventas_por_dia['Venta Máxima'] - ventas_por_dia['Venta Mínima']

        # Convertir a índice Fecha para gráficos dinámicos
        ventas_por_dia_indexed = ventas_por_dia.set_index('Fecha')

        tab1, tab2 = st.tabs(["Gráfico", "Datos"])
        with tab1:
            st.line_chart(ventas_por_dia_indexed[['Venta Mínima', 'Venta Máxima', 'Rango Venta']], height=350)
        with tab2:
            st.dataframe(ventas_por_dia, height=300, use_container_width=True)
    # -------------------------------------------------------------------- Fin gráfico de rango de ventas diarias

    

    # ----- *** Comparar los ingresos Total generados por cada Product line *** -----
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True, height=550):
            st.markdown("#### 💰 Comparación de Ingresos por Línea de Producto")

            ventas_por_producto = (
                df_cliente
                .groupby('Product line')['Total']
                .agg(['sum', 'mean', 'std'])
                .reset_index()
                .rename(columns={
                    'Product line': 'Producto',
                    'sum': 'Total Ventas',
                    'mean': 'Promedio Ventas',
                    'std': 'Desviación Estándar'
                })
            )
            ventas_por_producto['Total Ventas'] = ventas_por_producto['Total Ventas'].round(2)
            ventas_por_producto['Promedio Ventas'] = ventas_por_producto['Promedio Ventas'].round(2)
            ventas_por_producto['Desviación Estándar'] = ventas_por_producto['Desviación Estándar'].round(2)

            tab1, tab2 = st.tabs(["Gráfico", "Datos"])
            with tab1:
                # Gráfico dinámico con Altair
                chart = alt.Chart(ventas_por_producto).mark_bar().encode(
                    x=alt.X('Producto', sort='-y', title='Línea de Producto'),
                    y=alt.Y('Total Ventas', title='Total Ventas'),
                    tooltip=['Producto', 'Total Ventas', 'Promedio Ventas', 'Desviación Estándar'],
                    color=alt.Color('Producto', legend=None)
                ).properties(
                    width=350,
                    height=400,
                    title='Total de Ventas por Línea de Producto'
                )
                st.altair_chart(chart, use_container_width=True)

            with tab2:
                st.dataframe(ventas_por_producto, use_container_width=True)
        # ------------------------------------------------- Fin de comparación de ingresos por línea de producto

    with col2:
        with st.container(border=True, height=550):
            st.markdown("#### 🥧  Porcentaje de Ingresos por Línea de Producto")

            # Filtrar y agrupar
            df_donut = df_cliente[df_cliente['Product line'].isin(selected_lines)]
            ventas_donut = (
                df_donut.groupby('Product line')['Total']
                .sum()
                .reset_index()
                .rename(columns={'Product line': 'Producto', 'Total': 'Total Ventas'})
            )

            tab1, tab2 = st.tabs(["Gráfico", "Datos"])
            with tab1:
                if not ventas_donut.empty:
                    fig_donut = px.pie(
                        ventas_donut,
                        names='Producto',
                        values='Total Ventas',
                        hole=0.4,  # Efecto donut
                    )
                    fig_donut.update_traces(textinfo='percent+label', pull=[0.05]*len(ventas_donut))
                    fig_donut.update_layout(margin=dict(t=50, b=10, l=0, r=0), height=400)
                    st.plotly_chart(fig_donut, use_container_width=True)
                else:
                    st.info("No hay datos para mostrar el gráfico con los filtros actuales.")
            with tab2:
                st.dataframe(ventas_donut, use_container_width=True)
        # ------------------------------------------------- Fin de gráfico circular de ventas por Product line
    # ------------------------------------------------- Fin de comparación de ingresos por línea de producto

    # ---- *** Analisis de distribucion de Ratings *** -----
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True, height=550):

            st.markdown("#### 🧪 Prueba Chi-cuadrado sobre Ratings")
            st.write("Se evalúa si la distribución de las calificaciones (Ratings) es uniforme usando la prueba Chi-cuadrado.")
            try:
                stat, p_value = chi_cuadrado(df_cliente, 'Rating')
                st.write(f"**Estadístico Chi-cuadrado:** {stat:.2f}")
                st.write(f"**p-valor:** {p_value:.4f}")
                if p_value < 0.05:
                    st.warning("La distribución de Ratings **NO es uniforme** (p < 0.05).")
                    st.markdown("""
                    <div style="background-color:#ffebee; border-radius:8px; padding:16px; margin-top:16px;">
                    <b>Resolución:</b><br>
                    La prueba de Chi-cuadrado aplicada a la variable <code>Rating</code> indica que las calificaciones de los clientes presentan diferencias significativas respecto a una distribución uniforme. Esto sugiere que las valoraciones están distribuidas de manera heterogénea entre las distintas categorías, lo que podría indicar patrones atípicos en la satisfacción del cliente según las calificaciones registradas.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success("No se rechaza la hipótesis de uniformidad (p ≥ 0.05).")
                    st.markdown("""
                    <div style="background-color:#e8f5e9; border-radius:8px; padding:16px; margin-top:16px;">
                    <b>Resolución:</b><br>
                    La prueba de Chi-cuadrado aplicada a la variable <code>Rating</code> indica que las calificaciones de los clientes no presentan diferencias significativas respecto a una distribución uniforme. Esto sugiere que las valoraciones están distribuidas de manera homogénea entre las distintas categorías, sin sesgos notables hacia valores altos o bajos. Por lo tanto, no se identifican patrones atípicos en la satisfacción del cliente según las calificaciones registradas.
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error en la prueba Chi-cuadrado: {e}")

    with col2:
        with st.container(border=True, height=550):

            st.markdown("#### 📊 Histograma Interactivo de Ratings")

            # Agrupar y contar
            hist_data = (
                df_cliente['Rating']
                .round(0)
                .value_counts()
                .sort_index()
                .reset_index()
            )
            hist_data.columns = ['Rating', 'Frecuencia']

            tab1, tab2 = st.tabs(["Gráfico", "Datos"])
            with tab1:
                # Gráfico con Plotly
                fig_hist = px.bar(
                    hist_data,
                    x='Rating',
                    y='Frecuencia',
                    text='Frecuencia',
                    labels={'Rating': 'Rating', 'Frecuencia': 'Cantidad'},
                    title='Distribución de Ratings de Clientes'
                )
                fig_hist.update_traces(textposition='outside')
                fig_hist.update_layout(height=450, xaxis=dict(tickmode='linear'))
                st.plotly_chart(fig_hist, use_container_width=True)
            with tab2:
                st.dataframe(hist_data, use_container_width=True)

    # ------------------------------------------------- Fin de análisis de distribución de Ratings

    # ---- *** Gráfico circular de ventas por tipo de cliente *** -----
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True, height=550):
            st.markdown("#### 🥧 Porcentaje de Ingresos por Tipo de Cliente")

            # Agrupar y calcular estadísticas
            ventas_por_cliente = (
                df_cliente.groupby('Customer type')['Total']
                .agg(['sum', 'mean', 'std'])
                .reset_index()
                .rename(columns={
                    'Customer type': 'Tipo de Cliente',
                    'sum': 'Total Ventas',
                    'mean': 'Promedio Ventas',
                    'std': 'Desviación Estándar'
                })
            )
            ventas_por_cliente['Total Ventas'] = ventas_por_cliente['Total Ventas'].round(2)
            ventas_por_cliente['Promedio Ventas'] = ventas_por_cliente['Promedio Ventas'].round(2)
            ventas_por_cliente['Desviación Estándar'] = ventas_por_cliente['Desviación Estándar'].round(2)

            tab1, tab2 = st.tabs(["Gráfico", "Datos"])
            with tab1:
                if not ventas_por_cliente.empty:
                    fig_cliente = px.pie(
                        ventas_por_cliente,
                        names='Tipo de Cliente',
                        values='Total Ventas',
                        hole=0.4,
                    )
                    fig_cliente.update_traces(textinfo='percent+label', pull=[0.05]*len(ventas_por_cliente))
                    fig_cliente.update_layout(margin=dict(t=50, b=10, l=0, r=0), height=400)
                    st.plotly_chart(fig_cliente, use_container_width=True)
                else:
                    st.info("No hay datos para mostrar el gráfico con los filtros actuales.")
            with tab2:
                st.dataframe(ventas_por_cliente, use_container_width=True)
        # ------------------------------------------------- Fin de gráfico circular de ventas por tipo de cliente

    with col2:
        with st.container(border=True, height=550):
            st.markdown("#### 🧾 Comparación de Gasto entre Tipos de Clientes (t-test)")
            st.write("Se compara el gasto total entre los tipos de clientes usando la prueba t de Student para muestras independientes.")

            try:
                comparacion_gasto = comparar_gasto_ttest(df_cliente, 'Customer type', 'Total')
                comparacion_gasto_df = pd.DataFrame([comparacion_gasto])

                st.dataframe(comparacion_gasto_df, use_container_width=True)

                if comparacion_gasto['p-valor'] < 0.05:
                    st.warning("Existe una **diferencia significativa** en el gasto entre los tipos de clientes (p < 0.05).")
                    st.markdown("""
                    <div style="background-color:#ffebee; border-radius:8px; padding:16px; margin-top:16px;">
                    <b>Resolución:</b><br>
                    El análisis t-test indica que el gasto total entre los tipos de clientes es significativamente diferente. Esto sugiere que los clientes de un tipo gastan más (o menos) que los del otro, lo que puede ser relevante para estrategias de marketing o fidelización.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success("No hay diferencia significativa en el gasto entre los tipos de clientes (p ≥ 0.05).")
                    st.markdown("""
                    <div style="background-color:#e8f5e9; border-radius:8px; padding:16px; margin-top:16px;">
                    <b>Resolución:</b><br>
                    El análisis t-test indica que no existen diferencias significativas en el gasto total entre los tipos de clientes. Esto sugiere que ambos grupos presentan patrones de gasto similares.
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error en la comparación de gasto: {e}")
        # ------------------------------------------------- Fin de comparación de gasto entre tipos de clientes
        # ---- *** Gráfico de línea de ventas por tipo de pago (Payment) *** -----

    with st.container(border=True, height=550):
        st.markdown("#### 💳 Ventas por Tipo de Pago (Payment)")

        # Construir columna DateTime si no existe
        if 'DateTime' not in df_cliente.columns:
            if 'Date' in df_cliente.columns and 'Time' in df_cliente.columns:
                df_cliente['DateTime'] = pd.to_datetime(df_cliente['Date'] + ' ' + df_cliente['Time'])
            else:
                st.warning("No se encontró una columna 'DateTime' ni 'Date' y 'Time' para construirla.")

        # Crear columna de fecha simple
        df_cliente['Fecha'] = pd.to_datetime(df_cliente['DateTime'], errors='coerce').dt.date

        # Agrupar ventas por Fecha y Payment
        ventas_payment = (
            df_cliente
            .groupby(['Fecha', 'Payment'])['Total']
            .sum()
            .reset_index()
        )

        # Filtro dinámico desde el sidebar
        selected_payments = selected_payments_sidebar

        ventas_payment_filtrado = ventas_payment[ventas_payment['Payment'].isin(selected_payments)]

        tab1, tab2 = st.tabs(["Gráfico", "Datos"])

        with tab1:
            if len(selected_payments) >= 1 and not ventas_payment_filtrado.empty:
                fig = px.line(
                    ventas_payment_filtrado,
                    x='Fecha',
                    y='Total',
                    color='Payment',
                    markers=True,
                    labels={'Fecha': 'Fecha', 'Total': 'Total de Ventas', 'Payment': 'Método de Pago'},
                    title="Ventas por Día según Tipo de Pago"
                )
                fig.update_layout(height=450, xaxis_title='Fecha', yaxis_title='Total Ventas')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Selecciona al menos un método de pago en el filtro del sidebar para visualizar el gráfico.")

        with tab2:
            st.dataframe(ventas_payment_filtrado, use_container_width=True)
    # ------------------------------------------------- Fin de gráfico de línea de ventas por tipo de pago (Payment)

    # ---- *** Gráfico circular de ventas por tipo de pago (Payment) *** -----
    with st.container(border=True, height=600):
        st.markdown("#### 🥧 Porcentaje de Ventas por Tipo de Pago (Total Acumulado)")

        if not ventas_payment_filtrado.empty:
            # Agrupar total por método de pago
            total_por_pago = (
                ventas_payment_filtrado
                .groupby('Payment')['Total']
                .sum()
                .reset_index()
                .rename(columns={'Total': 'Total Ventas'})
            )

            # Gráfico circular con Plotly
            fig_donut = px.pie(
                total_por_pago,
                names='Payment',
                values='Total Ventas',
                hole=0.4,
                title='Distribución Porcentual de Ventas Totales por Método de Pago'
            )
            fig_donut.update_traces(textinfo='percent+label', pull=[0.05]*len(total_por_pago))
            fig_donut.update_layout(height=400, margin=dict(t=40, b=10, l=0, r=0))
            st.plotly_chart(fig_donut, use_container_width=True)

            # Mostrar tabla de respaldo
            total_por_pago['Porcentaje'] = (total_por_pago['Total Ventas'] / total_por_pago['Total Ventas'].sum() * 100).round(2)
            total_por_pago['%'] = total_por_pago['Porcentaje'].astype(str) + '%'
            st.dataframe(total_por_pago[['Payment', 'Total Ventas', '%']], use_container_width=True)
        else:
            st.info("No hay datos suficientes para mostrar el gráfico circular.")
    # ------------------------------------------------- Fin de gráfico circular de ventas por tipo de pago (Payment)

    # ---- *** Análisis comparativo de métodos de pago *** -----
    porcentajes = total_por_pago[['Payment', 'Porcentaje']].sort_values('Porcentaje', ascending=False).reset_index(drop=True)

    parrafo = ""

    # Regla 1: Distribución pareja (diferencia máxima ≤ 5%)
    if (porcentajes['Porcentaje'].max() - porcentajes['Porcentaje'].min()) <= 5:
        parrafo = (
            "La distribución de los métodos de pago es bastante homogénea, "
            "con diferencias porcentuales menores o iguales al 5% entre ellos. "
            "Esto sugiere que no hay una clara preferencia por un tipo de pago específico."
        )

    # Regla 2: Uno muy dominante (≥ 15% superior a todos los demás)
    elif all(
        (porcentajes.loc[0, 'Porcentaje'] - porcentajes.loc[i, 'Porcentaje']) >= 15
        for i in range(1, len(porcentajes))
    ):
        metodo_dominante = porcentajes.loc[0, 'Payment']
        parrafo = (
            f"El método de pago **{metodo_dominante}** sobresale significativamente, "
            f"superando al resto por al menos un 15%. Esto indica una marcada preferencia "
            f"por esta forma de pago entre los clientes."
        )

    # Regla 3: Dos métodos están claramente sobre una tercera
    elif len(porcentajes) >= 3:
        d1 = porcentajes.loc[0, 'Porcentaje'] - porcentajes.loc[2, 'Porcentaje']
        d2 = porcentajes.loc[1, 'Porcentaje'] - porcentajes.loc[2, 'Porcentaje']
        if d1 >= 5 and d2 >= 5:
            m1 = porcentajes.loc[0, 'Payment']
            m2 = porcentajes.loc[1, 'Payment']
            m3 = porcentajes.loc[2, 'Payment']
            parrafo = (
                f"Los métodos de pago **{m1}** y **{m2}** presentan una participación superior "
                f"en comparación con **{m3}**, superándolo cada uno por al menos 5%. "
                f"Esto indica que **{m3}** es significativamente menos utilizado."
            )

    # Si no aplica ninguna regla
    if not parrafo:
        parrafo = (
            "La distribución porcentual de los métodos de pago no presenta un patrón claro según las reglas definidas. "
            "Hay variaciones entre las categorías, pero no lo suficientemente marcadas como para identificar un sesgo fuerte."
        )

    # Mostrar el párrafo en Streamlit
    st.markdown(f"""
    <div style="background-color:#f9f9f9; padding:20px; border-radius:8px; border:1px solid #ddd;">
    <b>🧠 Análisis automático:</b><br>
    {parrafo}
    </div>
    """, unsafe_allow_html=True)
    # ------------------------------------------------- Fin de análisis comparativo de métodos de pago

    # ---- *** Mapa de Correlación entre Variables Numéricas *** -----
    with st.container(border=True):
        st.markdown("#### 🔗 Mapa de Correlación entre Variables Numéricas")

        # Selección de variables
        variables_corr = ['Unit price', 'Quantity', 'Tax 5%', 'Total', 'cogs', 'gross income', 'Rating']

        # Cálculo de la matriz de correlación
        df_corr = df_cliente[variables_corr].copy()
        df_corr = df_corr.apply(pd.to_numeric, errors='coerce')  # asegurarse que todas sean numéricas
        correlation_matrix = df_corr.corr().round(2)

        # Crear heatmap con Plotly
        z = correlation_matrix.values
        x = correlation_matrix.columns.tolist()
        y = correlation_matrix.index.tolist()

        fig_heatmap = ff.create_annotated_heatmap(
            z=z,
            x=x,
            y=y,
            annotation_text=correlation_matrix.values.round(2),
            colorscale='Viridis',
            showscale=True,
            hoverinfo="z"
        )

        fig_heatmap.update_layout(
            title="Mapa de Correlación (Heatmap)",
            height=600,
            margin=dict(t=60, l=10, r=10)
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Mostrar tabla (opcional)
        with st.expander("🔍 Ver matriz de correlación como tabla"):
            st.dataframe(correlation_matrix, use_container_width=True)

        # Mostrar conclusión como texto interpretativo
        st.markdown("""
        <div style="background-color:#f9f9f9; border-left: 5px solid #4caf50; padding: 20px; border-radius: 8px;">
            <b>🧠 Análisis Interpretativo:</b><br>
            La fuerte correlación observada entre las variables <code>gross income</code>, <code>cogs</code>, <code>Total</code> y <code>Tax 5%</code> es consecuencia directa de su naturaleza derivada: todas ellas se calculan a partir de los valores de <code>Unit price</code> y <code>Quantity</code>. Por lo tanto, la relación lineal entre estas columnas era previsible y responde a la propia estructura del dataset, no a una asociación significativa desde el punto de vista analítico. <br><br>
            En consecuencia, no es apropiado extraer conclusiones matemáticas sobre correlaciones entre estas variables, ya que no aportan información adicional relevante para el análisis del comportamiento de ventas o clientes.
        </div>
        """, unsafe_allow_html=True)
    # ------------------------------------------------- Fin de mapa de correlación entre variables numéricas

    # ----- *** Crear tabla dinámica con ingresos brutos por Branch y Product line  *** -----
    # ----- *** Filtro de Branch en el sidebar *** -----

    # -------------------------------------------------- Fin de filtro de Branch

    df_temp = df_cliente.copy()
    df_temp['gross income'] = pd.to_numeric(df_temp['gross income'], errors='coerce')

    pivot_income = (
        df_temp
        .groupby(['Branch', 'Product line'])['gross income']
        .sum()
        .reset_index()
        .pivot(index='Branch', columns='Product line', values='gross income')
    )

    # Convertir a porcentaje por fila
    pivot_income = pivot_income.div(pivot_income.sum(axis=1), axis=0).multiply(100).round(2)

    st.markdown("#### 📊 Composición del Ingreso Bruto por Sucursal y Línea de Producto (Gráfico Interactivo)")

    df_pivot = pivot_income.reset_index()

    tab1, tab2 = st.tabs(["Gráfico", "Datos"])
    with tab1:
        # Crear gráfico de barras apiladas
        fig = go.Figure()
        for product in pivot_income.columns:
            fig.add_trace(go.Bar(
                x=df_pivot['Branch'],
                y=df_pivot[product],
                name=product,
                text=df_pivot[product].round(1).astype(str) + '%',
                hovertemplate='%{y:.1f}% de %{x}<br>Línea: %{text}<extra></extra>',
                textposition='inside'
            ))
        fig.update_layout(
            barmode='stack',
            title='Composición del Ingreso Bruto por Sucursal y Línea de Producto',
            xaxis_title='Sucursal (Branch)',
            yaxis_title='Porcentaje del Ingreso Bruto',
            height=500,
            legend_title='Línea de Producto',
            yaxis=dict(tickformat=".0f", range=[0, 100])
        )
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.dataframe(df_pivot, use_container_width=True)
    # ------------------------------------------------- Fin de gráfico de composición del ingreso bruto por sucursal y línea de producto

    # ---- *** Análisis comparativo de aportación por línea de producto y sucursal *** ----
    comparaciones = []
    for idx, row in df_pivot.iterrows():
        branch = row['Branch']
        max_col = row.drop('Branch').idxmax()
        max_val = row[max_col]
        comparaciones.append(f"En la sucursal **{branch}**, la línea de producto con mayor aportación al ingreso bruto es **{max_col}** con un {max_val:.2f}% del total.")

    parrafo_final = "<br>".join(comparaciones)

    st.markdown(f"""
    <div style="background-color:#f9f9f9; border-left: 5px solid #2196f3; padding: 20px; border-radius: 8px;">
    <b>🔎 Comparación por Sucursal:</b><br>
    {parrafo_final}
    </div>
    """, unsafe_allow_html=True)