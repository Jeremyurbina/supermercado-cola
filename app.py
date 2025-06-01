import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import math
#TODO: Se pueden separar las funciones para mejorar la organizacion del codigo
#Funciones para la grafica
def calcular_rho(lambda_llegadas, mu_servicio, c_servidores):
    """Calcula la utilización del sistema (rho)."""
    if c_servidores == 0 or mu_servicio == 0:
        return float('inf') # Evitar división por cero o escenario sin sentido
    return lambda_llegadas / (c_servidores * mu_servicio)

def calcular_p0(lambda_llegadas, mu_servicio, c_servidores, rho):
    """Calcula la probabilidad de que no haya clientes en el sistema (P0)."""
    if rho >= 1.0:
        return 0.0 # Sistema inestable o saturado

    termino_suma = 0.0
    for n in range(c_servidores):
        termino_suma += (math.pow(lambda_llegadas / mu_servicio, n)) / math.factorial(n)

    termino_c = (math.pow(lambda_llegadas / mu_servicio, c_servidores)) / \
                (math.factorial(c_servidores) * (1.0 - rho))
    
    if (termino_suma + termino_c) == 0: # Evitar división por cero si todos los términos son cero
        return float('inf') # O algún otro indicador de problema
    
    p0 = 1.0 / (termino_suma + termino_c)
    return p0

def calcular_lq(lambda_llegadas, mu_servicio, c_servidores):
    """Calcula la longitud promedio de la cola (Lq)."""
    rho = calcular_rho(lambda_llegadas, mu_servicio, c_servidores)
    
    if rho >= 1.0:
        return float('inf') # Cola infinita si el sistema es inestable

    p0 = calcular_p0(lambda_llegadas, mu_servicio, c_servidores, rho)
    
    numerador_lq = p0 * math.pow(lambda_llegadas / mu_servicio, c_servidores) * rho
    denominador_lq = math.factorial(c_servidores) * math.pow(1.0 - rho, 2)
    
    if denominador_lq == 0:
         return float('inf') # Evitar división por cero

    lq = numerador_lq / denominador_lq
    return lq

#Interfaz de Streamlit
st.set_page_config(layout="wide")
st.title("Simulador de Líneas de Espera: Modelo M/M/c")
st.markdown("""
Esta aplicación web permite simular un sistema de colas M/M/c (llegadas Poisson, tiempos de servicio exponenciales, 'c' servidores).
Puedes modificar los parámetros de entrada para observar cómo afectan la longitud promedio de la cola.
""")

#Controles en la barra lateral
st.sidebar.header("Parámetros del Sistema")

# Tasa de llegada (lambda) - el dato que modificaremos para el gráfico principal
lambda_actual = st.sidebar.slider(
    "Tasa de llegada de clientes (λ clientes/hora):", 
    min_value=0.1, 
    max_value=20.0, 
    value=5.0,  # Valor inicial
    step=0.1
)

# Tasa de servicio por cajero (mu)
mu_actual = st.sidebar.slider(
    "Tasa de servicio por cajero (μ clientes/hora por cajero):",
    min_value=0.1,
    max_value=10.0,
    value=2.0, # Valor inicial
    step=0.1
)

# Número de cajeros (c) - fijo a 3 para el ejemplo, pero tambien podría ser un input
c_servidores = st.sidebar.number_input(
    "Número de cajeros/servidores (c):",
    min_value=1,
    max_value=10,
    value=3, # Ejemplo: 3 cajeros
    step=1,
    help="Para este ejemplo, el valor inicial es 3, como en el supermercado."
)

#Cálculos y Visualización
col1, col2 = st.columns(2)

with col1:
    st.subheader("Resultados para los Parámetros Actuales")
    
    rho_actual = calcular_rho(lambda_actual, mu_actual, c_servidores)
    lq_actual = calcular_lq(lambda_actual, mu_actual, c_servidores)
    
    st.metric(label="Factor de Utilización (ρ)", value=f"{rho_actual:.3f}")
    if rho_actual >= 1.0:
        st.warning("¡El sistema es inestable! (ρ >= 1). La tasa de llegada es demasiado alta para la capacidad de servicio.")
        st.metric(label="Longitud Promedio de la Cola (Lq)", value="Infinita (o muy grande)")
    else:
        st.metric(label="Longitud Promedio de la Cola (Lq)", value=f"{lq_actual:.3f} clientes")

    st.markdown(f"""
    **Parámetros usados:**
    - Tasa de llegada (λ): {lambda_actual} clientes/hora
    - Tasa de servicio por cajero (μ): {mu_actual} clientes/hora
    - Número de cajeros (c): {c_servidores}
    """)

with col2:
    st.subheader("Gráfico: Lq vs. Tasa de Llegada (λ)")
    st.markdown(f"Este gráfico muestra cómo cambia $L_q$ al variar λ, manteniendo μ={mu_actual} y c={c_servidores} fijos.")

    # Generar datos para el gráfico
    # Variamos lambda desde un valor bajo hasta cerca de la saturación
    max_lambda_estable = c_servidores * mu_actual * 0.999 # Un poco menos que la saturación
    if max_lambda_estable <= 0.1: # Asegurar que hay un rango válido
        max_lambda_estable = lambda_actual * 2 if lambda_actual > 0 else 10.0

    lambda_valores = np.linspace(0.1, max_lambda_estable, 100) # 100 puntos para el gráfico
    lq_valores = [calcular_lq(l, mu_actual, c_servidores) for l in lambda_valores]
    
    # Filtrar valores infinitos o muy grandes para una mejor visualización
    lq_valores_filtrados = [lq if lq != float('inf') and lq < 1000 else np.nan for lq in lq_valores] # Limitar Lq a 1000 para la gráfica

    df_grafico = pd.DataFrame({
        'Tasa de Llegada (λ)': lambda_valores,
        'Longitud Promedio de Cola (Lq)': lq_valores_filtrados
    })

    if not df_grafico['Longitud Promedio de Cola (Lq)'].dropna().empty:
        fig = px.line(df_grafico, x='Tasa de Llegada (λ)', y='Longitud Promedio de Cola (Lq)',
                      title=f"Lq vs. λ (μ={mu_actual}, c={c_servidores})")
        fig.add_vline(x=lambda_actual, line_width=2, line_dash="dash", line_color="red", 
                      annotation_text="λ actual", annotation_position="top right")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No se pueden generar datos para el gráfico con los parámetros actuales (posiblemente μ o c es 0, o λ es demasiado alto para generar un rango).")

st.sidebar.markdown("---")
st.sidebar.info("Modifica los parámetros y observa cómo cambian los resultados y el gráfico en tiempo real.")