# main.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import math # Necesario si alguna función de mmc lo usa directamente (aunque mmc ya lo importa)

# Importar los módulos personalizados
import mmc #En mmc.py están las funciones que realizan los cálculos principales del modelo M/M/C
import montecarlo #En montecarlo.py se hace la simulación de montecarlo


def format_time(value_hours):
    """Formatea tiempos en horas a formato legible (horas o minutos)"""
    if value_hours == float('inf'):
        return "Infinito"
    
    if value_hours < 0.1:  # Mostrar en minutos si es menos de 60 min
        minutes = value_hours * 60
        return f"{minutes:.1f} min"
    elif value_hours < 1:  # Mostrar en minutos con decimales
        minutes = value_hours * 60
        return f"{minutes:.0f} min"
    else:  # Mostrar en horas
        return f"{value_hours:.2f} horas"

# Configuración de la página de Streamlit
st.set_page_config(layout="wide", page_title="Simulador M/M/c Avanzado", page_icon="📊")

# --- Interfaz Principal / Menú de Navegación en la Sidebar ---
st.sidebar.title("Menú de Operaciones") 
opcion_menu = st.sidebar.radio(
    "Selecciona una operación:",
    ("Introducción", "Analizar Rendimiento (Medir)", "Simulación de Eventos (Simular)")
)

# --- Contenido Principal Basado en la Selección del Menú ---
if opcion_menu == "Introducción":
    st.title("Bienvenido al Simulador Avanzado de Líneas de Espera M/M/c 📊")
    st.markdown("""
    Esta aplicación web te permite simular y analizar un sistema de colas M/M/c 
    (llegadas Poisson, tiempos de servicio exponenciales, 'c' servidores) de forma interactiva.
    
    **Funcionalidades:**
    - **Analizar Rendimiento:** Calcula métricas clave del sistema (utilización, longitud de cola, tiempos de espera, etc.) basándose en las fórmulas teóricas del modelo M/M/c.
    - **Optimizar y Predecir:** (Próximamente) Explora cómo los cambios en los parámetros afectan el sistema y herramientas para optimizar la configuración (ej. número de servidores basado en costos).
    - **Simulación de Eventos:** Ejecuta una simulación de Monte Carlo para observar el comportamiento dinámico del sistema y comparar los resultados con los teóricos.

    Utiliza el menú de la izquierda para navegar entre las diferentes operaciones.
    Desarrollado como parte del proyecto de Modelos y Simulación.
    """)
    st.markdown("---")
    st.image("https://i.imgur.com/nBNv5Nj.jpeg", caption="Esquema de un sistema de colas M/M/c", width=400)
    st.info("Selecciona una operación del menú de la izquierda para comenzar.")

elif opcion_menu == "Analizar Rendimiento (Medir)":
    st.title("Análisis de Rendimiento Teórico del Sistema M/M/c")
    st.markdown("""
    Modifica los parámetros en la barra lateral para observar cómo afectan las métricas del sistema y
    la longitud promedio de la cola ($L_q$). Los cálculos se basan en las fórmulas exactas del modelo M/M/c.
    """)
    #Aquí se agregan los valores de λ y de μ y de c  
    st.sidebar.header("Parámetros del Sistema (Medir)")
    lambda_actual = st.sidebar.slider(
        "Tasa de llegada λ (clientes/hora):", 0.1, 50.0, 5.0, 0.1, key="lambda_medir"
    )
    mu_actual = st.sidebar.slider(
        "Tasa de servicio μ (por servidor):", 0.1, 20.0, 2.0, 0.1, key="mu_medir"
    )
    c_servidores = st.sidebar.number_input(
        "Número de servidores c:", 1, 15, 3, 1, key="c_medir"
    )

    # Cálculos usando el módulo mmc
    rho = mmc.calcular_rho(lambda_actual, mu_actual, c_servidores)
    p0 = mmc.calcular_p0(lambda_actual, mu_actual, c_servidores, rho)
    lq = mmc.calcular_lq(lambda_actual, mu_actual, c_servidores) # lq usa p0 y rho internamente
    wq = mmc.calcular_wq(lq, lambda_actual) if lambda_actual > 0 else 0.0
    w = mmc.calcular_w(wq, mu_actual) if mu_actual > 0 else float('inf')
    l_sistema = mmc.calcular_l(lq, lambda_actual, mu_actual) if mu_actual > 0 else float('inf')
    pw = mmc.calcular_probabilidad_espera(lambda_actual, mu_actual, c_servidores, p0, rho)

    st.subheader("Métricas de Rendimiento Calculadas")
    
    # Presentación de métricas
    if rho == float('inf'):
        st.error("Configuración inválida (μ o c probablemente es 0 o negativo).")
    elif rho >= 1.0:
        st.warning(f"¡Sistema INESTABLE! (ρ = {rho:.3f} >= 1). La tasa de llegada es demasiado alta.")
        mcol1, mcol2 = st.columns(2)
        mcol1.metric("Utilización ρ", f"{rho:.3f}")
        mcol2.metric("Probabilidad de Espera $P_w$", f"{pw:.3f}" if pw is not None else "N/A")
        st.info("Lq, Wq, L, W tienden a infinito.")

    else:
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Utilización ρ", f"{rho:.3f}")
        mcol1.metric("Probabilidad Sistema Vacío $P_0$", f"{p0:.4f}")
        mcol1.metric("Probabilidad de Espera $P_w$", f"{pw:.3f}")

        mcol2.metric("Longitud Prom. Cola $L_q$", f"{lq:.3f} clientes")
        mcol2.metric("Tiempo Prom. en Cola $W_q$",format_time(wq))
        
        mcol3.metric("Clientes Prom. Sistema $L$", f"{l_sistema:.3f} clientes")
        mcol3.metric("Tiempo Prom. en Sistema $W$", format_time(w))

    st.markdown("---")
    #-----Gráfica----#
    st.subheader("Gráfico: $L_q$ vs. Tasa de Llegada (λ)")
    
    if mu_actual > 0 and c_servidores > 0:
        st.markdown(f"Variando λ, con μ={mu_actual} y c={c_servidores} fijos.")
        
        max_lambda_estable = c_servidores * mu_actual * 0.999 
        if max_lambda_estable <= 0.1: 
            max_lambda_estable = lambda_actual * 2 if lambda_actual > 0.05 else 1.0

        lambda_valores_graf = np.linspace(0.01, max_lambda_estable, 100)
        lq_valores_graf = [mmc.calcular_lq(l_val, mu_actual, c_servidores) for l_val in lambda_valores_graf]
        
        lq_valores_filtrados_graf = [lq_g if lq_g != float('inf') and lq_g < (lq * 10 if not math.isinf(lq) and lq > 10 else 100) else np.nan for lq_g in lq_valores_graf] # Ajustar límite dinámicamente

        df_grafico = pd.DataFrame({
            'Tasa de Llegada (λ)': lambda_valores_graf,
            'Longitud Promedio de Cola (Lq)': lq_valores_filtrados_graf
        })

        if not df_grafico['Longitud Promedio de Cola (Lq)'].dropna().empty:
            fig = px.line(df_grafico, x='Tasa de Llegada (λ)', y='Longitud Promedio de Cola (Lq)',
                          labels={'Tasa de Llegada (λ)': 'Tasa de Llegada λ (clientes/hora)', 
                                  'Longitud Promedio de Cola (Lq)': 'Lq (clientes)'},
                          title=f"Impacto de λ en $L_q$ (μ={mu_actual}, c={c_servidores})")
            if lambda_actual <= max_lambda_estable :
                 fig.add_vline(x=lambda_actual, line_width=2, line_dash="dash", line_color="red", 
                               annotation_text="λ actual", annotation_position="top right")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No se pudieron generar datos graficables para $L_q$ vs λ con los parámetros actuales.")
    else:
        st.warning("μ y c deben ser mayores que 0 para generar el gráfico $L_q$ vs λ.")


    st.title("Optimización y Predicción de Escenarios M/M/c")
    st.markdown("Esta sección permitirá explorar la optimización de costos y predecir el impacto de cambios en los parámetros.")
    st.info("Funcionalidad de Optimización y Predicción en desarrollo. Próximamente podrás:")
    st.markdown("""
    - Calcular el número óptimo de servidores basado en costos de servicio y costos de espera.
    - Analizar la sensibilidad de las métricas a cambios en λ, μ y c (incluyendo derivadas).
    - Realizar análisis "What-if" para diferentes escenarios.
    """)

elif opcion_menu == "Simulación de Eventos (Simular)":
    # Simulación de montecarlo que se encuentra en montecarlo.py
    montecarlo.display_montecarlo_section()

# Pie de página en la sidebar
st.sidebar.markdown("---")
st.sidebar.info("Proyecto: Modelo de Sistema de Colas Grupo 2")