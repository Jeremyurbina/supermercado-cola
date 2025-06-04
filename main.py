
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import math
from scipy.integrate import simpson
import mmc
import montecarlo

# Función de formato para tiempos
def format_time(value_hours):
    """Formatea tiempos en horas a formato legible (horas o minutos)"""
    if value_hours == float('inf'):
        return "Infinito"
    
    if value_hours < 0.0167:  # Menos de 1 minuto
        seconds = value_hours * 3600
        return f"{seconds:.0f} seg"
    elif value_hours < 0.1:   # Menos de 6 minutos
        minutes = value_hours * 60
        return f"{minutes:.1f} min"
    elif value_hours < 1:     # Menos de 1 hora
        minutes = value_hours * 60
        return f"{minutes:.0f} min"
    else:                     # 1 hora o más
        return f"{value_hours:.2f} horas"

# Función para calcular derivada numérica de Lq respecto a lambda
def calcular_derivada_lq(lambda_val, mu, c, h_factor=0.01):
    """
    Calcula la derivada numérica dLq/dλ usando diferencia centrada.
    
    Args:
        lambda_val: Valor actual de lambda (tasa de llegada)
        mu: Tasa de servicio por servidor
        c: Número de servidores
        h_factor: Fracción de lambda para el paso h (default: 1%)
    
    Returns:
        Derivada numérica dLq/dλ en el punto lambda_val
    """
    if lambda_val > 0:
        h = h_factor * lambda_val
    else:
        h = 0.01
    
    # Calcular Lq en lambda - h y lambda + h
    lq_minus = mmc.calcular_lq(lambda_val - h, mu, c)
    lq_plus = mmc.calcular_lq(lambda_val + h, mu, c)
    
    # Si alguno es infinito, la derivada podría ser infinita
    if math.isinf(lq_minus) or math.isinf(lq_plus):
        return float('inf')
    
    return (lq_plus - lq_minus) / (2 * h)

# Configuración de la página
st.set_page_config(layout="wide", page_title="Simulador M/M/c Avanzado", page_icon="📊")

# --- Interfaz Principal / Menú de Navegación en la Sidebar ---
st.sidebar.title("Menú de Operaciones") 
opcion_menu = st.sidebar.radio(
    "Selecciona una operación:",
    ("Introducción", "Analizar Rendimiento (Medir)", "Simulación de Eventos (Simular)")
)

# --- Contenido Principal Basado en la Selección del Menú ---
if opcion_menu == "Introducción":

    col1, col2 = st.columns(2)

    with col1:
        st.image("/images/imagen1.jpg", use_container_width=True)

    with col2:
        st.image("/images/imagen2.jpg", use_container_width=True)


    st.title("Bienvenido a la Aplicación de simulación y medición de sistemas de Modelo de cola M/M/c 📊")

    st.markdown("""
    **Realizado por:**

    Jeremy Urbina 27098844  
    Jesús Cisneros 20820073  
    Guillermo Del Aguila 30089792  
    Cesar Bruzual 28286095  
    Anderson Omaña 27472804
    """)

    st.markdown("""
    Esta aplicación web te permite simular y analizar un sistema de colas M/M/c 
    (llegadas Poisson, tiempos de servicio exponenciales, 'c' servidores) de forma interactiva.
    
    **Funcionalidades:**
    - **Analizar Rendimiento:** Calcula métricas clave del sistema (utilización, longitud de cola, tiempos de espera, etc.) basándose en las fórmulas teóricas del modelo M/M/c.
    - **Simulación de Eventos:** Ejecuta una simulación de Monte Carlo para observar el comportamiento dinámico del sistema y comparar los resultados con los teóricos.

    Utiliza el menú de la izquierda para navegar entre las diferentes operaciones.
    """)
    st.markdown("---")
    st.info("Selecciona una operación del menú de la izquierda para comenzar.")

elif opcion_menu == "Analizar Rendimiento (Medir)":
    st.title("Análisis de Rendimiento Teórico del Sistema M/M/c")
    st.markdown("""
    Modifica los parámetros en la barra lateral para observar cómo afectan las métricas del sistema y
    la longitud promedio de la cola ($L_q$). Los cálculos se basan en las fórmulas exactas del modelo M/M/c.
    """)
    
    # Controles en la barra lateral
    st.sidebar.header("Parámetros del Sistema (Medir)")
    lambda_actual = st.sidebar.slider("Tasa de llegada λ (clientes/hora):", 0.1, 50.0, 5.0, 0.1)
    mu_actual = st.sidebar.slider("Tasa de servicio μ (por servidor):", 0.1, 20.0, 2.0, 0.1)
    c_servidores = st.sidebar.number_input("Número de servidores c:", 1, 15, 3, 1)

    # Cálculos usando el módulo mmc
    rho = mmc.calcular_rho(lambda_actual, mu_actual, c_servidores)
    p0 = mmc.calcular_p0(lambda_actual, mu_actual, c_servidores, rho)
    lq = mmc.calcular_lq(lambda_actual, mu_actual, c_servidores)
    wq = mmc.calcular_wq(lq, lambda_actual) if lambda_actual > 0 else 0.0
    w = mmc.calcular_w(wq, mu_actual) if mu_actual > 0 else float('inf')
    l_sistema = mmc.calcular_l(lq, lambda_actual, mu_actual)
    pw = mmc.calcular_probabilidad_espera(lambda_actual, mu_actual, c_servidores, p0, rho)

    # Presentación de métricas
    st.subheader("Métricas de Rendimiento Calculadas")

    with st.expander("📐 Ver fórmulas utilizadas en los cálculos"):
        st.markdown(r"""
        ### Fórmulas del Modelo M/M/c

        - **Utilización del sistema:**  
          $$\rho = \frac{\lambda}{c \cdot \mu}$$

        - **Probabilidad de sistema vacío:**  
          $$P_0 = \left[ \sum_{n=0}^{c-1} \frac{1}{n!} \left(\frac{\lambda}{\mu}\right)^n + \frac{1}{c!} \left(\frac{\lambda}{\mu}\right)^c \cdot \frac{c \mu}{c \mu - \lambda} \right]^{-1}$$

        - **Longitud promedio de la cola:**  
          $$L_q = \frac{(\lambda \mu) \cdot \rho^c}{(c-1)! (c\mu - \lambda)^2} \cdot P_0$$

        - **Tiempo promedio en la cola:**  
          $$W_q = \frac{L_q}{\lambda}$$

        - **Tiempo promedio total en el sistema:**  
          $$W = W_q + \frac{1}{\mu}$$

        - **Longitud promedio total en el sistema:**  
          $$L = \lambda \cdot W$$

        - **Probabilidad de tener que esperar (espera en cola):**  
          $$P_w = \frac{(\lambda/\mu)^c}{c! \cdot (1 - \rho)} \cdot P_0$$
        """)
    
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
        mcol2.metric("Tiempo Prom. en Cola $W_q$", format_time(wq))
        
        mcol3.metric("Clientes Prom. Sistema $L$", f"{l_sistema:.3f} clientes")
        mcol3.metric("Tiempo Prom. en Sistema $W$", format_time(w))

    # Gráfico Lq vs λ
    st.markdown("---")
    st.subheader("Gráfico: $L_q$ vs. Tasa de Llegada (λ)")
    
    if mu_actual > 0 and c_servidores > 0:
        st.markdown(f"Variando λ, con μ={mu_actual} y c={c_servidores} fijos.")
        
        max_lambda_estable = c_servidores * mu_actual * 0.999
        if max_lambda_estable <= 0.1: 
            max_lambda_estable = lambda_actual * 2 if lambda_actual > 0.05 else 1.0

        lambda_valores_graf = np.linspace(0.01, max_lambda_estable, 100)
        lq_valores_graf = [mmc.calcular_lq(l_val, mu_actual, c_servidores) for l_val in lambda_valores_graf]
        
        # Filtrar valores infinitos
        lq_valores_filtrados = [lq if not math.isinf(lq) else np.nan for lq in lq_valores_graf]
        
        df_grafico = pd.DataFrame({
            'Tasa de Llegada (λ)': lambda_valores_graf,
            'Longitud Promedio de Cola (Lq)': lq_valores_filtrados
        })

        if not df_grafico['Longitud Promedio de Cola (Lq)'].dropna().empty:
            # Crear gráfico principal
            fig = px.line(df_grafico, x='Tasa de Llegada (λ)', y='Longitud Promedio de Cola (Lq)',
                          labels={'Tasa de Llegada (λ)': 'Tasa de Llegada λ (clientes/hora)', 
                                  'Longitud Promedio de Cola (Lq)': 'Lq (clientes)'},
                          title=f"Impacto de λ en $L_q$ (μ={mu_actual}, c={c_servidores})")
            
            if lambda_actual <= max_lambda_estable:
                fig.add_vline(x=lambda_actual, line_width=2, line_dash="dash", line_color="red", 
                              annotation_text="λ actual", annotation_position="top right")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # --- ANÁLISIS DERIVADA E INTEGRAL ---
            st.subheader("Derivación de Lq con respecto a λ ")
            tab_deriv, tab_integral = st.tabs(["Sensibilidad Instantánea (Derivada)", "Impacto Acumulado (Integral)"])
            
            with tab_deriv:
                st.markdown("""
                **Derivada dLq/dλ**:
                - Mide cuánto cambia la longitud de la cola ($L_q$) ante pequeños cambios en la tasa de llegada (λ)
                - Valores altos indican que el sistema es muy sensible a cambios en λ
                """)
                
                # Calcular derivada en el punto actual
                derivada_actual = calcular_derivada_lq(lambda_actual, mu_actual, c_servidores)
                
                col_deriv1, col_deriv2 = st.columns([1, 2])
                
                with col_deriv1:
                    st.metric("Sensibilidad actual (dLq/dλ)", f"{derivada_actual:.2f}")
                    
                    # Interpretación cualitativa
                    if derivada_actual == float('inf'):
                        st.error("El sistema es infinitamente sensible en este punto")
                    elif derivada_actual > 10:
                        st.warning("Sistema altamente sensible: pequeños cambios en λ causan grandes colas")
                    elif derivada_actual > 1:
                        st.info("Sistema moderadamente sensible")
                    else:
                        st.success("Sistema robusto: cambios en λ tienen bajo impacto")
                
                with col_deriv2:
                    # Calcular derivada para todos los puntos
                    derivadas = [calcular_derivada_lq(l, mu_actual, c_servidores) 
                                 for l in lambda_valores_graf]
                    
                    df_deriv = pd.DataFrame({
                        'λ': lambda_valores_graf,
                        'dLq/dλ': derivadas
                    })
                    
                    fig_deriv = px.line(df_deriv, x='λ', y='dLq/dλ',
                                        title="Sensibilidad del sistema a cambios en λ")
                    fig_deriv.add_vline(x=lambda_actual, line_dash="dash", line_color="red")
                    fig_deriv.update_layout(yaxis_title="dLq/dλ")
                    st.plotly_chart(fig_deriv, use_container_width=True)
            
            with tab_integral:
                st.markdown("""
                **Integral ∫Lq dλ**:
                - Representa el impacto acumulado de las colas en el rango de λ
                - Área bajo la curva Lq vs λ
                - Útil para comparar diferentes configuraciones del sistema
                """)
                
                # Calcular integral numérica
                mask = np.isfinite(lq_valores_filtrados)
                lambda_finite = lambda_valores_graf[mask]
                lq_finite = np.array(lq_valores_filtrados)[mask]
                
                if len(lambda_finite) > 1:
                    integral = simpson(lq_finite, lambda_finite)
                    
                    col_integ1, col_integ2 = st.columns([1, 2])
                    
                    with col_integ1:
                        st.metric("Impacto acumulado", 
                                  f"{integral:.2f} clientes·λ", 
                                  help="Área bajo la curva Lq vs λ")
                        st.info("""
                        **Interpretación**:
                        - Valores más altos = Mayor carga acumulada
                        - Comparar entre configuraciones (ej. diferente c)
                        """)
                    
                    with col_integ2:
                        # Crear gráfico con área sombreada
                        fig_integ = px.area(df_grafico.dropna(), 
                                           x='Tasa de Llegada (λ)', 
                                           y='Longitud Promedio de Cola (Lq)',
                                           title="Área bajo la curva Lq vs λ")
                        fig_integ.add_trace(px.line(df_grafico.dropna(), 
                                                  x='Tasa de Llegada (λ)', 
                                                  y='Longitud Promedio de Cola (Lq)').data[0])
                        fig_integ.add_vline(x=lambda_actual, line_dash="dash", line_color="red")
                        
                        # Anotar valor de la integral
                        fig_integ.add_annotation(
                            x=0.7 * max(lambda_finite),
                            y=0.7 * max(lq_finite),
                            text=f"∫Lq dλ = {integral:.1f}",
                            showarrow=False,
                            bgcolor="white",
                            font=dict(size=14)
                        )
                        st.plotly_chart(fig_integ, use_container_width=True)
                else:
                    st.warning("No hay suficientes datos válidos para calcular la integral")
        else:
            st.warning("No se pudieron generar datos graficables para $L_q$ vs λ con los parámetros actuales.")
    else:
        st.warning("μ y c deben ser mayores que 0 para generar el gráfico.")

elif opcion_menu == "Simulación de Eventos (Simular)":
    # La lógica de UI y ejecución de la simulación está encapsulada en montecarlo.py
    montecarlo.display_montecarlo_section()

# Pie de página en la sidebar
st.sidebar.markdown("---")
st.sidebar.info("Proyecto: Sistema de Modelo de colas GRUPO 2")