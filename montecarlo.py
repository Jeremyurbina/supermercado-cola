import streamlit as st
import numpy as np
import plotly.express as px
import heapq
import pandas as pd
import mmc  # Para comparar con valores teóricos
import time

class Event:
    """Clase para representar un evento en la simulación."""
    def __init__(self, time, event_type, customer_id=None, server_id=None):
        self.time = time
        self.event_type = event_type  # "LLEGADA" o "PARTIDA"
        self.customer_id = customer_id
        self.server_id = server_id

    def __lt__(self, other):
        return self.time < other.time

def run_simulation(lambda_llegada, mu_servicio, c_servidores, max_clientes, progress_bar=None):
    """
    Ejecuta una simulación de eventos discretos para un sistema M/M/c.
    
    Retorna:
        - Métricas simuladas (Lq, Wq, L, W, Rho)
        - Tiempos individuales de espera y en sistema
        - Historial de longitud de cola
    """
    # Validación de parámetros
    if lambda_llegada <= 0 or mu_servicio <= 0 or c_servidores <= 0:
        st.warning("Lambda, Mu y C deben ser positivos para la simulación.")
        return {}, [], [], []

    # Inicialización
    tiempo_actual = 0.0
    num_clientes_procesados = 0
    next_customer_id = 0
    
    # Estado del sistema
    servidores = [None] * c_servidores  # (tiempo_fin_servicio, customer_id)
    cola = []  # (tiempo_llegada, customer_id)
    eventos = []
    
    # Estadísticas
    tiempos_llegada = {}
    tiempos_inicio_servicio = {}
    tiempos_salida = {}
    tiempos_espera_individuales = []
    tiempos_en_sistema_individuales = []
    log_longitud_cola = []
    
    # Métricas acumuladas
    area_bajo_lq = 0.0
    area_bajo_ls = 0.0
    tiempo_ultimo_evento = 0.0
    tiempo_servidores_ocupados = 0.0
    
    # Programar primera llegada
    heapq.heappush(eventos, Event(tiempo_actual + np.random.exponential(1.0/lambda_llegada), 
                                 "LLEGADA", next_customer_id))
    next_customer_id += 1
    
    # Bucle de simulación
    while num_clientes_procesados < max_clientes and eventos:
        evento = heapq.heappop(eventos)
        tiempo_previo = tiempo_actual
        tiempo_actual = evento.time
        
        # Actualizar áreas acumuladas
        delta_t = tiempo_actual - tiempo_previo
        area_bajo_lq += len(cola) * delta_t
        area_bajo_ls += (len(cola) + sum(1 for s in servidores if s is not None)) * delta_t
        tiempo_servidores_ocupados += sum(1 for s in servidores if s is not None) * delta_t
        
        # Registrar longitud de cola
        log_longitud_cola.append((tiempo_actual, len(cola)))
        
        if evento.event_type == "LLEGADA":
            # Registrar llegada
            tiempos_llegada[evento.customer_id] = tiempo_actual
            
            # Programar próxima llegada
            if next_customer_id < max_clientes * 1.5:
                heapq.heappush(eventos, Event(
                    tiempo_actual + np.random.exponential(1.0/lambda_llegada),
                    "LLEGADA",
                    next_customer_id
                ))
                next_customer_id += 1
            
            # Buscar servidor libre
            servidor_libre = next((i for i, s in enumerate(servidores) if s is None), None)
            
            if servidor_libre is not None:
                # Iniciar servicio inmediatamente
                tiempo_servicio = np.random.exponential(1.0/mu_servicio)
                tiempo_fin_servicio = tiempo_actual + tiempo_servicio
                servidores[servidor_libre] = (tiempo_fin_servicio, evento.customer_id)
                tiempos_inicio_servicio[evento.customer_id] = tiempo_actual
                
                # Programar partida
                heapq.heappush(eventos, Event(
                    tiempo_fin_servicio,
                    "PARTIDA",
                    evento.customer_id,
                    servidor_libre
                ))
            else:
                # Ir a la cola
                cola.append((tiempo_actual, evento.customer_id))
        
        elif evento.event_type == "PARTIDA":
            num_clientes_procesados += 1
            # Actualizar barra de progreso
            if progress_bar and num_clientes_procesados % (max_clientes // 10) == 0:
                progress = num_clientes_procesados / max_clientes
                progress_bar.progress(progress, f"Procesados: {num_clientes_procesados}/{max_clientes}")
            
            # Registrar salida
            tiempos_salida[evento.customer_id] = tiempo_actual
            servidores[evento.server_id] = None
            
            # Calcular tiempos para este cliente
            if evento.customer_id in tiempos_llegada:
                tiempo_llegada = tiempos_llegada[evento.customer_id]
                tiempo_salida = tiempo_actual
                tiempo_en_sistema = tiempo_salida - tiempo_llegada
                tiempos_en_sistema_individuales.append(tiempo_en_sistema)
                
                if evento.customer_id in tiempos_inicio_servicio:
                    tiempo_espera = tiempos_inicio_servicio[evento.customer_id] - tiempo_llegada
                    tiempos_espera_individuales.append(tiempo_espera)
            
            # Atender siguiente cliente en cola
            if cola:
                tiempo_llegada_cola, cliente_id = cola.pop(0)
                tiempo_servicio = np.random.exponential(1.0/mu_servicio)
                tiempo_fin_servicio = tiempo_actual + tiempo_servicio
                servidores[evento.server_id] = (tiempo_fin_servicio, cliente_id)
                tiempos_inicio_servicio[cliente_id] = tiempo_actual
                
                # Programar partida
                heapq.heappush(eventos, Event(
                    tiempo_fin_servicio,
                    "PARTIDA",
                    cliente_id,
                    evento.server_id
                ))
    
    # Calcular métricas finales
    tiempo_total = tiempo_actual if tiempo_actual > 0 else 1e-9
    Lq_sim = area_bajo_lq / tiempo_total
    L_sim = area_bajo_ls / tiempo_total
    Wq_sim = np.mean(tiempos_espera_individuales) if tiempos_espera_individuales else 0
    W_sim = np.mean(tiempos_en_sistema_individuales) if tiempos_en_sistema_individuales else 0
    Rho_sim = tiempo_servidores_ocupados / (c_servidores * tiempo_total)
    lambda_efectiva = num_clientes_procesados / tiempo_total
    
    metricas = {
        "Lq_sim": Lq_sim,
        "Wq_sim": Wq_sim,
        "L_sim": L_sim,
        "W_sim": W_sim,
        "Rho_sim": Rho_sim,
        "lambda_efectiva": lambda_efectiva,
        "tiempo_total_sim": tiempo_total,
        "clientes_simulados": num_clientes_procesados
    }
    
    return metricas, tiempos_espera_individuales, tiempos_en_sistema_individuales, log_longitud_cola

def display_montecarlo_section():
    st.title("Simulación de Eventos Discretos (Monte Carlo) para M/M/c")
    st.markdown("""
    Ejecuta una simulación de eventos discretos para observar el comportamiento dinámico del sistema.
    Compara los resultados con los valores teóricos.
    """)
    
    st.sidebar.header("Parámetros de Simulación")
    lambda_sim = st.sidebar.slider("Tasa de llegada λ (sim):", 0.1, 50.0, 5.0, 0.1, key="lambda_sim")
    mu_sim = st.sidebar.slider("Tasa de servicio μ (sim):", 0.1, 20.0, 2.0, 0.1, key="mu_sim")
    c_sim = st.sidebar.number_input("Número de servidores c (sim):", 1, 15, 3, 1, key="c_sim")
    max_clientes_sim = st.sidebar.number_input("Número de clientes a simular:", 100, 50000, 1000, 100, key="max_clientes_sim")
    
    # Botón para ejecutar simulación
    if st.sidebar.button("Ejecutar Simulación", key="run_sim_button"):
        progress_bar = st.progress(0, "Preparando simulación...")
        start_time = time.time()
        
        # Ejecutar simulación
        metricas, tiempos_espera, tiempos_sistema, log_cola = run_simulation(
            lambda_sim, mu_sim, c_sim, max_clientes_sim, progress_bar
        )
        
        # Calcular valores teóricos para comparación
        rho_teo = mmc.calcular_rho(lambda_sim, mu_sim, c_sim)
        p0_teo = mmc.calcular_p0(lambda_sim, mu_sim, c_sim, rho_teo)
        lq_teo = mmc.calcular_lq(lambda_sim, mu_sim, c_sim)
        wq_teo = mmc.calcular_wq(lq_teo, lambda_sim) if lambda_sim > 0 else 0
        w_teo = mmc.calcular_w(wq_teo, mu_sim) if mu_sim > 0 else 0
        l_teo = mmc.calcular_l(lq_teo, lambda_sim, mu_sim)
        
        # Mostrar resultados
        st.subheader("Resultados de la Simulación")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Lq (simulada)", f"{metricas['Lq_sim']:.3f}", f"Teórico: {lq_teo:.3f}")
            st.metric("Wq (simulada)", f"{metricas['Wq_sim']:.3f} horas", f"Teórico: {wq_teo:.3f} horas")
            st.metric("Utilización ρ (simulada)", f"{metricas['Rho_sim']:.3f}", f"Teórico: {rho_teo:.3f}")
        
        with col2:
            st.metric("L (simulada)", f"{metricas['L_sim']:.3f}", f"Teórico: {l_teo:.3f}")
            st.metric("W (simulada)", f"{metricas['W_sim']:.3f} horas", f"Teórico: {w_teo:.3f} horas")
            st.metric("λ efectiva", f"{metricas['lambda_efectiva']:.3f}", f"Configurada: {lambda_sim:.3f}")
        
        st.markdown(f"**Clientes simulados:** {metricas['clientes_simulados']}")
        st.markdown(f"**Tiempo total simulado:** {metricas['tiempo_total_sim']:.2f} horas")
        st.markdown(f"**Tiempo de ejecución:** {time.time() - start_time:.2f} segundos")
        
        # Gráficos
        st.subheader("Distribuciones y Comportamiento")
        
        if tiempos_espera:
            fig1 = px.histogram(
                x=tiempos_espera,
                nbins=50,
                title="Distribución de Tiempos de Espera en Cola (Wq)",
                labels={"x": "Tiempo en cola (horas)"}
            )
            fig1.add_vline(x=wq_teo, line_dash="dash", line_color="red", annotation_text=f"Teórico: {wq_teo:.3f}")
            st.plotly_chart(fig1, use_container_width=True)
        
        if tiempos_sistema:
            fig2 = px.histogram(
                x=tiempos_sistema,
                nbins=50,
                title="Distribución de Tiempos en Sistema (W)",
                labels={"x": "Tiempo en sistema (horas)"}
            )
            fig2.add_vline(x=w_teo, line_dash="dash", line_color="red", annotation_text=f"Teórico: {w_teo:.3f}")
            st.plotly_chart(fig2, use_container_width=True)
        
        if log_cola:
            tiempos, longitudes = zip(*log_cola)
            fig3 = px.line(
                x=tiempos,
                y=longitudes,
                title="Longitud de Cola a lo Largo del Tiempo",
                labels={"x": "Tiempo (horas)", "y": "Clientes en cola"}
            )
            fig3.add_hline(y=lq_teo, line_dash="dash", line_color="red", annotation_text=f"Lq teórico: {lq_teo:.3f}")
            st.plotly_chart(fig3, use_container_width=True)
        
        progress_bar.progress(1.0, "Simulación completada!")
    else:
        st.info("Configura los parámetros y haz clic en 'Ejecutar Simulación' para comenzar.")