# montecarlo.py
import streamlit as st
import numpy as np
import plotly.express as px
import heapq # Para la cola de eventos prioritarios

class Event:
    """Clase para representar un evento en la simulación."""
    def __init__(self, time, event_type, customer_id=None):
        self.time = time  # Tiempo en que ocurre el evento
        self.event_type = event_type  # "LLEGADA" o "PARTIDA"
        self.customer_id = customer_id # ID del cliente asociado (para partidas)

    # Comparador para la cola de prioridad (heapq)
    def __lt__(self, other):
        return self.time < other.time

def run_simulation(lambda_llegada, mu_servicio, c_servidores, max_clientes, show_progress=True):
    """
    Ejecuta una simulación de eventos discretos para un sistema M/M/c.
    Retorna:
        - Métricas simuladas (Lq, Wq, L, W, Rho_sim).
        - Listas de tiempos de espera individuales, tiempos en sistema.
        - Registros de longitud de cola a lo largo del tiempo.
    """
    if lambda_llegada <= 0 or mu_servicio <= 0 or c_servidores <= 0:
        st.warning("Lambda, Mu y C deben ser positivos para la simulación.")
        return {}, [], [], []

    # Inicialización
    tiempo_actual = 0.0
    num_llegadas = 0
    num_partidas = 0
    num_clientes_en_cola = 0
    num_clientes_en_sistema = 0
    
    clientes_en_servicio = [None] * c_servidores # None si el servidor está libre, sino tiempo de partida
    cola_de_clientes = [] # (tiempo_llegada, customer_id)
    
    eventos = [] # Cola de eventos (min-heap)
    heapq.heappush(eventos, Event(np.random.exponential(1.0/lambda_llegada), "LLEGADA", 0))
    
    # Estadísticas
    total_tiempo_espera_cola = 0.0
    total_tiempo_en_sistema = 0.0
    total_tiempo_servidores_ocupados = 0.0 # Para calcular Rho
    tiempos_espera_individuales = []
    tiempos_en_sistema_individuales = []
    log_longitud_cola = [] # (tiempo, longitud_cola)
    
    next_customer_id = 1

    # Bucle de simulación
    while num_partidas < max_clientes :
        if not eventos: # No más eventos programados
            break 
            
        evento_actual = heapq.heappop(eventos)
        tiempo_previo = tiempo_actual
        tiempo_actual = evento_actual.time
        
        # Actualizar tiempo ocupado de servidores
        delta_tiempo = tiempo_actual - tiempo_previo
        servidores_ocupados_ahora = sum(1 for s_time in clientes_en_servicio if s_time is not None)
        total_tiempo_servidores_ocupados += servidores_ocupados_ahora * delta_tiempo

        if evento_actual.event_type == "LLEGADA":
            num_llegadas += 1
            num_clientes_en_sistema += 1
            
            # Programar la siguiente llegada
            if num_llegadas < max_clientes * 1.5 : # Evitar demasiadas llegadas si la simulación se detiene por partidas
                 tiempo_proxima_llegada = tiempo_actual + np.random.exponential(1.0/lambda_llegada)
                 heapq.heappush(eventos, Event(tiempo_proxima_llegada, "LLEGADA", next_customer_id))
                 next_customer_id += 1

            servidor_libre_idx = -1
            for i in range(c_servidores):
                if clientes_en_servicio[i] is None:
                    servidor_libre_idx = i
                    break
            
            if servidor_libre_idx != -1: # Hay servidor libre
                tiempo_servicio = np.random.exponential(1.0/mu_servicio)
                tiempo_partida = tiempo_actual + tiempo_servicio
                clientes_en_servicio[servidor_libre_idx] = tiempo_partida # Marca servidor como ocupado hasta t_partida
                heapq.heappush(eventos, Event(tiempo_partida, "PARTIDA", evento_actual.customer_id))
                # Cliente no esperó en cola
                tiempos_espera_individuales.append(0.0)
            else: # Todos los servidores ocupados, cliente a la cola
                num_clientes_en_cola += 1
                cola_de_clientes.append((tiempo_actual, evento_actual.customer_id)) # (tiempo_llegada_a_cola, id)
        
        elif evento_actual.event_type == "PARTIDA":
            num_partidas += 1
            num_clientes_en_sistema -= 1
            
            # Registrar tiempo en sistema para el cliente que parte (necesitaría su tiempo de llegada original)
            # Esto requiere un seguimiento más detallado de cada cliente.
            # Por ahora, calculamos Wq promedio y luego W promedio.
            
            # Marcar el servidor como libre
            servidor_que_termino = -1
            for i in range(c_servidores):
                if clientes_en_servicio[i] is not None and abs(clientes_en_servicio[i] - tiempo_actual) < 1e-9 : # Tiempo de partida coincide
                    clientes_en_servicio[i] = None
                    servidor_que_termino = i
                    break
            
            if cola_de_clientes: # Hay clientes esperando
                num_clientes_en_cola -= 1
                tiempo_llegada_a_cola_cliente, id_cliente_cola = cola_de_clientes.pop(0)
                tiempo_espera = tiempo_actual - tiempo_llegada_a_cola_cliente
                total_tiempo_espera_cola += tiempo_espera
                tiempos_espera_individuales.append(tiempo_espera)
                
                tiempo_servicio = np.random.exponential(1.0/mu_servicio)
                tiempo_proxima_partida = tiempo_actual + tiempo_servicio
                if servidor_que_termino != -1: # Usar el servidor que acaba de quedar libre
                    clientes_en_servicio[servidor_que_termino] = tiempo_proxima_partida
                    heapq.heappush(eventos, Event(tiempo_proxima_partida, "PARTIDA", id_cliente_cola))
                else: # No debería pasar si un servidor acaba de partir, pero por si acaso
                    # Buscar otro servidor libre (si la lógica de partida es más compleja)
                    pass 

        log_longitud_cola.append((tiempo_actual, num_clientes_en_cola))
        
        if show_progress and num_partidas % (max_clientes // 10 if max_clientes >=10 else 1) == 0 and num_partidas > 0:
            st.session_state.sim_progress = num_partidas / max_clientes

    # Cálculo de métricas finales
    if num_partidas == 0: num_partidas = 1 # Evitar división por cero si no hubo partidas
    
    Lq_sim = total_tiempo_espera_cola / tiempo_actual if tiempo_actual > 0 else 0 # Lq = sum(tiempos_en_cola_de_cada_cliente) / T_total
    # Mejor: Lq = Integral(Nq(t) dt) / T. Sum(longitudes_cola_ponderadas_por_tiempo) / tiempo_total
    # O Wq_sim * lambda_efectiva
    
    Wq_sim = np.mean(tiempos_espera_individuales) if tiempos_espera_individuales else 0.0
    
    # W_sim: necesitaría tiempos en sistema individuales.
    # Aproximación W_sim = Wq_sim + 1/mu_servicio
    W_sim = Wq_sim + (1.0/mu_servicio) if mu_servicio > 0 else float('inf')
    
    # L_sim = W_sim * lambda_efectiva (lambda_efectiva = num_partidas / tiempo_actual)
    lambda_efectiva = num_partidas / tiempo_actual if tiempo_actual > 0 else 0
    L_sim = W_sim * lambda_efectiva if lambda_efectiva > 0 else 0

    Rho_sim = total_tiempo_servidores_ocupados / (c_servidores * tiempo_actual) if (c_servidores * tiempo_actual) > 0 else 0
    
    metricas_simuladas = {
        "Lq_sim": Wq_sim * lambda_efectiva, # Lq = lambda * Wq
        "Wq_sim": Wq_sim,
        "L_sim": L_sim,
        "W_sim": W_sim,
        "Rho_sim": Rho_sim,
        "lambda_efectiva": lambda_efectiva,
        "tiempo_total_sim": tiempo_actual,
        "clientes_simulados": num_partidas
    }
    
    return metricas_simuladas, tiempos_espera_individuales, log_longitud_cola


def display_montecarlo_section():
    st.title("Simulación de Eventos Discretos (Monte Carlo) para M/M/c")
    st.markdown("""
    Ejecuta una simulación de Monte Carlo para el sistema M/M/c. 
    Observa cómo las métricas simuladas se comparan con los resultados teóricos.
    """)

    st.sidebar.header("Parámetros de Simulación")
    lambda_sim = st.sidebar.slider("Tasa de llegada λ (sim):", 0.1, 50.0, 5.0, 0.1, key="lambda_sim")
    mu_sim = st.sidebar.slider("Tasa de servicio μ (sim):", 0.1, 20.0, 2.0, 0.1, key="mu_sim")
    c_sim = st.sidebar.number_input("Número de servidores c (sim):", 1, 15, 3, 1, key="c_sim")
    max_clientes_sim = st.sidebar.number_input("Número de clientes a simular (partidas):", 100, 50000, 1000, 100, key="max_clientes_sim")

    if 'sim_progress' not in st.session_state:
        st.session_state.sim_progress = 0.0

    if st.sidebar.button("Ejecutar Simulación", key="run_sim_button"):
        st.session_state.sim_progress = 0.0
        progress_bar = st.progress(0.0, "Iniciando simulación...")
        
        # Nota: La actualización de la barra de progreso desde dentro de run_simulation
        # vía st.session_state es una forma. Otra sería pasar la barra como argumento.
        metricas, tiempos_espera, log_cola = run_simulation(lambda_sim, mu_sim, c_sim, max_clientes_sim)
        
        progress_bar.progress(1.0, "Simulación completada.")

        st.subheader("Resultados de la Simulación")
        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            st.metric("Lq (simulada)", f"{metricas.get('Lq_sim', 0):.3f}")
            st.metric("Wq (simulada)", f"{metricas.get('Wq_sim', 0):.3f} horas")
            st.metric("Utilización ρ (simulada)", f"{metricas.get('Rho_sim', 0):.3f}")
        with col_sim2:
            st.metric("L (simulada)", f"{metricas.get('L_sim', 0):.3f}")
            st.metric("W (simulada)", f"{metricas.get('W_sim', 0):.3f} horas")
            st.metric("λ efectiva (simulada)", f"{metricas.get('lambda_efectiva',0):.3f}")
        
        st.markdown(f"**Clientes simulados (partidas):** {metricas.get('clientes_simulados',0)}")
        st.markdown(f"**Tiempo total de simulación:** {metricas.get('tiempo_total_sim',0):.2f} horas")


        if tiempos_espera:
            fig_wq_dist = px.histogram(x=tiempos_espera, nbins=50, labels={'x':'Tiempo de Espera en Cola (Wq)'}, title="Distribución de Tiempos de Espera (Simulación)")
            st.plotly_chart(fig_wq_dist, use_container_width=True)
        
        if log_cola:
            tiempos_log, longitudes_log = zip(*log_cola)
            fig_lq_tiempo = px.line(x=tiempos_log, y=longitudes_log, labels={'x':'Tiempo de Simulación', 'y':'Clientes en Cola'}, title="Longitud de Cola vs. Tiempo (Simulación)")
            st.plotly_chart(fig_lq_tiempo, use_container_width=True)

    # Placeholder si no se ha ejecutado
    if 'run_sim_button' not in st.session_state or not st.session_state.run_sim_button:
        st.info("Ajusta los parámetros en la barra lateral y haz clic en 'Ejecutar Simulación'.")