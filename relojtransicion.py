import math
import random
import collections
import numpy as np
import matplotlib.pyplot as plt

# --- Funciones Analíticas (Estado Estacionario) ---
def calcular_P0(lambda_val, mu_val, c_val):
    """
    Calcula P0 (Probabilidad de 0 clientes en el sistema) para una cola M/M/c.
    """
    rho = lambda_val / (c_val * mu_val)
    if rho >= 1:
        return 0

    sumatoria = 0
    for n in range(c_val):
        sumatoria += (lambda_val / mu_val)**n / math.factorial(n)
    
    sumatoria += ((lambda_val / mu_val)**c_val) / (math.factorial(c_val) * (1 - rho))
    P0 = 1 / sumatoria
    return P0

def calcular_Lq(lambda_val, mu_val, c_val):
    """
    Calcula Lq (Longitud promedio de la cola) para una cola M/M/c.
    """
    rho = lambda_val / (c_val * mu_val)
    if rho >= 1:
        return float('nan')

    P0 = calcular_P0(lambda_val, mu_val, c_val)
    numerador = (lambda_val / mu_val)**c_val * lambda_val * mu_val
    denominator = math.factorial(c_val - 1) * (c_val * mu_val - lambda_val)**2
    Lq = (numerador / denominator) * P0
    return Lq

# --- Función de Simulación de Eventos Discretos ---
def simular_cola_mmk_completa(tasa_llegada_hr, tasa_servicio_servidor_hr, num_servidores, tiempo_simulacion_horas):
    """
    Simula una cola M/M/k usando simulación de eventos discretos.
    Devuelve una lista de los tiempos de los clientes en el sistema.
    """
    # Convertir las tasas de horas a minutos para la simulación
    tasa_llegada_min = tasa_llegada_hr / 60
    tasa_servicio_servidor_min = tasa_servicio_servidor_hr / 60
    tiempo_simulacion_min = tiempo_simulacion_horas * 60

    # Inicializar variables de la simulación
    tiempo_actual = 0.0

    # Lista de eventos: Una tupla donde cada evento es:
    # (tiempo_evento, tipo_evento, id_cliente, id_servidor_si_es_salida)
    # tipo_evento: 1 para llegada, 2 para salida
    # id_servidor_si_es_salida: -1 para llegadas, o el ID del servidor (0 a num_servidores-1) para salidas
    lista_eventos = []

    contador_id_cliente = 0  # Para asignar IDs únicos a los clientes

    # Cola de clientes esperando servicio (almacena id_cliente)
    clientes_en_cola = collections.deque()

    # Estado de cada servidor (almacena el id_cliente que está siendo atendido, o -1 si está libre)
    estado_servidores = [-1] * num_servidores

    # Almacenamiento de datos del recorrido de cada cliente
    tiempos_llegada_clientes = {}
    tiempos_salida_clientes = {}

    # Lista para almacenar el tiempo total en el sistema para los clientes que terminaron
    tiempos_en_sistema_clientes = []

    # Programar el primer evento de llegada
    tiempo_proxima_llegada = random.expovariate(tasa_llegada_min)
    lista_eventos.append((tiempo_proxima_llegada, 1, contador_id_cliente, -1))
    
    # Ordenar la lista de eventos por tiempo
    lista_eventos.sort()

    # Bucle principal de la simulación
    while lista_eventos and tiempo_actual < tiempo_simulacion_min:
        tiempo_evento, tipo_evento, id_cliente_actual, id_servidor_arg = lista_eventos.pop(0)

        if tiempo_evento > tiempo_simulacion_min:
            break

        tiempo_actual = tiempo_evento  # Avanzar el reloj

        if tipo_evento == 1:  # Evento de Llegada
            tiempos_llegada_clientes[id_cliente_actual] = tiempo_actual

            servidor_libre_encontrado = False
            for i in range(num_servidores):
                if estado_servidores[i] == -1:  # Servidor libre
                    estado_servidores[i] = id_cliente_actual
                    tiempo_salida = tiempo_actual + random.expovariate(tasa_servicio_servidor_min)
                    lista_eventos.append((tiempo_salida, 2, id_cliente_actual, i))
                    servidor_libre_encontrado = True
                    break

            if not servidor_libre_encontrado:
                clientes_en_cola.append(id_cliente_actual)

            contador_id_cliente += 1
            tiempo_proxima_llegada = tiempo_actual + random.expovariate(tasa_llegada_min)
            lista_eventos.append((tiempo_proxima_llegada, 1, contador_id_cliente, -1))
            lista_eventos.sort() # Reordenar después de añadir nuevo evento

        elif tipo_evento == 2:  # Evento de Salida
            id_servidor_que_salio = id_servidor_arg
            tiempos_salida_clientes[id_cliente_actual] = tiempo_actual

            if id_cliente_actual in tiempos_llegada_clientes:
                tiempo_en_sistema = tiempos_salida_clientes[id_cliente_actual] - tiempos_llegada_clientes[id_cliente_actual]
                tiempos_en_sistema_clientes.append(tiempo_en_sistema)

            estado_servidores[id_servidor_que_salio] = -1  # Liberar servidor

            if clientes_en_cola:
                id_proximo_cliente_cola = clientes_en_cola.popleft()
                
                estado_servidores[id_servidor_que_salio] = id_proximo_cliente_cola
                tiempo_salida = tiempo_actual + random.expovariate(tasa_servicio_servidor_min)
                lista_eventos.append((tiempo_salida, 2, id_proximo_cliente_cola, id_servidor_que_salio))
                lista_eventos.sort() # Reordenar después de añadir nuevo evento

    return tiempos_en_sistema_clientes

# --- Ejecución Principal para el Problema Hipotético ---
if __name__ == "__main__":
    # Parámetros del problema
    lambda_val = 10  # Tasa de llegada (clientes por hora) - Se mantiene fija para los escenarios
    mu_val = 4       # Tasa de servicio por servidor (clientes por hora) - Se mantiene fija

    # --- Escenario 1: Situación Actual (3 Cajeros) ---
    print('--- Escenario 1: Situación Actual en "Mi Ahorro" (3 Cajeros) ---')
    c_actual = 3  # Número de servidores en la situación actual

    # Validar estabilidad analítica
    if lambda_val >= c_actual * mu_val:
        print(f'¡Advertencia! El sistema es inestable con {c_actual} cajeros. La cola crecerá indefinidamente.')
    else:
        # Cálculos Analíticos
        P0_actual = calcular_P0(lambda_val, mu_val, c_actual)
        Lq_actual = calcular_Lq(lambda_val, mu_val, c_actual)
        Wq_actual_horas = Lq_actual / lambda_val
        Wq_actual_minutos = Wq_actual_horas * 60

        print(f'Cálculos Analíticos (Estado Estacionario) para {c_actual} Cajeros:')
        print(f'  Probabilidad de 0 clientes en el sistema (P0): {P0_actual:.4f}')
        print(f'  Longitud promedio de la cola (Lq): {Lq_actual:.4f} clientes')
        print(f'  Tiempo promedio de espera en la cola (Wq): {Wq_actual_minutos:.2f} minutos')

        # Simulación de Eventos Discretos para Escenario 1
        tiempo_simulacion_horas = 1000  # Suficiente tiempo para obtener datos estables
        print(f'\nEjecutando Simulación para {c_actual} Cajeros (Duración: {tiempo_simulacion_horas} horas)...')
        tiempos_en_sistema_esc1 = simular_cola_mmk_completa(lambda_val, mu_val, c_actual, tiempo_simulacion_horas)

        if tiempos_en_sistema_esc1:
            plt.figure()
            plt.hist(tiempos_en_sistema_esc1, bins='auto', edgecolor='black', facecolor=[0.7, 0.9, 0.7])
            plt.xlabel('Tiempo en el Sistema (Minutos)')
            plt.ylabel('Frecuencia')
            title_sim1 = (f'Distribución del Tiempo de Clientes en el Sistema M/M/{c_actual}\n'
                          f'(Actual: $\\lambda$={lambda_val:.0f} cl/hr, $\\mu$={mu_val:.0f} cl/hr/serv, k={c_actual} serv)\n'
                          f'Duración de Simulación: {tiempo_simulacion_horas} horas')
            plt.title(title_sim1)
            plt.grid(True)
            print(f'  Tiempo promedio en el sistema (simulado): {np.mean(tiempos_en_sistema_esc1):.2f} minutos (basado en {len(tiempos_en_sistema_esc1)} clientes)')
        else:
            print('  La simulación para el escenario 1 no generó datos de clientes completados. Considera aumentar el tiempo de simulación.')

    # --- Escenario 2: Propuesta de Mejora (4 Cajeros) ---
    print('\n--- Escenario 2: Propuesta de Mejora (4 Cajeros) ---')
    c_propuesta = 4  # Número de servidores en la propuesta

    # Validar estabilidad analítica
    if lambda_val >= c_propuesta * mu_val:
        print(f'¡Advertencia! El sistema es inestable con {c_propuesta} cajeros. La cola crecerá indefinidamente.')
    else:
        # Cálculos Analíticos
        P0_propuesta = calcular_P0(lambda_val, mu_val, c_propuesta)
        Lq_propuesta = calcular_Lq(lambda_val, mu_val, c_propuesta)
        Wq_propuesta_horas = Lq_propuesta / lambda_val
        Wq_propuesta_minutos = Wq_propuesta_horas * 60

        print(f'Cálculos Analíticos (Estado Estacionario) para {c_propuesta} Cajeros:')
        print(f'  Probabilidad de 0 clientes en el sistema (P0): {P0_propuesta:.4f}')
        print(f'  Longitud promedio de la cola (Lq): {Lq_propuesta:.4f} clientes')
        print(f'  Tiempo promedio de espera en la cola (Wq): {Wq_propuesta_minutos:.2f} minutos')

        # Simulación de Eventos Discretos para Escenario 2
        print(f'\nEjecutando Simulación para {c_propuesta} Cajeros (Duración: {tiempo_simulacion_horas} horas)...')
        tiempos_en_sistema_esc2 = simular_cola_mmk_completa(lambda_val, mu_val, c_propuesta, tiempo_simulacion_horas)

        if tiempos_en_sistema_esc2:
            plt.figure()
            plt.hist(tiempos_en_sistema_esc2, bins='auto', edgecolor='black', facecolor=[0.7, 0.7, 0.9])
            plt.xlabel('Tiempo en el Sistema (Minutos)')
            plt.ylabel('Frecuencia')
            title_sim2 = (f'Distribución del Tiempo de Clientes en el Sistema M/M/{c_propuesta}\n'
                          f'(Propuesta: $\\lambda$={lambda_val:.0f} cl/hr, $\\mu$={mu_val:.0f} cl/hr/serv, k={c_propuesta} serv)\n'
                          f'Duración de Simulación: {tiempo_simulacion_horas} horas')
            plt.title(title_sim2)
            plt.grid(True)
            print(f'  Tiempo promedio en el sistema (simulado): {np.mean(tiempos_en_sistema_esc2):.2f} minutos (basado en {len(tiempos_en_sistema_esc2)} clientes)')
        else:
            print('  La simulación para el escenario 2 no generó datos de clientes completados. Considera aumentar el tiempo de simulación.')

    # --- Comparación y Conclusiones para la Gerencia ---
    print('\n--- Comparación y Recomendación para la Gerencia ---')
    if lambda_val < c_actual * mu_val and lambda_val < c_propuesta * mu_val:  # Asegurarse de que ambos sistemas sean estables
        print('\nResumen de Métricas Clave:')
        print('--------------------------------------------------------------------------------')
        print('| Escenario         | P0      | Lq (clientes) | Wq (minutos) | T_Sistema (min) Simulado |')
        print('--------------------------------------------------------------------------------')
        print(f'| Actual (k={c_actual}) | {P0_actual:.4f}  | {Lq_actual:.4f}         | {Wq_actual_minutos:.2f}         | {np.mean(tiempos_en_sistema_esc1):.2f}                       |')
        print(f'| Propuesta (k={c_propuesta})| {P0_propuesta:.4f}  | {Lq_propuesta:.4f}         | {Wq_propuesta_minutos:.2f}         | {np.mean(tiempos_en_sistema_esc2):.2f}                       |')
        print('--------------------------------------------------------------------------------')

        print('\n**Recomendación para la Gerencia:**')
        print('Basándonos en este análisis, **aumentar el número de cajeros de 3 a 4 mejora significativamente** los indicadores clave de rendimiento:')
        print(f'- La **Longitud promedio de la cola (Lq)** se reduce de {Lq_actual:.2f} a {Lq_propuesta:.2f} clientes.')
        print(f'- El **Tiempo promedio de espera en la cola (Wq)** disminuye drásticamente de {Wq_actual_minutos:.2f} minutos a {Wq_propuesta_minutos:.2f} minutos.')
        print(f'- El **Tiempo promedio en el sistema (simulado)** muestra una reducción sustancial de {np.mean(tiempos_en_sistema_esc1):.2f} minutos a {np.mean(tiempos_en_sistema_esc2):.2f} minutos.')
        print('\nEsta propuesta resultaría en una experiencia de cliente mucho más eficiente y satisfactoria, reduciendo los tiempos de espera y mejorando la calidad general del servicio.')

    # Mostrar las gráficas
    plt.show()