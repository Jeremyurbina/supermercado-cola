import math

# Valor máximo seguro para evitar desbordamientos
MAX_SAFE_VALUE = 1e300
MIN_SAFE_VALUE = 1e-300

def calcular_rho(lambda_llegadas, mu_servicio, c_servidores):
    """Calcula la utilización del sistema (rho)."""
    if c_servidores <= 0 or mu_servicio <= 0:
        return float('inf') 
    return lambda_llegadas / (c_servidores * mu_servicio)

def calcular_p0(lambda_llegadas, mu_servicio, c_servidores, rho):
    """Calcula P0 con técnica de normalización para evitar desbordamientos."""
    # Casos especiales y validaciones
    if c_servidores <= 0 or mu_servicio <= 0 or rho >= 1.0:
        return 0.0
    if lambda_llegadas == 0:
        return 1.0

    c = int(c_servidores)
    ratio = lambda_llegadas / mu_servicio

    # Inicializar lista de términos
    terms = []
    max_term = 0.0

    # Término para n=0
    term_n = 1.0  # (ratio^0) / 0! = 1
    terms.append(term_n)
    max_term = max(max_term, term_n)

    # Términos para n=1 a n=c-1
    for n in range(1, c):
        term_n = terms[n-1] * ratio / n
        terms.append(term_n)
        if term_n > max_term:
            max_term = term_n

    # Término para n>=c
    if c == 0:
        term_c = 0.0
    else:
        term_c = terms[-1] * ratio / c  # (ratio^c) / c!
    term_c /= (1 - rho)
    terms.append(term_c)
    if term_c > max_term:
        max_term = term_c

    # Escalar términos si es necesario
    if max_term > MAX_SAFE_VALUE:
        scale_factor = MAX_SAFE_VALUE / max_term
        scaled_terms = [t * scale_factor for t in terms]
        sum_terms = sum(scaled_terms)
        p0 = 1.0 / (sum_terms / scale_factor)
    else:
        sum_terms = sum(terms)
        p0 = 1.0 / sum_terms

    return p0

def calcular_lq(lambda_llegadas, mu_servicio, c_servidores):
    """Calcula Lq con manejo de casos extremos."""
    # Validaciones iniciales
    if c_servidores <= 0 or mu_servicio <= 0:
        return float('inf')
    
    rho = calcular_rho(lambda_llegadas, mu_servicio, c_servidores)
    if rho >= 1.0:
        return float('inf')
    
    # Caso especial: sin llegadas
    if lambda_llegadas == 0:
        return 0.0
    
    # Aproximación para rho muy cercano a 1
    if rho > 0.999:
        return (rho * c_servidores) / (1 - rho)
    
    # Calcular P0
    p0 = calcular_p0(lambda_llegadas, mu_servicio, c_servidores, rho)
    
    # Manejo de P0 muy pequeño
    if p0 < 1e-15:
        return (rho**(c_servidores + 1)) / (c_servidores * (1 - rho)**2)
    
    ratio = lambda_llegadas / mu_servicio
    c = int(c_servidores)
    
    try:
        # Calcular numerador: p0 * (ratio)^c * rho
        term1 = p0
        term2 = ratio**c
        term3 = rho
        
        # Manejar términos grandes
        if term2 > MAX_SAFE_VALUE:
            scale_factor = MAX_SAFE_VALUE / term2
            term1 *= scale_factor
            term2 = MAX_SAFE_VALUE
        else:
            scale_factor = 1.0
            
        numerador = term1 * term2 * term3
        
        # Calcular denominador
        denom1 = math.factorial(c)
        denom2 = (1.0 - rho)**2
        denominador = denom1 * denom2
        
        # Manejar denominador muy pequeño
        if denominador < MIN_SAFE_VALUE:
            return float('inf')
            
        lq = numerador / denominador
        
        # Ajustar escala
        return lq / scale_factor
        
    except (OverflowError, ValueError):
        # Aproximación de respaldo
        return (rho * c_servidores) / (1 - rho)

def calcular_wq(lq, lambda_llegadas):
    """Calcula el tiempo promedio de espera en la cola (Wq)."""
    if lambda_llegadas <= 0: 
        return 0.0
    if math.isinf(lq):
        return float('inf')
    return lq / lambda_llegadas

def calcular_w(wq, mu_servicio):
    """Calcula el tiempo promedio total en el sistema (W)."""
    if mu_servicio <= 0: 
        return float('inf')
    if math.isinf(wq):
        return float('inf')
    return wq + (1.0 / mu_servicio)

def calcular_l(lq, lambda_llegadas, mu_servicio):
    """Calcula el número promedio de clientes en el sistema (L)."""
    if mu_servicio <= 0:
        return float('inf')
    if math.isinf(lq):
        return float('inf')
    if lambda_llegadas == 0:
        return lq
        
    return lq + (lambda_llegadas / mu_servicio)

def calcular_probabilidad_espera(lambda_llegadas, mu_servicio, c_servidores, p0, rho):
    """Calcula Pw con manejo numérico mejorado."""
    # Casos especiales
    if c_servidores <= 0:
        return 1.0
    if mu_servicio <= 0:
        return 1.0
    if lambda_llegadas < 0:
        return 0.0
    if lambda_llegadas == 0:
        return 0.0
    if rho >= 1.0:
        return 1.0

    c = int(c_servidores)
    
    try:
        # Calcular término (λ/μ)^c
        ratio_lm = lambda_llegadas / mu_servicio
        term_lm = ratio_lm**c
        
        # Calcular denominador
        denom_factor = math.factorial(c) * (1.0 - rho)
        
        # Manejar casos extremos
        if denom_factor < MIN_SAFE_VALUE:
            return 1.0
            
        # Calcular Pw
        numerador = p0 * term_lm
        prob_espera = numerador / denom_factor
        
        # Asegurar rango válido
        return max(0.0, min(prob_espera, 1.0))
        
    except (OverflowError, ValueError):
        # Aproximación para alta carga
        return min(1.0, rho * c_servidores)