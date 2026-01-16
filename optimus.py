import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import time
from collections import deque

# ============================================
# CONFIGURACIÓN DE PÁGINA
# ============================================
st.set_page_config(page_title="Optimus", layout="wide")

# ============================================
# GESTIÓN DE ESTADO (PERSISTENCIA DE DATOS)
# ============================================
# Inicialización de variables de sesión para evitar reinicios indeseados
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'users_db' not in st.session_state:
    st.session_state.users_db = {'admin': 'admin'}
# Variables para persistencia de optimización
if 'opt_resultado' not in st.session_state:
    st.session_state.opt_resultado = None
if 'opt_error' not in st.session_state:
    st.session_state.opt_error = None
if 'opt_params_usados' not in st.session_state:
    st.session_state.opt_params_usados = {}

def login_user(user, password):
    if user in st.session_state.users_db:
        if st.session_state.users_db[user] == password:
            st.session_state.logged_in = True
            st.session_state.username = user
            st.success("Inicio de sesión exitoso.")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    else:
        st.error("El usuario no existe.")

def register_user(new_user, new_password, confirm_password):
    if new_user in st.session_state.users_db:
        st.error("El usuario ya existe.")
    elif new_password != confirm_password:
        st.error("Las contraseñas no coinciden.")
    elif len(new_password) < 4:
        st.warning("La contraseña debe tener al menos 4 caracteres.")
    else:
        st.session_state.users_db[new_user] = new_password
        st.success(f"Usuario '{new_user}' registrado con éxito. Por favor inicia sesión.")

# ============================================
# INTERFAZ DE AUTENTICACIÓN
# ============================================
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("Optimus")
        st.markdown("### Sistema de Optimización Panel Caving")
        
        tab_login, tab_register = st.tabs(["Iniciar Sesión", "Registrarse"])
        
        with tab_login:
            st.subheader("Acceso")
            username_input = st.text_input("Usuario", key="login_user")
            password_input = st.text_input("Contraseña", type="password", key="login_pass")
            
            if st.button("Entrar", type="primary", use_container_width=True):
                login_user(username_input, password_input)
                
            with st.expander("Información de acceso"):
                st.info("Credenciales por defecto: admin / admin")

        with tab_register:
            st.subheader("Crear Cuenta")
            new_user = st.text_input("Nuevo Usuario", key="reg_user")
            new_pass = st.text_input("Nueva Contraseña", type="password", key="reg_pass")
            conf_pass = st.text_input("Confirmar Contraseña", type="password", key="reg_conf")
            
            if st.button("Registrar", use_container_width=True):
                register_user(new_user, new_pass, conf_pass)
    
    st.stop()

# ============================================
# APLICACIÓN PRINCIPAL (OPTIMUS)
# ============================================

with st.sidebar:
    st.markdown("### Perfil de Usuario")
    st.write(f"Usuario: **{st.session_state.username}**")
    if st.button("Cerrar Sesión", type="secondary", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = ''
        # Limpiar resultados al cerrar sesión
        st.session_state.opt_resultado = None
        st.rerun()
    st.markdown("---")

st.title("Optimus")
st.markdown("##### Optimizador de Producción Diaria con Malla Tipo Teniente")

# ============================================
# CLASE PARA MANEJAR LA RED DE CALLES Y ZANJAS
# ============================================
class MallaTeniente:
    """
    Representa la malla tipo Teniente con calles paralelas y zanjas a 60°.
    Calles: paralelas entre sí, separadas por dist_entre_calles (típicamente 30m)
    Zanjas: interceptan las calles cada dist_entre_zanjas (típicamente 15m) en ángulo de 60°
    """
    
    def __init__(self, dist_entre_calles=30, dist_entre_zanjas=15):
        self.dist_entre_calles = dist_entre_calles
        self.dist_entre_zanjas = dist_entre_zanjas
        self.angulo_zanja = 60  # grados
        self.nodos = {}  # coordenadas de nodos (intersecciones)
        self.aristas = []  # conexiones entre nodos
        self.grafo = {}  # diccionario de adyacencia para BFS
        
    def construir_red(self, bateas, piques, margen=50):
        """
        Construye la red de calles y zanjas basándose en las ubicaciones 
        de bateas y piques, con un margen adicional.
        """
        # Obtener rango de coordenadas
        todas_x = list(bateas['x']) + list(piques['x'])
        todas_y = list(bateas['y']) + list(piques['y'])
        
        x_min, x_max = min(todas_x) - margen, max(todas_x) + margen
        y_min, y_max = min(todas_y) - margen, max(todas_y) + margen
        
        # Crear calles (líneas horizontales paralelas al eje X)
        num_calles = int((y_max - y_min) / self.dist_entre_calles) + 2
        calles_y = [y_min + i * self.dist_entre_calles for i in range(num_calles)]
        
        # Crear zanjas (líneas a 60° que cruzan las calles)
        # Las zanjas tienen pendiente: tan(60°) = sqrt(3) ≈ 1.732
        pendiente_zanja = np.tan(np.radians(self.angulo_zanja))
        
        # Número de zanjas necesarias para cubrir el área
        rango_x = x_max - x_min
        num_zanjas = int(rango_x / self.dist_entre_zanjas) + 2
        
        # Generar nodos en las intersecciones calle-zanja
        nodo_id = 0
        self.nodos = {}
        nodos_por_calle = {i: [] for i in range(num_calles)}
        
        for i_calle, y_calle in enumerate(calles_y):
            for i_zanja in range(num_zanjas):
                # Punto inicial de la zanja en el eje X
                x_inicio_zanja = x_min + i_zanja * self.dist_entre_zanjas
                
                # Calcular intersección de zanja con calle
                # Zanja: y - y0 = m(x - x0), donde y0 = y_min, x0 = x_inicio_zanja
                # Calle: y = y_calle
                # Intersección: y_calle = y_min + m(x - x_inicio_zanja)
                # x = x_inicio_zanja + (y_calle - y_min) / m
                
                x_interseccion = x_inicio_zanja + (y_calle - y_min) / pendiente_zanja
                
                if x_min <= x_interseccion <= x_max:
                    self.nodos[nodo_id] = {'x': x_interseccion, 'y': y_calle, 
                                           'calle': i_calle, 'zanja': i_zanja}
                    nodos_por_calle[i_calle].append(nodo_id)
                    nodo_id += 1
        
        # Crear aristas (conexiones entre nodos)
        self.aristas = []
        self.grafo = {nodo: [] for nodo in self.nodos.keys()}
        
        # Aristas a lo largo de las calles (horizontal)
        for i_calle in range(num_calles):
            nodos_calle = sorted(nodos_por_calle[i_calle], 
                                key=lambda n: self.nodos[n]['x'])
            for j in range(len(nodos_calle) - 1):
                n1, n2 = nodos_calle[j], nodos_calle[j + 1]
                distancia = self.dist_entre_zanjas / np.cos(np.radians(self.angulo_zanja))
                self.aristas.append((n1, n2, distancia))
                self.grafo[n1].append((n2, distancia))
                self.grafo[n2].append((n1, distancia))
        
        # Aristas a lo largo de las zanjas (diagonal a 60°)
        nodos_por_zanja = {}
        for nodo_id, info in self.nodos.items():
            i_zanja = info['zanja']
            if i_zanja not in nodos_por_zanja:
                nodos_por_zanja[i_zanja] = []
            nodos_por_zanja[i_zanja].append(nodo_id)
        
        for i_zanja, nodos_zanja in nodos_por_zanja.items():
            nodos_zanja = sorted(nodos_zanja, key=lambda n: self.nodos[n]['y'])
            for j in range(len(nodos_zanja) - 1):
                n1, n2 = nodos_zanja[j], nodos_zanja[j + 1]
                x1, y1 = self.nodos[n1]['x'], self.nodos[n1]['y']
                x2, y2 = self.nodos[n2]['x'], self.nodos[n2]['y']
                distancia = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                self.aristas.append((n1, n2, distancia))
                self.grafo[n1].append((n2, distancia))
                self.grafo[n2].append((n1, distancia))
    
    def nodo_mas_cercano(self, x, y):
        """Encuentra el nodo más cercano a una coordenada (x, y) (Distancia aérea)."""
        min_dist = float('inf')
        nodo_cercano = None
        for nodo_id, coords in self.nodos.items():
            dist = np.sqrt((coords['x'] - x)**2 + (coords['y'] - y)**2)
            if dist < min_dist:
                min_dist = dist
                nodo_cercano = nodo_id
        return nodo_cercano, min_dist
    
    def _distancia_teniente_local(self, x1, y1, x2, y2):
        """
        Calcula la distancia y el punto intermedio para ir de un punto arbitrario (x1, y1)
        a otro (x2, y2) respetando SOLO movimientos horizontales (Calle, 0°) y 
        diagonales (Zanja, 60°). 
        
        Resuelve el sistema vectorial: P2 = P1 + a*Vec_Calle + b*Vec_Zanja
        
        Retorna:
            distancia_total: |a| + |b|
            punto_intermedio: Coordenada (x, y) donde cambia de Calle a Zanja.
        """
        dx = x2 - x1
        dy = y2 - y1
        
        rad_zanja = np.radians(self.angulo_zanja)
        sin60 = np.sin(rad_zanja)
        cos60 = np.cos(rad_zanja)
        
        # Sistema de ecuaciones:
        # dx = a * 1 + b * cos(60)
        # dy = a * 0 + b * sin(60)  => b = dy / sin(60)
        
        if abs(sin60) < 1e-9: # Seguridad matemática
             return abs(dx) + abs(dy), (x2, y1)

        b = dy / sin60
        a = dx - (b * cos60)
        
        distancia = abs(a) + abs(b)
        
        # Calculamos el punto intermedio asumiendo ruta Calle -> Zanja
        # Caminamos 'a' en horizontal desde el origen
        x_int = x1 + a
        y_int = y1
        
        return distancia, (x_int, y_int)

    def distancia_ruta(self, x_origen, y_origen, x_destino, y_destino):
        """
        Calcula la distancia de ruta usando geometría Teniente.
        Punto -> (Calle/Zanja) -> Red -> (Calle/Zanja) -> Punto
        """
        # Encontrar nodos más cercanos en la red
        nodo_origen, _ = self.nodo_mas_cercano(x_origen, y_origen)
        nodo_destino, _ = self.nodo_mas_cercano(x_destino, y_destino)
        
        if nodo_origen is None or nodo_destino is None:
            return float('inf')
        
        # Distancia dentro de la red mallada (BFS)
        distancia_red = self._bfs_distancia(nodo_origen, nodo_destino)
        
        if distancia_red == float('inf'):
            return float('inf')
        
        # Coordenadas de los nodos de entrada y salida a la red
        nx_o, ny_o = self.nodos[nodo_origen]['x'], self.nodos[nodo_origen]['y']
        nx_d, ny_d = self.nodos[nodo_destino]['x'], self.nodos[nodo_destino]['y']

        # Distancia desde el origen real hasta el nodo de entrada (Geometría Teniente)
        dist_inicio, _ = self._distancia_teniente_local(x_origen, y_origen, nx_o, ny_o)
        
        # Distancia desde el nodo de salida hasta el destino real (Geometría Teniente)
        dist_fin, _ = self._distancia_teniente_local(nx_d, ny_d, x_destino, y_destino)
        
        return dist_inicio + distancia_red + dist_fin
    
    def _bfs_distancia(self, nodo_inicio, nodo_fin):
        """BFS para encontrar la distancia más corta entre dos nodos."""
        if nodo_inicio == nodo_fin:
            return 0
        
        visitados = set()
        cola = deque([(nodo_inicio, 0)])  # (nodo, distancia_acumulada)
        
        while cola:
            nodo_actual, dist_actual = cola.popleft()
            
            if nodo_actual in visitados:
                continue
            
            visitados.add(nodo_actual)
            
            if nodo_actual == nodo_fin:
                return dist_actual
            
            for vecino, dist_arista in self.grafo.get(nodo_actual, []):
                if vecino not in visitados:
                    cola.append((vecino, dist_actual + dist_arista))
        
        return float('inf')  # No se encontró ruta
    
    def obtener_ruta(self, x_origen, y_origen, x_destino, y_destino):
        """
        Devuelve la lista de coordenadas (x, y) de la ruta para graficar,
        incluyendo los quiebres geométricos para entrar/salir de la red.
        """
        nodo_origen, _ = self.nodo_mas_cercano(x_origen, y_origen)
        nodo_destino, _ = self.nodo_mas_cercano(x_destino, y_destino)
        
        if nodo_origen is None or nodo_destino is None:
            return []
        
        ruta_nodos = self._bfs_ruta(nodo_origen, nodo_destino)
        
        if not ruta_nodos:
            return []
        
        ruta_coords = []
        
        # 1. Tramo Inicial: Origen -> Nodo de entrada
        # Calculamos punto intermedio para respetar angulos
        nx_o, ny_o = self.nodos[nodo_origen]['x'], self.nodos[nodo_origen]['y']
        _, p_int_inicio = self._distancia_teniente_local(x_origen, y_origen, nx_o, ny_o)
        
        ruta_coords.append((x_origen, y_origen))
        ruta_coords.append(p_int_inicio)
        # El nodo exacto (nx_o, ny_o) se añade en el bucle siguiente
        
        # 2. Tramo Red: Nodos
        for nodo in ruta_nodos:
            ruta_coords.append((self.nodos[nodo]['x'], self.nodos[nodo]['y']))
        
        # 3. Tramo Final: Nodo de salida -> Destino
        nx_d, ny_d = self.nodos[nodo_destino]['x'], self.nodos[nodo_destino]['y']
        _, p_int_fin = self._distancia_teniente_local(nx_d, ny_d, x_destino, y_destino)
        
        ruta_coords.append(p_int_fin)
        ruta_coords.append((x_destino, y_destino))
        
        return ruta_coords
    
    def _bfs_ruta(self, nodo_inicio, nodo_fin):
        """BFS que devuelve la ruta de nodos."""
        if nodo_inicio == nodo_fin:
            return [nodo_inicio]
        
        visitados = set()
        cola = deque([(nodo_inicio, [nodo_inicio])])
        
        while cola:
            nodo_actual, ruta_actual = cola.popleft()
            
            if nodo_actual in visitados:
                continue
            
            visitados.add(nodo_actual)
            
            if nodo_actual == nodo_fin:
                return ruta_actual
            
            for vecino, _ in self.grafo.get(nodo_actual, []):
                if vecino not in visitados:
                    cola.append((vecino, ruta_actual + [vecino]))
        
        return []

# ============================================
# FUNCIÓN PARA CALCULAR DISTANCIAS CON MALLA
# ============================================
def calcular_distancias_con_malla(bateas, piques, malla):
    """Calcula la matriz de distancias usando la red de calles y zanjas."""
    n_bateas = len(bateas)
    n_piques = len(piques)
    distancias = np.zeros((n_bateas, n_piques))
    
    for i in range(n_bateas):
        x_b, y_b = bateas.iloc[i]['x'], bateas.iloc[i]['y']
        for j in range(n_piques):
            x_p, y_p = piques.iloc[j]['x'], piques.iloc[j]['y']
            distancias[i, j] = malla.distancia_ruta(x_b, y_b, x_p, y_p)
    
    return distancias

# ============================================
# FUNCIÓN DE CÁLCULO DE PRODUCCIÓN
# ============================================
def calcular_produccion_potencial(equipos_base):
    if 'horas_operacion_dia' not in equipos_base.columns:
        horas_operacion_dia = 24
    else:
        horas_operacion_dia = equipos_base['horas_operacion_dia'].mean()
    equipos_base['horas_operacion_dia'] = horas_operacion_dia 

    if 'disponibilidad_mecanica' not in equipos_base.columns:
        equipos_base['disponibilidad_mecanica'] = 0.85 
    if 'utilizacion_efectiva' not in equipos_base.columns:
        equipos_base['utilizacion_efectiva'] = 0.75 

    if 'ciclos_por_hora' not in equipos_base.columns:
        equipos_base['ciclos_por_hora'] = 20 + (10 * (equipos_base['capacidad_balde'].max() - equipos_base['capacidad_balde']) / equipos_base['capacidad_balde'].max())

    equipos_base['ton_hora_teorico'] = equipos_base['capacidad_balde'] * equipos_base['ciclos_por_hora']
    equipos_base['ton_hora_efectivo'] = (
        equipos_base['ton_hora_teorico'] * equipos_base['disponibilidad_mecanica'] * equipos_base['utilizacion_efectiva']
    )
    equipos_base['ton_dia_max'] = equipos_base['ton_hora_efectivo'] * equipos_base['horas_operacion_dia']
    
    return equipos_base

# ============================================
# CARGA DE DATOS DESDE EXCEL
# ============================================
@st.cache_data
def cargar_datos(archivo, dist_calles, dist_zanjas):
    """Carga las 4 hojas del Excel y construye la red de malla tipo Teniente."""
    xl = pd.ExcelFile(archivo)
    
    bateas = pd.read_excel(xl, sheet_name='Bateas')
    bateas = bateas.rename(columns={'prioridad_draw': 'prioridad_draw'}, errors='ignore')
    bateas['finos_disponibles_ton'] = bateas['tonelaje_disponible'] * bateas['ley_cu'] / 100
    
    piques = pd.read_excel(xl, sheet_name='Piques')
    piques['disponible'] = piques['estado'] == 'Operativo'
    
    equipos_base = pd.read_excel(xl, sheet_name='Equipos')
    if 'operador_experiencia' not in equipos_base.columns:
        equipos_base['operador_experiencia'] = 'Indefinido'
        
    equipos = calcular_produccion_potencial(equipos_base)
    
    historico = pd.read_excel(xl, sheet_name='Historico')
    historico['fecha'] = pd.to_datetime(historico['fecha'])
    historico['finos_cu_ton'] = historico['produccion_ton'] * historico['ley_promedio'] / 100
    
    # Construir la red de malla tipo Teniente
    malla = MallaTeniente(dist_entre_calles=dist_calles, dist_entre_zanjas=dist_zanjas)
    malla.construir_red(bateas, piques)
    
    # Calcular distancias usando la red
    distancias = calcular_distancias_con_malla(bateas, piques, malla)
    
    return bateas, piques, equipos, distancias, historico, malla

# ============================================
# FUNCIÓN DE OPTIMIZACIÓN DE ASIGNACIONES (PL)
# ============================================
def optimizar_produccion(bateas, piques, distancias, params):
    """Optimiza asignación de bateas a piques."""
    n_bateas = len(bateas)
    n_piques = len(piques)
    
    alpha = params['alpha']
    beta = params['beta']
    max_dist = params['max_distancia']
    
    piques_disp = piques[piques['disponible']].index.tolist()
    
    if len(piques_disp) == 0:
        return None, "No hay piques disponibles"
    
    n_vars = n_bateas * n_piques
    c = np.zeros(n_vars)
    
    for i in range(n_bateas):
        for j in range(n_piques):
            idx = i * n_piques + j
            ton = bateas.iloc[i]['tonelaje_disponible']
            ley = bateas.iloc[i]['ley_cu'] 
            dist = distancias[i, j]
            
            valor = alpha * ley + beta * ton 
            
            if j in piques_disp and dist <= max_dist:
                c[idx] = -valor
            else:
                c[idx] = 0
    
    A_eq = np.zeros((n_bateas, n_vars))
    b_eq = np.ones(n_bateas)
    for i in range(n_bateas):
        for j in range(n_piques):
            A_eq[i, i * n_piques + j] = 1
    
    A_ub = np.zeros((len(piques_disp), n_vars))
    b_ub = []
    for idx_p, j in enumerate(piques_disp):
        for i in range(n_bateas):
            A_ub[idx_p, i * n_piques + j] = bateas.iloc[i]['tonelaje_disponible']
        cap_disp = piques.iloc[j]['capacidad_diaria'] * (1 - piques.iloc[j]['nivel_llenado'])
        b_ub.append(cap_disp)
    
    b_ub = np.array(b_ub)
    
    bounds = [(0, 1) for _ in range(n_vars)]
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                          bounds=bounds, method='highs')
        
        if result.success:
            x_opt = result.x.reshape(n_bateas, n_piques)
            
            produccion_total = 0
            ley_ponderada_sum = 0
            finos_total = 0
            asignaciones = []
            
            for i in range(n_bateas):
                for j in range(n_piques):
                    if j in piques_disp and distancias[i, j] <= max_dist:
                        ton_asignado = bateas.iloc[i]['tonelaje_disponible'] * x_opt[i, j]
                        if ton_asignado > 0.01: 
                            ley = bateas.iloc[i]['ley_cu']
                            finos_asignado = ton_asignado * ley / 100
                            
                            produccion_total += ton_asignado
                            ley_ponderada_sum += ley * ton_asignado
                            finos_total += finos_asignado
                                                                        
                            valor_objetivo_asignado = alpha * (ley * x_opt[i, j]) + beta * ton_asignado

                            asignaciones.append({
                                'batea': bateas.iloc[i]['batea_id'],
                                'pique': piques.iloc[j]['pique_id'],
                                'tonelaje': ton_asignado,
                                'ley': ley,
                                'distancia': distancias[i, j],
                                'valor_objetivo': valor_objetivo_asignado
                            })
            
            ley_promedio_final = (ley_ponderada_sum / produccion_total) if produccion_total > 0 else 0
            valor_objetivo_total = -result.fun 

            return {
                'success': True,
                'produccion_total': produccion_total,
                'finos_total': finos_total,
                'ley_promedio': ley_promedio_final,
                'valor_objetivo_total': valor_objetivo_total,
                'asignaciones': pd.DataFrame(asignaciones),
                'x_opt': x_opt,
                'alpha_usado': alpha,
                'beta_usado': beta
            }, None
        else:
            return None, f"Optimización fallida: {result.message}"
    except Exception as e:
        return None, str(e)

# ============================================
# FUNCIÓN: BÚSQUEDA DEL ÓPTIMO
# ============================================
def buscar_alfa_beta_optimos(bateas, piques, distancias, max_distancia):
    """Itera sobre valores de alfa para encontrar el óptimo."""
    mejores_resultados = {
        'produccion_max': -1,
        'alpha_optimo': 0,
        'beta_optimo': 1,
        'resultado_final': None
    }
    
    for alpha_test in np.arange(0.0, 1.01, 0.05):
        alpha_test = round(alpha_test, 2) 
        beta_test = round(1.0 - alpha_test, 2)
        
        params = {
            'alpha': alpha_test,
            'beta': beta_test,
            'max_distancia': max_distancia
        }
        
        resultado, _ = optimizar_produccion(bateas, piques, distancias, params)
        
        if resultado and resultado['produccion_total'] > mejores_resultados['produccion_max']:
            mejores_resultados['produccion_max'] = resultado['produccion_total']
            mejores_resultados['alpha_optimo'] = alpha_test
            mejores_resultados['beta_optimo'] = beta_test
            mejores_resultados['resultado_final'] = resultado
            
    return mejores_resultados

# ============================================
# INTERFAZ STREAMLIT
# ============================================
st.sidebar.header("Cargar Datos de Mina")
archivo = st.sidebar.file_uploader("Subir Excel de Datos Operacionales", type=['xlsx', 'xls'], help="Archivo con hojas: Bateas, Piques, Equipos, Historico")

# Parámetros de la malla tipo Teniente
st.sidebar.markdown("---")
st.sidebar.header("Parámetros Malla Teniente")
dist_calles = st.sidebar.number_input("Distancia entre Calles (m)", min_value=10, max_value=50, value=30)
dist_zanjas = st.sidebar.number_input("Distancia entre Zanjas (m)", min_value=5, max_value=30, value=15)

if archivo is not None:
    try:
        bateas, piques, equipos, distancias, historico, malla = cargar_datos(archivo, dist_calles, dist_zanjas)
        n_bateas, n_piques, n_equipos = len(bateas), len(piques), len(equipos)
        horas_operacion_dia = equipos['horas_operacion_dia'].iloc[0] 
        
        st.sidebar.success(f"Datos: {n_bateas} Bateas, {n_piques} Piques, {n_equipos} Equipos")
        st.sidebar.success(f"Red de Malla: {len(malla.nodos)} nodos, {len(malla.aristas)} aristas")
        with st.sidebar.expander("Detalle Operacional"):
            st.info(f"Horas de Operación: **{horas_operacion_dia} hrs**")
            
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        st.stop()

    st.sidebar.header("Configuración Optimización")
    max_distancia = st.sidebar.slider("Distancia Máxima Batea-Pique (m)", 50, 1000, 500)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Modo de Optimización:**")

    modo_opt = st.sidebar.radio(
        "Seleccione el modo:",
        ["Modo 1: Ejecutar con Alpha y Beta definidos", "Modo 2: Buscar Optimo (Auto)"]
    )
    
    params_manual = {}
    if "Alpha" in modo_opt:
        st.sidebar.markdown("**Condición:** $\\alpha + \\beta = 1.0$")
        alpha_input = st.sidebar.slider("$\\alpha$ (Peso Ley)", 0.0, 1.0, 0.5, 0.01)
        beta_input = round(1.0 - alpha_input, 2)
        st.sidebar.markdown(f"**$\\beta$ (Peso Tonelaje): {beta_input}**")
        
        params_manual = {
            'alpha': alpha_input,
            'beta': beta_input,
            'max_distancia': max_distancia
        }
    else:
        st.sidebar.info("El algoritmo buscará el óptimo que maximiza el tonelaje.")

    # Tabs principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dashboard", "Layout Mina + Red", "Optimización", 
        "KPIs Equipos", "Predicción"
    ])

    # ============================================
    # TAB 1: DASHBOARD
    # ============================================
    with tab1:
        st.subheader("Resumen Operacional General")
        
        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Bateas Activas", n_bateas)
                st.metric("Tonelaje Disponible", f"{bateas['tonelaje_disponible'].sum():,.0f} ton")
            
            with col2:
                piques_op = piques['disponible'].sum()
                st.metric("Piques Operativos", f"{piques_op}/{n_piques}")
                cap_total = piques[piques['disponible']]['capacidad_diaria'].sum()
                st.metric("Capacidad Total Op.", f"{cap_total:,.0f} ton/día")
            
            with col3:
                st.metric("Equipos LHD", n_equipos)
                st.metric(f"Producción Potencial Máx.", f"{equipos['ton_dia_max'].sum():,.0f} ton/día")
            
            with col4:
                st.metric("Ley Promedio Bateas", f"{bateas['ley_cu'].mean():.2f} % Cu")
                st.metric("Finos Potenciales", f"{bateas['finos_disponibles_ton'].sum():,.0f} ton Cu")
        
        st.markdown("---")
        
        col_pie, col_hist = st.columns(2)
        with col_pie:
            st.markdown("**Estado de Piques**")
            fig_piques = px.pie(piques, names='estado',
                                 color='estado', 
                                 color_discrete_map={'Operativo': '#28a745', 'Mantenimiento': '#ffc107',
                                                     'Tapado': '#dc3545', 'Reparación': '#fd7e14'})
            st.plotly_chart(fig_piques, use_container_width=True)
        
        with col_hist:
            st.markdown("**Histórico de Producción**")
            fig_hist = px.line(historico, x='fecha', y='produccion_ton')
            fig_hist.add_hline(y=historico['produccion_ton'].mean(), 
                                 line_dash="dash", annotation_text="Promedio")
            st.plotly_chart(fig_hist, use_container_width=True)

    # ============================================
    # TAB 2: LAYOUT MINA + RED
    # ============================================
    with tab2:
        st.subheader("Layout Nivel de Producción + Red de Calles y Zanjas")
        
        st.caption(f"Red de Malla Tipo Teniente: {len(malla.nodos)} nodos (intersecciones) y {len(malla.aristas)} aristas (segmentos)")
        
        if 'x' in bateas.columns and 'y' in bateas.columns:
            fig = go.Figure()
            
            # Dibujar la red de calles y zanjas
            for arista in malla.aristas:
                n1, n2, dist = arista
                x1, y1 = malla.nodos[n1]['x'], malla.nodos[n1]['y']
                x2, y2 = malla.nodos[n2]['x'], malla.nodos[n2]['y']
                
                fig.add_trace(go.Scatter(
                    x=[x1, x2], y=[y1, y2],
                    mode='lines',
                    line=dict(color='#e0e0e0', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Nodos de la red
            nodos_x = [malla.nodos[n]['x'] for n in malla.nodos]
            nodos_y = [malla.nodos[n]['y'] for n in malla.nodos]
            fig.add_trace(go.Scatter(
                x=nodos_x, y=nodos_y,
                mode='markers',
                marker=dict(size=3, color='#b0b0b0', opacity=0.5),
                name='Intersecciones',
                hoverinfo='skip'
            ))
            
            # Bateas
            fig.add_trace(go.Scatter(
                x=bateas['x'], y=bateas['y'],
                mode='markers+text',
                marker=dict(size=bateas['tonelaje_disponible']/50, 
                             color=bateas['ley_cu'], colorscale='Viridis',
                             showscale=True, colorbar=dict(title='Ley Cu %')),
                text=bateas['batea_id'],
                textposition='top center',
                name='Bateas',
                hovertemplate='<b>%{text}</b><br>Ley: %{marker.color:.2f}%'
            ))
            
            # Piques
            colors_pique = ['#28a745' if d else '#dc3545' for d in piques['disponible']]
            fig.add_trace(go.Scatter(
                x=piques['x'], y=piques['y'],
                mode='markers+text',
                marker=dict(size=20, color=colors_pique, symbol='square'),
                text=piques['pique_id'],
                textposition='bottom center',
                name='Piques',
                hovertemplate='<b>%{text}</b><br>Estado: ' + piques['estado']
            ))
            
            fig.update_layout(
                title='Visualización Espacial',
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                height=700,
                showlegend=True,
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Faltan coordenadas en los datos.")
            
        with st.expander("Ver Tablas de Datos (Bateas y Piques)"):
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(bateas.drop(columns=['finos_disponibles_ton']), height=300)
            with col2:
                st.dataframe(piques, height=300)

    # ============================================
    # TAB 3: OPTIMIZACIÓN
    # ============================================
    with tab3:
        st.subheader("Optimización de Rutas y Asignación")
        
        with st.expander("Ver Formula Objetivo", expanded=False):
            st.markdown(f"""
            **Maximizar Retorno Ponderado:**
            
            $\\text{{Maximizar: }} \\alpha \\times \\sum_{{i,j}} (\\text{{Ley}}_{{i}} \\times x_{{ij}}) + \\beta \\times \\sum_{{i,j}} (\\text{{Ton}}_{{i}} \\times x_{{ij}})$
            
            * **Distancias:** Calculadas usando la red de calles y zanjas.
            """)
        
        col_btn, col_blank = st.columns([1, 4])
        with col_btn:
            ejecutar = st.button("Ejecutar Optimización", type="primary")
        
        if ejecutar:
            if "Alpha" in modo_opt:
                with st.spinner("Calculando asignación óptima..."):
                    st.session_state.opt_resultado, st.session_state.opt_error = optimizar_produccion(bateas, piques, distancias, params_manual)
            else:
                with st.spinner("Buscando parámetros óptimos..."):
                    busqueda = buscar_alfa_beta_optimos(bateas, piques, distancias, max_distancia)
                    st.session_state.opt_resultado = busqueda['resultado_final']
                    st.session_state.opt_error = None
                    if st.session_state.opt_resultado:
                        st.success(f"Óptimo: alpha={busqueda['alpha_optimo']}, beta={busqueda['beta_optimo']}")

        # Mostrar Resultados desde Session State (Persistencia)
        if st.session_state.opt_resultado:
            resultado = st.session_state.opt_resultado
            
            st.success("Asignación completada.")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Alpha", f"{resultado['alpha_usado']:.2f}")
            with col2:
                st.metric("Beta", f"{resultado['beta_usado']:.2f}")
            with col3:
                st.metric("Ton Diario", f"{resultado['produccion_total']:,.0f} ton")
            with col4:
                st.metric("Ley Promedio", f"{resultado['ley_promedio']:.2f} %")
            with col5:
                st.metric("Finos Cu", f"{resultado['finos_total']:,.0f} ton")
            
            st.markdown("---")
            
            if len(resultado['asignaciones']) > 0:
                tab_res_1, tab_res_2 = st.tabs(["Mapa de Rutas", "Tabla de Asignaciones"])
                
                with tab_res_1:
                    if 'x' in bateas.columns and 'y' in bateas.columns:
                        fig_asig = go.Figure()
                        
                        # Red de fondo
                        for arista in malla.aristas:
                            n1, n2, dist = arista
                            x1, y1 = malla.nodos[n1]['x'], malla.nodos[n1]['y']
                            x2, y2 = malla.nodos[n2]['x'], malla.nodos[n2]['y']
                            
                            fig_asig.add_trace(go.Scatter(
                                x=[x1, x2], y=[y1, y2],
                                mode='lines',
                                line=dict(color='#f0f0f0', width=0.5),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                        
                        # Piques
                        fig_asig.add_trace(go.Scatter(
                            x=piques['x'], y=piques['y'],
                            mode='markers',
                            marker=dict(size=15, color='#28a745', symbol='square'),
                            name='Piques'
                        ))
                        
                        # Bateas
                        fig_asig.add_trace(go.Scatter(
                            x=bateas['x'], y=bateas['y'],
                            mode='markers',
                            marker=dict(size=6, color='#007bff', symbol='circle'),
                            name='Bateas'
                        ))
                        
                        # Rutas
                        for _, row in resultado['asignaciones'].iterrows():
                            b_idx = bateas[bateas['batea_id'] == row['batea']].index[0]
                            p_idx = piques[piques['pique_id'] == row['pique']].index[0]
                            
                            x_origen = bateas.iloc[b_idx]['x']
                            y_origen = bateas.iloc[b_idx]['y']
                            x_destino = piques.iloc[p_idx]['x']
                            y_destino = piques.iloc[p_idx]['y']
                            
                            ruta_coords = malla.obtener_ruta(x_origen, y_origen, x_destino, y_destino)
                            
                            if ruta_coords:
                                ruta_x = [coord[0] for coord in ruta_coords]
                                ruta_y = [coord[1] for coord in ruta_coords]
                                
                                line_width = max(1, row['tonelaje'] / 200)
                                
                                fig_asig.add_trace(go.Scatter(
                                    x=ruta_x,
                                    y=ruta_y,
                                    mode='lines',
                                    line=dict(width=line_width, color='rgba(220, 53, 69, 0.6)'),
                                    showlegend=False,
                                    hovertemplate=f"Ruta: {row['batea']} -> {row['pique']}<br>Ton: {row['tonelaje']:.0f}<extra></extra>"
                                ))
                        
                        fig_asig.update_layout(
                            title='Rutas Óptimas (Grosor = Tonelaje)',
                            xaxis_title='X (m)',
                            yaxis_title='Y (m)',
                            height=600,
                            plot_bgcolor='white'
                        )
                        st.plotly_chart(fig_asig, use_container_width=True)
                
                with tab_res_2:
                    st.dataframe(resultado['asignaciones'].round({
                        'tonelaje': 0, 'ley': 2, 'distancia': 0, 'valor_objetivo': 2
                    }), use_container_width=True)
                
                st.subheader("Balance de Carga")
                resumen_pique = resultado['asignaciones'].groupby('pique').agg({
                    'tonelaje': 'sum',
                    'batea': 'count'
                }).reset_index()
                resumen_pique.columns = ['Pique', 'Tonelaje Asignado', 'N° Bateas']
                
                resumen_pique = resumen_pique.merge(
                    piques[['pique_id', 'capacidad_diaria']].rename(columns={'pique_id': 'Pique'}),
                    on='Pique', how='left'
                )
                
                fig_bar = px.bar(resumen_pique, x='Pique', y=['Tonelaje Asignado', 'capacidad_diaria'],
                                 title='Carga vs Capacidad',
                                 barmode='group',
                                 color_discrete_sequence=['#007bff', '#6c757d'])
                st.plotly_chart(fig_bar, use_container_width=True)

            else:
                st.warning("No se generaron asignaciones.")
        elif st.session_state.opt_error:
             st.error(f"Error: {st.session_state.opt_error}")

    # ============================================
    # TAB 4: KPIs EQUIPOS
    # ============================================
    with tab4:
        st.subheader("KPIs de Equipos LHD")
        
        with st.expander("Ver Tabla Completa de Equipos"):
            st.dataframe(equipos[['equipo_id', 'capacidad_balde', 'disponibilidad_mecanica', 'utilizacion_efectiva',
                                  'ciclos_por_hora', 'ton_hora_teorico', 'ton_hora_efectivo', 'ton_dia_max',
                                  'horas_operacion_dia', 'operador_experiencia']].round(2), use_container_width=True)

        col1, col2 = st.columns(2)
        
        with col1:
            fig_disp = px.bar(equipos, x='equipo_id', y='disponibilidad_mecanica',
                              title='Disponibilidad Mecánica',
                              color='disponibilidad_mecanica', color_continuous_scale='RdYlGn')
            fig_disp.add_hline(y=0.85, line_dash="dash", annotation_text="Meta 85%")
            st.plotly_chart(fig_disp, use_container_width=True)
        
        with col2:
            fig_prod = px.bar(equipos, x='equipo_id', y='ton_dia_max',
                              title='Capacidad de Producción Diaria',
                              color='operador_experiencia')
            st.plotly_chart(fig_prod, use_container_width=True)
        
        fig_eff = px.scatter(equipos, x='disponibilidad_mecanica', y='utilizacion_efectiva',
                             size='ton_dia_max', color='operador_experiencia',
                             hover_data=['equipo_id', 'capacidad_balde'],
                             title='Eficiencia: Disp. vs Utilización')
        st.plotly_chart(fig_eff, use_container_width=True)

    # ============================================
    # TAB 5: PREDICCIÓN
    # ============================================
    with tab5:
        st.subheader("Modelo Predictivo de Producción")
        
        if len(historico) > 10:
            X = historico[['piques_disponibles', 'equipos_disponibles', 'ley_promedio',
                           'turnos', 'factor_operacional']]
            y = historico['produccion_ton']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R2 Score", f"{r2_score(y_test, y_pred):.3f}")
            with col2:
                st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):,.0f} ton")
            
            st.markdown("#### Simulador")
            
            with st.container():
                col_pique, col_equipo, col_ley = st.columns(3)
                pred_piques = col_pique.slider("Piques Disp.", 1, n_piques, piques['disponible'].sum())
                pred_equipos = col_equipo.slider("Equipos Disp.", 1, n_equipos, n_equipos - 1)
                pred_ley = col_ley.slider("Ley Promedio (%)", 0.5, 2.0, historico['ley_promedio'].mean())
                
                col_turnos, col_factor, col_blank = st.columns(3)
                turnos_historico = historico['turnos'].unique().tolist()
                pred_turnos = col_turnos.selectbox("Turnos", sorted(turnos_historico) if turnos_historico else [2, 3])
                pred_factor = col_factor.slider("Factor Operacional", 0.7, 1.0, 0.9)
                
                prediccion = model.predict([[pred_piques, pred_equipos, pred_ley, pred_turnos, pred_factor]])[0]
                
                st.info(f"Producción Estimada: **{prediccion:,.0f} ton/día**")
        else:
            st.warning(f"Se necesitan más de 10 registros históricos. Actualmente: {len(historico)}.")

else:
    st.info("Sube el archivo Excel con las 4 hojas para iniciar el optimizador.")
    
    with st.expander("Ver estructura de archivo requerida"):
        st.markdown("""
        * **Hoja 'Bateas':** batea_id, x, y, tonelaje_disponible, ley_cu
        * **Hoja 'Piques':** pique_id, x, y, capacidad_diaria, estado, nivel_llenado
        * **Hoja 'Equipos':** equipo_id, capacidad_balde, disponibilidad_mecanica, utilizacion_efectiva
        * **Hoja 'Historico':** dia, fecha, piques_disponibles, equipos_disponibles
        """)
