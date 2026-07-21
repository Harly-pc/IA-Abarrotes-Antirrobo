"""
=============================================================================
PROTOTIPO FINAL: SISTEMA DE PREVENCIÓN DE ROBOS MEDIANTE IA (V11)
=============================================================================
Descripción: 
Sistema de visión artificial que utiliza YOLOv8-pose para detectar dos 
comportamientos de riesgo en tiendas de abarrotes:
1. Robo individual (ocultamiento de productos en bolsillos/ropa).
2. Táctica de "manada" o "hacer pantalla" (grupos bloqueando la visión).

Características:
- Análisis biomecánico de articulaciones (manos, hombros, caderas).
- Detección de proximidad grupal con temporizador de 45 segundos.
- Interfaz gráfica interactiva para recolección de datos (Robo vs Falso Positivo).
- Alertas sonoras diferenciadas (Prevención vs Alerta Crítica).
=============================================================================
"""

import cv2
import time
import math
import os
import numpy as np
from datetime import datetime
from collections import deque
import winsound 
from ultralytics import YOLO

# ==========================================
# 1. PREPARACIÓN DEL SISTEMA DE APRENDIZAJE
# ==========================================
# Creamos las carpetas donde se guardarán las imágenes para entrenar 
# futuros modelos de IA (Aprendizaje Supervisado).
CARPETA_DATOS = "Base_Datos_Aprendizaje"
CARPETA_CONFIRMADOS = os.path.join(CARPETA_DATOS, "Robo_Confirmado") 
CARPETA_FALSOS = os.path.join(CARPETA_DATOS, "Falso_Positivo")       
CARPETA_NO_CONFIRMADOS = os.path.join(CARPETA_DATOS, "No_Confirmado")

# Verificación y creación de directorios si no existen
for carpeta in [CARPETA_CONFIRMADOS, CARPETA_FALSOS, CARPETA_NO_CONFIRMADOS]:
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)

# Carga del modelo de estimación de pose (versión nano para mayor velocidad)
modelo_pose = YOLO("yolov8n-pose.pt") 

# ==========================================
# 2. CONFIGURACIÓN DE CÁMARA Y PARÁMETROS
# ==========================================
ES_VIDEO_LOCAL = True 
# URL RTSP de la cámara IP (cambiar ES_VIDEO_LOCAL a False para usar en vivo)
URL_CAMARA = "tienda.mp4" if ES_VIDEO_LOCAL else "rtsp://user:login@192.168.x.x:554/H.264"
camara = cv2.VideoCapture(URL_CAMARA)

# --- Diccionarios de Memoria ---
memoria_clientes = {}            # Guarda el historial y temporizadores de cada persona detectada
memoria_video = deque(maxlen=30) # Guarda los últimos cuadros limpios para extraer recortes
alertas_gestionadas_sesion = {}  # Evita mostrar alertas repetidas para la misma persona

# --- Parámetros de Detección Individual (Robo) ---
UMBRAL_TIEMPO_ALERTA = 2.5       # Segundos que la mano debe estar en zona sospechosa
TIEMPO_GRACIA_INICIAL = 2.0      # Segundos antes de empezar a analizar a alguien nuevo
UMBRAL_ENFRIAMIENTO_ALERTA = 6.0 # Segundos entre alertas sonoras globales
UMBRAL_ANGULO_SENTADO = 120      # Ángulo de la rodilla para determinar si está sentado
TIEMPO_MAXIMO_VENTANA = 45       # Segundos que el popup espera una respuesta del cajero

# --- Parámetros de Detección Grupal (Anti-Manada) ---
UMBRAL_DISTANCIA_GRUPO = 110     # Píxeles máximos entre personas (muy pegados = bloque)
MIN_PERSONAS_GRUPO = 3           # Cantidad mínima para considerar un grupo sospechoso
TIEMPO_ALERTA_GRUPO = 45.0       # Segundos continuos que deben estar agrupados para alertar
TIEMPO_ENFRIAMIENTO_GRUPO = 10.0 # Segundos de espera para no saturar con el pitido de grupo

# --- Paleta de Colores (BGR) ---
COLOR_ROJO = (0, 0, 255)
COLOR_BLANCO = (255, 255, 255)
COLOR_GRIS = (100, 100, 100)
COLOR_VERDE = (0, 200, 0)
COLOR_NEGRO = (0, 0, 0)
COLOR_AMARILLO = (0, 255, 255) 
COLOR_NARANJA = (0, 165, 255) 

NOMBRE_VENTANA_POPUP = "ALERTA_PREVENCION"
tracker_config = "bytetrack.yaml" # Algoritmo de seguimiento (tracking)

# --- Funciones Sonoras ---
def reproducir_sonido_aviso_profesional():
    """Sonido agudo de dos tonos para alerta de ROBO individual."""
    winsound.Beep(2000, 100) 
    winsound.Beep(1600, 100) 

def reproducir_sonido_grupo():
    """Sonido grave y único para alerta PREVENTIVA de grupo."""
    winsound.Beep(1200, 150)

# ==========================================
# 3. MOTOR DE INTERFAZ GRÁFICA (POPUP)
# ==========================================
# Diccionario que controla el estado de la ventana emergente de validación
estado_alerta = {
    "activa": False, "img_interfaz": None, "img_limpia": None,
    "id_cliente": None, "menu_desplegado": False,
    "tiempo_apertura": 0, "cerrar_ahora": False
}

def manejador_clics(event, x, y, flags, param):
    """Detecta los clics del mouse en la ventana emergente para clasificar los eventos."""
    global estado_alerta, alertas_gestionadas_sesion
    if event == cv2.EVENT_LBUTTONDOWN and estado_alerta["activa"]:
        h, w = estado_alerta["img_interfaz"].shape[:2]
        # Validar si el clic fue en la zona inferior (los botones)
        if y >= h - 50:
            current_id = estado_alerta["id_cliente"]
            # Si el menú no está desplegado, hacer clic en los 3 puntos (...)
            if not estado_alerta["menu_desplegado"]:
                if x >= w - 60: estado_alerta["menu_desplegado"] = True
            else:
                # Si está desplegado, evaluar si eligió "SI: ROBO" (izq) o "NO: FALSO" (der)
                fecha_hora = datetime.now().strftime("%Y%m%d_%H%M%S")
                if x < w // 2:
                    ruta = os.path.join(CARPETA_CONFIRMADOS, f"robo_ID{current_id}_{fecha_hora}.jpg")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🧠 APRENDIZAJE: ROBO REAL guardado (ID {current_id}).")
                else:
                    ruta = os.path.join(CARPETA_FALSOS, f"falso_ID{current_id}_{fecha_hora}.jpg")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🧠 APRENDIZAJE: FALSA ALARMA guardada (ID {current_id}).")
                
                # Guardar imagen y cerrar ventana
                cv2.imwrite(ruta, estado_alerta["img_limpia"])
                alertas_gestionadas_sesion[current_id] = {"gestionado": True} 
                estado_alerta["cerrar_ahora"] = True 

def dibujar_interfaz_alerta(imagen_base):
    """Dibuja los botones interactivos sobre la imagen recortada del sospechoso."""
    img_render = imagen_base.copy()
    h, w = img_render.shape[:2]
    
    # Línea separadora
    cv2.line(img_render, (0, h - 50), (w, h - 50), COLOR_BLANCO, 1) 
    
    if not estado_alerta["menu_desplegado"]:
        # Dibujar botón de tres puntos (...)
        cv2.rectangle(img_render, (w - 60, h - 45), (w - 10, h - 5), COLOR_GRIS, -1)
        cv2.putText(img_render, "...", (w - 50, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_BLANCO, 2)
    else:
        # Dibujar botones SI / NO dividiendo la zona en dos
        mitad = w // 2
        cv2.rectangle(img_render, (0, h - 50), (mitad, h), COLOR_ROJO, -1)
        cv2.putText(img_render, "SI: ROBO", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLANCO, 2)
        cv2.rectangle(img_render, (mitad, h - 50), (w, h), COLOR_GRIS, -1)
        cv2.putText(img_render, "NO: FALSO", (mitad + 10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLANCO, 2)
    
    return img_render

# ==========================================
# 4. FUNCIONES MATEMÁTICAS Y DE COMPORTAMIENTO
# ==========================================

def evaluar_aglomeracion(coords_deteccion, track_ids, umbral, min_personas):
    """
    Agrupa a las personas detectadas en función de su distancia física.
    Si están muy cerca (por debajo del umbral), se consideran un grupo.
    """
    centros = []
    # Calcular centro (x, y) de cada bounding box
    for box, id_p in zip(coords_deteccion, track_ids):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        centros.append((id_p, cx, cy, box))

    grupos_sospechosos = []
    visitados = set()

    for i in range(len(centros)):
        if i in visitados: continue
        grupo_actual = [centros[i]]
        visitados.add(i)

        for j in range(i + 1, len(centros)):
            if j in visitados: continue
            # Calcular distancia euclidiana entre centros
            dist = math.dist((centros[i][1], centros[i][2]), (centros[j][1], centros[j][2]))
            if dist < umbral:
                grupo_actual.append(centros[j])
                visitados.add(j)

        if len(grupo_actual) >= min_personas:
            grupos_sospechosos.append(grupo_actual)

    return grupos_sospechosos

def calcular_angulo(a, b, c):
    """Calcula el ángulo entre 3 puntos (ej: cadera, rodilla, tobillo)."""
    try:
        if None in [a, b, c]: return 180
        return abs(math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])))
    except: return 180

def calcular_distancia_escalada(p1, p2, hombro_anchura):
    """Calcula la distancia normalizada según el tamaño de la persona en pantalla."""
    if hombro_anchura <= 0: return 999
    return math.dist(p1, p2) / hombro_anchura

def analizar_varianza_movimiento(historial_puntos):
    """Evalúa si la mano está estática (ocultando algo) o en movimiento normal."""
    if len(historial_puntos) < 5: return True 
    return np.var([p[1] for p in historial_puntos]) > 15.0 

def evaluar_cliente_pro(keypoints, confianzas, id_cliente):
    """
    Núcleo biomecánico: Analiza las posiciones de las articulaciones para 
    determinar si una mano está de forma sospechosa cerca de los bolsillos/cintura.
    """
    if confianzas is None or len(keypoints) < 13: return False, None, False
    try:
        hombro_izq, hombro_der = keypoints[5], keypoints[6]
        mano_izq, mano_der = keypoints[9], keypoints[10]
        cadera_izq, cadera_der = keypoints[11], keypoints[12]
        rodilla_izq, tobillo_izq = keypoints[13], keypoints[15]
        
        # Ignorar si hay baja confianza en la detección de las piezas clave
        if confianzas[9] < 0.4 or confianzas[11] < 0.4 or abs(hombro_der[0] - hombro_izq[0]) < 20:
            return False, None, False

        anchura_hombros = abs(hombro_der[0] - hombro_izq[0])
        es_sentado = False
        
        # Verificar postura (sentado/de pie) para ajustar la zona de riesgo
        if confianzas[13] > 0.5 and confianzas[15] > 0.5:
            if calcular_angulo(cadera_izq, rodilla_izq, tobillo_izq) < UMBRAL_ANGULO_SENTADO:
                es_sentado = True

        umbral_dist = 0.25 if es_sentado else 0.35 
        tiempo_actual = time.time()
        
        # Esperar tiempo de gracia antes de analizar a alguien recién detectado
        if (tiempo_actual - memoria_clientes[id_cliente]["primer_visto"]) < TIEMPO_GRACIA_INICIAL:
            return False, None, es_sentado

        # Guardar historial de trayectoria de las manos
        memoria_clientes[id_cliente]["trayectoria_mano_izq"].append(mano_izq)
        memoria_clientes[id_cliente]["trayectoria_mano_der"].append(mano_der)
        
        # Lógica de detección: Mano Izquierda
        if calcular_distancia_escalada(mano_izq, cadera_izq, anchura_hombros) < umbral_dist and mano_izq[1] > hombro_izq[1]:
            if not analizar_varianza_movimiento(memoria_clientes[id_cliente]["trayectoria_mano_izq"]):
                return True, mano_izq, es_sentado 

        # Lógica de detección: Mano Derecha
        if calcular_distancia_escalada(mano_der, cadera_der, anchura_hombros) < umbral_dist and mano_der[1] > hombro_der[1]:
            if not analizar_varianza_movimiento(memoria_clientes[id_cliente]["trayectoria_mano_der"]):
                return True, mano_der, es_sentado 

    except Exception: pass
    return False, None, False

def generar_recortes_visuales(frame_pasado, bounding_box, customer_id, coord_mano):
    """Corta la imagen de la persona sospechosa para mostrarla en el Popup de alerta."""
    x1, y1, x2, y2 = map(int, bounding_box)
    h, w, _ = frame_pasado.shape
    
    # Agregar padding (margen) para no cortar la imagen muy ajustada
    pad_y, pad_x = int(abs(y2-y1) * 0.20), int(abs(x2-x1) * 0.25) 
    c_y1, c_y2 = max(0, y1 - pad_y), min(h, y2 + pad_y)
    c_x1, c_x2 = max(0, x1 - pad_x), min(w, x2 + pad_x)
    
    recorte_limpio = frame_pasado[c_y1:c_y2, c_x1:c_x2].copy() 
    
    # Dibujar indicadores en el recorte
    cv2.rectangle(frame_pasado, (x1, y1), (x2, y2), COLOR_ROJO, 5)
    cv2.putText(frame_pasado, f"SOSPECHA ID {customer_id}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ROJO, 2)

    if coord_mano:
        rad = 25 
        mx1, my1 = max(0, int(coord_mano[0] - rad)), max(0, int(coord_mano[1] - rad))
        mx2, my2 = min(w, int(coord_mano[0] + rad)), min(h, int(coord_mano[1] + rad))
        
        cv2.rectangle(frame_pasado, (mx1, my1), (mx2, my2), COLOR_AMARILLO, 3)
        cv2.putText(frame_pasado, "MANO", (mx1, max(20, my1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_AMARILLO, 2)

    recorte_base = frame_pasado[c_y1:c_y2, c_x1:c_x2]
    
    # Añadir barra inferior negra para colocar los botones de la interfaz
    recorte_con_barra = cv2.copyMakeBorder(recorte_base, 0, 50, 0, 0, cv2.BORDER_CONSTANT, value=COLOR_NEGRO)
    
    # Redimensionar estandarizado si la imagen es muy pequeña
    if recorte_con_barra.shape[1] < 300:
        nuevo_h = int(recorte_con_barra.shape[0] * (300 / recorte_con_barra.shape[1]))
        recorte_con_barra = cv2.resize(recorte_con_barra, (300, nuevo_h))

    return recorte_con_barra, recorte_limpio

# ==========================================
# 5. BUCLE PRINCIPAL DEL SISTEMA (EJECUCIÓN)
# ==========================================

contador_cuadros = 0
tiempo_ultima_alerta_global = 0
tiempo_ultimo_aviso_grupo = 0 

print("\n" + "="*60)
print("✅ INICIANDO SISTEMA IA_ABARROTES ")
print("✅ Módulo de Aprendizaje Supervisado: ACTIVADO")
print("="*60 + "\n")

while True:
    exito, cuadro = camara.read() 
    
    if not exito:
        if ES_VIDEO_LOCAL:
            camara.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reiniciar video si es archivo local
            continue
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 ERROR: Conexión RTSP perdida. Reintentando...")
            camara.release()
            time.sleep(5)
            camara = cv2.VideoCapture(URL_CAMARA)
            continue

    contador_cuadros += 1
    # Optimización: Procesar 1 de cada 3 cuadros para liberar CPU
    if contador_cuadros % 3 != 0: continue

    cuadro = cv2.resize(cuadro, (640, 480))
    cuadro_limpio = cuadro.copy()
    memoria_video.append(cuadro_limpio) 

    # Inferencia del modelo (Pose + Tracking)
    resultados = modelo_pose.track(cuadro, stream=True, verbose=False, tracker=tracker_config, persist=True)
    
    for r in resultados:
        boxes = r.boxes
        keypoints_array = r.keypoints
        cuadro_dibujado = r.plot()
        
        if keypoints_array is not None and keypoints_array.has_visible and boxes.id is not None:
            track_ids = boxes.id.int().tolist()
            coords_deteccion = boxes.xyxy.int().tolist()
            tiempo_actual = time.time()

            # Asegurar que todas las personas detectadas tengan su espacio en memoria
            for id_persona in track_ids:
                if id_persona not in memoria_clientes:
                    memoria_clientes[id_persona] = {
                        "primer_visto": tiempo_actual, "inicio_sospecha": 0.0,
                        "alerta_emitida": False, "trayectoria_mano_izq": deque(maxlen=10),
                        "trayectoria_mano_der": deque(maxlen=10),
                        "inicio_grupo": 0.0 # Temporizador individual para grupos
                    }

            # ----------------------------------------------------
            # EVALUACIÓN DE GRUPOS (MÓDULO ANTI-MANADA)
            # ----------------------------------------------------
            ids_en_grupo_actual = set()

            if len(coords_deteccion) >= MIN_PERSONAS_GRUPO:
                grupos_detectados = evaluar_aglomeracion(coords_deteccion, track_ids, UMBRAL_DISTANCIA_GRUPO, MIN_PERSONAS_GRUPO)
                
                for grupo in grupos_detectados:
                    tiempo_maximo_grupo = 0
                    
                    # Actualizar cronómetro para cada integrante del grupo detectado
                    for persona in grupo:
                        id_p = persona[0]
                        ids_en_grupo_actual.add(id_p)
                        
                        if memoria_clientes[id_p]["inicio_grupo"] == 0.0:
                            memoria_clientes[id_p]["inicio_grupo"] = tiempo_actual
                            
                        tiempo_juntos = tiempo_actual - memoria_clientes[id_p]["inicio_grupo"]
                        if tiempo_juntos > tiempo_maximo_grupo:
                            tiempo_maximo_grupo = tiempo_juntos
                    
                    # Si superan los 45 segundos aglomerados, disparar alerta preventiva
                    if tiempo_maximo_grupo >= TIEMPO_ALERTA_GRUPO:
                        # Calcular el recuadro general que envuelve a toda la manada
                        min_x = int(min([p[3][0] for p in grupo]))
                        min_y = int(min([p[3][1] for p in grupo]))
                        max_x = int(max([p[3][2] for p in grupo]))
                        max_y = int(max([p[3][3] for p in grupo]))

                        # Dibujar interfaz visual naranja
                        cv2.rectangle(cuadro_dibujado, (min_x, min_y), (max_x, max_y), COLOR_NARANJA, 4)
                        cv2.putText(cuadro_dibujado, f"PREVENCION: GRUPO ({int(tiempo_maximo_grupo)}s)", 
                                    (min_x, max(20, min_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_NARANJA, 2)
                        
                        # Sonar pitido de prevención si ya pasó el tiempo de enfriamiento
                        if (tiempo_actual - tiempo_ultimo_aviso_grupo) > TIEMPO_ENFRIAMIENTO_GRUPO:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ ALERTA: Grupo sospechoso detectado (>{int(TIEMPO_ALERTA_GRUPO)}s)")
                            reproducir_sonido_grupo()
                            tiempo_ultimo_aviso_grupo = tiempo_actual

            # Reiniciar cronómetro si la persona se separó del grupo
            for id_persona in list(memoria_clientes.keys()):
                if id_persona not in ids_en_grupo_actual:
                    memoria_clientes[id_persona]["inicio_grupo"] = 0.0

            # ----------------------------------------------------
            # ANÁLISIS INDIVIDUAL (MÓDULO DE ROBO)
            # ----------------------------------------------------
            for i, id_persona in enumerate(track_ids):
                persona_kp = keypoints_array.xy[i].tolist() if len(keypoints_array.xy) > i else None
                persona_conf = keypoints_array.conf[i].tolist() if len(keypoints_array.conf) > i else None

                es_sospechoso, coord_mano, es_sentado = evaluar_cliente_pro(persona_kp, persona_conf, id_persona)

                if es_sospechoso:
                    if memoria_clientes[id_persona]["inicio_sospecha"] == 0.0:
                        memoria_clientes[id_persona]["inicio_sospecha"] = tiempo_actual
                    
                    tiempo_acumulado = tiempo_actual - memoria_clientes[id_persona]["inicio_sospecha"]
                    id_fue_gestionado = alertas_gestionadas_sesion.get(id_persona, {}).get("gestionado", False)
                    
                    # Disparar Alerta Crítica (Popup) si se supera el tiempo límite
                    if tiempo_acumulado > UMBRAL_TIEMPO_ALERTA and not estado_alerta["activa"] and not id_fue_gestionado:
                        if (tiempo_actual - tiempo_ultima_alerta_global > UMBRAL_ENFRIAMIENTO_ALERTA) and not memoria_clientes[id_persona]["alerta_emitida"]:
                            
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 CRÍTICO: Posible robo detectado. Abriendo Popup (ID {id_persona}).")
                            memoria_clientes[id_persona]["alerta_emitida"] = True
                            tiempo_ultima_alerta_global = tiempo_actual
                            
                            # Obtener imagen pasada de la memoria de video
                            recorte_popup, recorte_limpio = generar_recortes_visuales(
                                memoria_video[0].copy(), coords_deteccion[i], id_persona, coord_mano
                            )
                            
                            # Configurar estado del Popup
                            estado_alerta = {
                                "activa": True, "img_interfaz": recorte_popup, "img_limpia": recorte_limpio,
                                "id_cliente": id_persona, "menu_desplegado": False,
                                "tiempo_apertura": time.time(), "cerrar_ahora": False
                            }
                            
                            reproducir_sonido_aviso_profesional()
                            
                            # Forzar la ventana sobre todas las demás aplicaciones
                            cv2.namedWindow(NOMBRE_VENTANA_POPUP, cv2.WINDOW_AUTOSIZE)
                            cv2.setWindowProperty(NOMBRE_VENTANA_POPUP, cv2.WND_PROP_TOPMOST, 1)
                            cv2.setMouseCallback(NOMBRE_VENTANA_POPUP, manejador_clics)

                    if coord_mano:
                        # Etiqueta visual flotante indicando segundos de sospecha
                        cv2.putText(cuadro_dibujado, f"Sospecha: {tiempo_acumulado:.1f}s", (int(coord_mano[0]), int(coord_mano[1]) - 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ROJO, 2)
                else:
                    # Si dejó de hacer el movimiento sospechoso, reiniciar cronómetro
                    if id_persona in memoria_clientes:
                        memoria_clientes[id_persona]["inicio_sospecha"] = 0.0
                        memoria_clientes[id_persona]["alerta_emitida"] = False

    cv2.imshow("IA - Monitor Central", cuadro_dibujado)
    
    # ==========================================
    # 6. GESTIÓN DE LA VENTANA DE ALERTA (POPUP)
    # ==========================================
    if estado_alerta["activa"]:
        tiempo_abierta = time.time() - estado_alerta["tiempo_apertura"]
        
        # Cierre automático si el cajero no respondió a tiempo
        if tiempo_abierta > TIEMPO_MAXIMO_VENTANA:
            current_id = estado_alerta["id_cliente"]
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ TIMEOUT: Popup cerrado automáticamente (ID {current_id}).")
            ruta = os.path.join(CARPETA_NO_CONFIRMADOS, f"noconfirmado_ID{current_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(ruta, estado_alerta["img_limpia"])
            alertas_gestionadas_sesion[current_id] = {"gestionado": True}
            estado_alerta["cerrar_ahora"] = True
            
        if not estado_alerta["cerrar_ahora"]:
            img_final = dibujar_interfaz_alerta(estado_alerta["img_interfaz"])
            # Dibujar cuenta regresiva en el popup
            cv2.putText(img_final, f"{int(TIEMPO_MAXIMO_VENTANA - tiempo_abierta)}s", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_VERDE, 2)
            cv2.imshow(NOMBRE_VENTANA_POPUP, img_final)
        else:
            cv2.destroyWindow(NOMBRE_VENTANA_POPUP)
            estado_alerta["activa"] = False

    # Detener el programa al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🛑 Sistema detenido por el usuario.")
        break

camara.release()
cv2.destroyAllWindows()
