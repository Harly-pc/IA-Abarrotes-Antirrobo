# 🛒 IA_ABARROTES (V11) - Sistema Antirrobo con Visión Artificial

Sistema de visión por computadora desarrollado en **Python** que utiliza **YOLOv8-pose** y **OpenCV** para la prevención de pérdidas y seguridad en tiendas de abarrotes y comercios.

## 🌟 Características Principales

- **Detección Biomecánica Individual:** Analiza las articulaciones de la persona en tiempo real. Alerta si la mano permanece en zonas de riesgo (bolsillos/cintura) por más de 2.5 segundos.
- **Prevención Anti-Manada:** Identifica aglomeraciones de 3 o más personas en un bloque reducido. Dispara una alerta preventiva si permanecen juntas más de 45 segundos (efecto pantalla).
- **Módulo de Aprendizaje Supervisado:** Ventana interactiva (Popup) que permite al cajero o personal clasificar los eventos como *Robo Confirmado* o *Falso Positivo*, guardando las capturas para re-entrenar la IA.
- **Alertas Sonoras y Visuales:** Tono agudo para robos inminentes y tono grave preventivo para grupos.

## 🛠️ Requisitos e Instalación

1. Clona o descarga este repositorio.
2. Instala las librerías necesarias ejecutando el siguiente comando en la terminal:

```bash
pip install ultralytics opencv-python numpy
