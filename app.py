import streamlit as st
import numpy as np
import cv2
from PIL import Image
import solver

st.set_page_config(layout="wide", page_title="Asistente de Rompecabezas")

st.title("🧩 Asistente de Armado de Rompecabezas")
st.markdown("""
Esta aplicación te ayuda a encontrar la ubicación de las fichas sueltas en tu rompecabezas.
1. Sube la **Imagen Original** (del rompecabezas completo).
2. Sube una **Foto de las Fichas** que quieres ubicar (sobre un fondo liso preferiblemente).
3. La IA intentará localizar dónde van esas fichas.
""")

col1, col2 = st.columns(2)

with col1:
    st.header("1. Imagen de Referencia")
    referencia_file = st.file_uploader("Sube la imagen de la caja o puzzle completo", type=['jpg', 'png', 'jpeg'], key="ref")

with col2:
    st.header("2. Fichas Sueltas")
    piezas_file = st.file_uploader("Sube foto de las fichas (fondo contraste)", type=['jpg', 'png', 'jpeg'], key="pieces")

if referencia_file and piezas_file:
    # Mostrar imágenes cargadas
    img_ref = solver.cargar_imagen(referencia_file)
    img_piezas = solver.cargar_imagen(piezas_file)

    st.subheader("Vistas Previas")
    c1, c2 = st.columns(2)
    with c1:
        st.image(img_ref, channels="BGR", caption="Puzzle Completo", use_container_width=True)
    with c2:
        st.image(img_piezas, channels="BGR", caption="Mis Fichas", use_container_width=True)

    if st.button("🔍 Analizar y Buscar Piezas", type="primary"):
        with st.spinner("Analizando texturas y formas de las piezas..."):
            try:
                # Procesar
                resultado, num_encontradas = solver.detectar_y_emparejar(img_ref, img_piezas)
                
                st.success(f"¡Análisis completado! Se han localizado {num_encontradas} posibles ubicaciones.")
                
                st.subheader("Resultado")
                st.image(resultado, channels="BGR", caption="Ubicaciones Sugeridas (Marcadas en Verde)", use_container_width=True)
                
                if num_encontradas == 0:
                    st.warning("No se encontraron coincidencias claras. Intenta tomar la foto de las fichas más cerca, con mejor luz, o asegúrate que no estén rotadas excesivamente (aunque el algoritmo tolera rotación).")

            except Exception as e:
                st.error(f"Ocurrió un error durante el análisis: {str(e)}")
else:
    st.info("👆 Por favor sube ambas imágenes para comenzar.")

st.markdown("---")
st.markdown("Desarrollado con ❤️ usando OpenCV y Python.")
