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
    tab_ref_file, tab_ref_cam = st.tabs(["📂 Subir Archivo", "📸 Usar Cámara"])
    
    with tab_ref_file:
        ref_file_val = st.file_uploader("Sube la imagen de la caja", type=['jpg', 'png', 'jpeg'], key="ref_upload")
    with tab_ref_cam:
        ref_cam_val = st.camera_input("Toma una foto de la caja", key="ref_cam")
    
    referencia_file = ref_file_val if ref_file_val else ref_cam_val

with col2:
    st.header("2. Fichas Sueltas")
    tab_piezas_file, tab_piezas_cam = st.tabs(["📂 Subir Archivo", "📸 Usar Cámara"])
    
    with tab_piezas_file:
        piezas_file_val = st.file_uploader("Sube foto de las fichas", type=['jpg', 'png', 'jpeg'], key="pieces_upload")
    with tab_piezas_cam:
        piezas_cam_val = st.camera_input("Toma una foto de las fichas", key="pieces_cam")
    
    piezas_file = piezas_file_val if piezas_file_val else piezas_cam_val

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
                resultado, lista_fichas = solver.detectar_y_emparejar(img_ref, img_piezas)
                
                num_encontradas = len(lista_fichas)
                st.success(f"¡Análisis completado! Se han localizado {num_encontradas} posibles ubicaciones.")
                
                st.subheader("Resultado")
                st.image(resultado, channels="BGR", caption="Ubicaciones Sugeridas (Marcadas en Verde)", use_container_width=True)
                
                if num_encontradas > 0:
                    st.write("### 📋 Detalle de Fichas Encontradas")
                    st.dataframe(lista_fichas, use_container_width=True)
                
                if num_encontradas == 0:
                    st.warning("No se encontraron coincidencias claras. Intenta tomar la foto de las fichas más cerca, con mejor luz, o asegúrate que no estén rotadas excesivamente (aunque el algoritmo tolera rotación).")

            except Exception as e:
                st.error(f"Ocurrió un error durante el análisis: {str(e)}")
else:
    st.info("👆 Por favor sube ambas imágenes para comenzar.")

st.markdown("---")
st.markdown("Desarrollado con ❤️ usando OpenCV y Python.")
