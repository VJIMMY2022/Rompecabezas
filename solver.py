import cv2
import numpy as np
from PIL import Image

def cargar_imagen(imagen_file):
    """Convierte el archivo subido a un formato compatible con OpenCV."""
    img = Image.open(imagen_file)
    img = np.array(img)
    # Convertir RGB a BGR (OpenCV usa BGR)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def detectar_y_emparejar(img_ref, img_piezas):
    """
    Detecta piezas en la imagen de piezas y busca su ubicación en la imagen de referencia.
    Retorna la imagen de referencia con las ubicaciones marcadas.
    """
    # Inicializar detector SIFT
    # Nota: SIFT es bueno para invarianza de escala y rotación.
    sift = cv2.SIFT_create()

    # Puntos clave y descriptores de la referencia
    kp_ref, des_ref = sift.detectAndCompute(img_ref, None)
    
    # Procesar imagen de piezas para separar las fichas individuales
    # Convertir a escala de grises y aplicar umbral
    gray_piezas = cv2.cvtColor(img_piezas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_piezas, 240, 255, cv2.THRESH_BINARY_INV) # Asumimos fondo claro/blanco

    # Encontrar contornos de las piezas
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Imagen copia para dibujar resultados
    img_resultado = img_ref.copy()
    
    # Configurar matcher
    bf = cv2.BFMatcher()

    piezas_encontradas = []

    for i, cnt in enumerate(contornos):
        # Ignorar contornos muy pequeños (ruido)
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        # Extraer ROI (Región de Interés) de la pieza
        x, y, w, h = cv2.boundingRect(cnt)
        pieza_roi = img_piezas[y:y+h, x:x+w]
        
        # Detectar puntos clave en la pieza individual
        kp_pieza, des_pieza = sift.detectAndCompute(pieza_roi, None)

        if des_pieza is None or len(kp_pieza) < 5:
            continue

        # Emparejar con referencia
        matches = bf.knnMatch(des_pieza, des_ref, k=2)

        # Aplicar test de ratio de Lowe
        buenos_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                buenos_matches.append(m)

        # Si hay suficientes coincidencias, localizar la pieza
        MIN_MATCH_COUNT = 10
        if len(buenos_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp_pieza[m.queryIdx].pt for m in buenos_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in buenos_matches]).reshape(-1, 1, 2)

            # Encontrar homografía
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                h_roi, w_roi = pieza_roi.shape[:2]
                pts = np.float32([[0, 0], [0, h_roi - 1], [w_roi - 1, h_roi - 1], [w_roi - 1, 0]]).reshape(-1, 1, 2)
                try:
                    dst = cv2.perspectiveTransform(pts, M)
                    # Dibujar caja en la imagen de referencia
                    img_resultado = cv2.polylines(img_resultado, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                    
                # Dibujar etiqueta
                    moments = cv2.moments(dst)
                    if moments["m00"] != 0:
                        cX = int(moments["m10"] / moments["m00"])
                        cY = int(moments["m01"] / moments["m00"])
                        cv2.putText(img_resultado, f"Pieza {i+1}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        
                        # Determinar confianza basada en numero de matches
                        confianza = "Alta" if len(buenos_matches) > 30 else "Media" if len(buenos_matches) > 15 else "Baja"
                        
                        detalles_pieza = {
                            "Pieza ID": f"#{i+1}",
                            "Coincidencias": len(buenos_matches),
                            "Confianza": confianza,
                            "Ubicacion Approx": f"({cX}, {cY})"
                        }
                        piezas_encontradas.append(detalles_pieza)
                except Exception as e:
                    print(f"Error transformando pieza {i}: {e}")

    return img_resultado, piezas_encontradas
