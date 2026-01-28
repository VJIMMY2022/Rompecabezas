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

def intentar_match(kp_pieza, des_pieza, kp_ref, des_ref, bf, ratio_thresh=0.7, min_matches=8, img_ref_shape=None):
    """Intenta emparejar descriptores y validar geometricamente."""
    if des_pieza is None or len(kp_pieza) < 5:
        return False, None, 0

    # Emparejar
    try:
        matches = bf.knnMatch(des_pieza, des_ref, k=2)
    except:
        return False, None, 0

    # Ratio test
    buenos_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            buenos_matches.append(m)

    if len(buenos_matches) > min_matches:
        src_pts = np.float32([kp_pieza[m.queryIdx].pt for m in buenos_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in buenos_matches]).reshape(-1, 1, 2)

        try:
            # Homografia
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                # Proyectar puntos
                # Asumimos que kp_pieza viene de una imagen normalizada o del ROI original
                # Necesitamos saber el tamaño del ROI original para definir la caja
                # Pero aqui solo tenemos keypoints. 
                # Simplificacion: Usamos los limites de los keypoints de la pieza para definir su "caja"
                h_pts = [p[0][1] for p in src_pts]
                w_pts = [p[0][0] for p in src_pts]
                if not h_pts or not w_pts: return False, None, 0
                
                h_min, h_max = min(h_pts), max(h_pts)
                w_min, w_max = min(w_pts), max(w_pts)
                
                pts = np.float32([[w_min, h_min], [w_min, h_max], [w_max, h_max], [w_max, h_min]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                # VALIDACION GEOMETRICA
                if not cv2.isContourConvex(np.int32(dst)):
                    return False, None, 0
                
                area_dst = cv2.contourArea(np.int32(dst))
                if img_ref_shape:
                     if area_dst < 100 or area_dst > (img_ref_shape[0]*img_ref_shape[1] * 0.7):
                         return False, None, 0

                return True, dst, len(buenos_matches)
        except Exception:
            return False, None, 0

    return False, None, 0

def detectar_y_emparejar(img_ref, img_piezas):
    """
    Version 2.0: Robustez Extrema.
    Prueba multiples transformaciones (CLAHE, Rotacion) para encontrar la pieza.
    """
    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(img_ref, None)
    bf = cv2.BFMatcher()
    
    # 1. Segmentacion Adaptativa
    gray_piezas = cv2.cvtColor(img_piezas, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray_piezas, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=3) # Dilatar mas para conectar piezas rotas
    
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_resultado = img_ref.copy()
    img_piezas_marcadas = img_piezas.copy()
    piezas_encontradas = []

    for i, cnt in enumerate(contornos):
        area = cv2.contourArea(cnt)
        if area < 500 or area > (img_piezas.shape[0] * img_piezas.shape[1] * 0.95):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        pieza_roi_bgr = img_piezas[y:y+h, x:x+w]
        
        # ESTRATEGIA MULTIPLES INTENTOS
        # Intentaremos encontrar match con varias variaciones de la pieza
        match_found = False
        best_match = {"matches": 0, "dst": None}

        variations = []
        
        # 1. Normal
        variations.append(("Normal", pieza_roi_bgr))
        
        # 2. CLAHE (Contraste mejorado)
        lab = cv2.cvtColor(pieza_roi_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        variations.append(("CLAHE", final_clahe))

        for label, img_var in variations:
            # Probar rotaciones para cada variacion de imagen
            # 0, 90, 180, 270
            for angle in [0, 90, 180, 270]:
                if angle == 0:
                    img_rot = img_var
                else:
                    if angle == 90:
                        img_rot = cv2.rotate(img_var, cv2.ROTATE_90_CLOCKWISE)
                    elif angle == 180:
                        img_rot = cv2.rotate(img_var, cv2.ROTATE_180)
                    elif angle == 270:
                        img_rot = cv2.rotate(img_var, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                kp_p, des_p = sift.detectAndCompute(img_rot, None)
                
                # Intentar match estricto primero
                success, dst, n_matches = intentar_match(kp_p, des_p, kp_ref, des_ref, bf, 
                                                       ratio_thresh=0.75, min_matches=8, 
                                                       img_ref_shape=img_ref.shape)
                
                # Si falla, intentar match relajado
                if not success:
                    success, dst, n_matches = intentar_match(kp_p, des_p, kp_ref, des_ref, bf, 
                                                           ratio_thresh=0.8, min_matches=7, 
                                                           img_ref_shape=img_ref.shape)

                if success:
                    if n_matches > best_match["matches"]:
                        best_match = {"matches": n_matches, "dst": dst}
                        match_found = True
            
            if match_found and best_match["matches"] > 20:
                break # Si encontramos algo muy bueno, dejamos de buscar

        if match_found:
            dst = best_match["dst"]
            img_resultado = cv2.polylines(img_resultado, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
            
            moments = cv2.moments(dst)
            if moments["m00"] != 0:
                cX = int(moments["m10"] / moments["m00"])
                cY = int(moments["m01"] / moments["m00"])
                cv2.putText(img_resultado, f"#{i+1}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                n_m = best_match["matches"]
                confianza = "Muy Alta" if n_m > 40 else "Alta" if n_m > 20 else "Media"
                
                piezas_encontradas.append({
                    "Pieza ID": f"#{i+1}",
                    "Coincidencias": n_m,
                    "Confianza": confianza,
                    "Ubicacion": f"({cX}, {cY})"
                })
                
                # Marcar origen
                cv2.rectangle(img_piezas_marcadas, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img_piezas_marcadas, f"#{i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return img_resultado, img_piezas_marcadas, piezas_encontradas
