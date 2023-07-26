import cv2
import pytesseract
from PIL import Image
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

cap = cv2.VideoCapture("multimedia/placasViedoP.mp4")  # Abrir el archivo de video
pausa = False

while True:
    if not pausa:
        ret, frame = cap.read()  # Leer un fotograma del video
        if not ret:
            break

    # Preprocesamiento de la imagen
    grises = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises reducir la complejidad del procesamiento
    grises = cv2.GaussianBlur(grises, (5, 5), 0)  # Aplicar un desenfoque gaussiano para suabizar la imagen reducir el ruido, detalles
    contorno = cv2.Canny(grises, 50, 150)  # Detectar bordes usando el operador de Canny (Algoritmo reslta los bordes)

    # Detección de contornos en imagen binaria
    contours, _ = cv2.findContours(contorno, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Inicialización de la mejor placa y su área
    mejor_placa = None
    mejor_area_placa = 0

    # Iteración sobre los contornos
    for contour in contours:
        # Aproximación del contorno a un polígono
        #La aproximación poligonal se realiza para simplificar la forma del contorno
        # reducir la cantidad de información necesaria para representarla.
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Filtrado de contornos no cuadriláteros
        if len(approx) != 4: #si no tienes 4 lados el contorno de la placa se pasa al siguiente
            continue

        # Cálculo que tan grande es el área del contorno nos ayuda a filtar contornos
        area = cv2.contourArea(approx)

        # Filtrado de contornos pequeños y grandes
        if area < 2000 or area > 50000:
            continue

        # Actualización de la mejor placa
        if area > mejor_area_placa: #compara el área del contorno actual con el área del mejor contorno encontrado hasta ahora
            mejor_placa = approx #Si el área del contorno actual es mayor, se actualiza la variable best_plate con el contorno actual
            mejor_area_placa = area # se actualiza el area

    # Extracción de la región de la mejor placa
    if mejor_placa is not None: #si no es nula
        x, y, w, h = cv2.boundingRect(mejor_placa) # se obtiene las coordenas del rectangulo de mejor_placa
        plate_roi = grises[y:y + h, x:x + w]  #extrae la región de interés (ROI) de la imagen en escala de grises gray que corresponde a la mejor placa

        # Convertir la región de la placa a formato de imagen PIL
        placa_pil = Image.fromarray(plate_roi)

        # Aplicación de Tesseract OCR a la región de la placa
        texto_placa = pytesseract.image_to_string(placa_pil, config='--psm 7')

        # Dibujo del contorno y texto en el fotograma
        cv2.drawContours(frame, [mejor_placa], -1, (0, 0, 255), 2)#dibujamos el contorno en RGB verde, grozo de linea 2
        cv2.putText(frame, texto_placa, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Imprimir el texto de la placa en la consola
        print("Texto de la placa:", texto_placa)

    # Mostrar el fotograma actual
    cv2.imshow('Video', frame)

    key = cv2.waitKey(1)
    if key == ord('x'):  # Terminar la ejecución al presionar la tecla 'x'
        break
    elif key == ord(' '):  # Tecla de espacio para pausar/reanudar
        pausa = not pausa

cap.release()  # Liberar el archivo de video
cv2.destroyAllWindows()  # Cerrar las ventanas
