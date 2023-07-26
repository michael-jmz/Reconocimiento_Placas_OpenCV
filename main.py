import cv2
import imutils
import pytesseract
import winsound

# Instala el directorio de Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Crea una función para leer la base de datos
def buscar_texto_en_archivo(nombre_archivo, texto_a_buscar):
    with open(nombre_archivo, 'r') as archivo_lectura:
        for linea in archivo_lectura:
            if texto_a_buscar in linea:
                return True
    return False

# Inicializa el flujo de video (captura de cámara)
captura_video = cv2.VideoCapture(0)

while True:
    # Captura cada fotograma del flujo de video
    ret, fotograma = captura_video.read()

    # Redimensiona y estandariza el fotograma
    fotograma = imutils.resize(fotograma, width=500)#Al redimensionar el fotograma a un tamaño estándar, se simplifica el procesamiento posterior

    # Convierte el fotograma a escala de grises
    gris = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)

    # Reduce el ruido del fotograma y lo suaviza
    gris = cv2.bilateralFilter(gris, 11, 17, 17)

    # Encuentra los bordes en el fotograma
    bordes = cv2.Canny(gris, 170, 200) #umbral min y max

    # Encuentra los contornos en el fotograma
    contornos, _ = cv2.findContours(bordes.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Sorted Ordena los contornos según sus áreas y toma los 30 más grandes reverse=orden descendente
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:30]
    placa_matricula_contorno = None

    # Recorre los contornos para encontrar la placa de matrícula
    for contorno in contornos:
        perimetro = cv2.arcLength(contorno, True)
        aproximacion = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
        if len(aproximacion) == 4:
            placa_matricula_contorno = aproximacion
            x, y, ancho, alto = cv2.boundingRect(contorno) #calcula un rectángulo delimitador que rodea el contorno
            imagen_recortada = fotograma[y:y + alto, x:x + ancho]# alamacena la región de la placa de matrícula recortada del fotograma original.

            # Utiliza pytesseract para convertir la imagen recortada a texto
            texto = pytesseract.image_to_string(imagen_recortada, lang='eng')
            texto = ''.join(e for e in texto if e.isalnum())  # Modifica el texto, eliminando espacios

            # Verifica si el número reconocido está en la base de datos
            frecuencia = 2500
            duracion = 1200
            if buscar_texto_en_archivo('./Database/Database.txt', texto) and texto != "":
                print('Registrado', texto)
                winsound.Beep(frecuencia, duracion)
            else:
                print("No Registrado", texto)

            # Dibuja un rectángulo alrededor de la placa reconocida en el fotograma original
            cv2.rectangle(fotograma, (x, y), (x + ancho, y + alto), (0, 255, 0), 2)

            # Coloca el texto de la placa reconocida en el fotograma
            cv2.putText(fotograma, texto, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            break

    # Muestra la imagen final con los contornos
    cv2.imshow("Imagen Final", fotograma)

    # Rompe el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera el objeto de captura de video y cierra todas las ventanas
captura_video.release()
cv2.destroyAllWindows()
