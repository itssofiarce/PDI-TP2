# Imagen escala de grises
# Detectar bordes
# Aplicar Morfologia, rellenar 
# Suavizar
# Filtar en componentes conectadas
# Por grupos de 3 o de 6
import cv2
from matplotlib import pyplot as plt
import numpy as np

# Cargo imagen 

posibles_pat={}
mas_un_pat=[]

PATENTES = [f'Patentes/img{i:02}.png' for i in range(1, 13)]
#PATENTES = [f'Patentes/img{i:02}.png' for i in range(1, 2)]

def proc_patentes(imagen_de_auto):
    """Tratamiento de la imagen a detectar las patentes"""

    source_img = cv2.imread(f'{imagen_de_auto}')
    #plt.imshow(source_img, cmap='gray'), plt.show()

    K_SIZE_GAUSSIAN_BLUR = (1, 19)

    # Blur y detección de bordes
    blur = cv2.GaussianBlur(source_img, K_SIZE_GAUSSIAN_BLUR, 0)
    
    # Top hat para resaltar la parte clara de las patentes
    # El tamaño del kernel corresponde aprox a las dimesiones de una patente aprox
    filterSize =(17,3) 
    K_TOPHAT = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize) 
    tophat_img = cv2.morphologyEx(blur.copy(),cv2.MORPH_BLACKHAT,K_TOPHAT)
    #plt.imshow(tophat_img, cmap='gray'), plt.show() 

    # Convierto Escala de Grises
    img_gray = cv2.cvtColor(tophat_img, cv2.COLOR_BGR2GRAY)
    #plt.imshow(img_gray, cmap='gray'), plt.show() 

    K_CIERRE = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
    cierre = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, K_CIERRE)
    light = cv2.threshold(cierre, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #plt.imshow(light, cmap='gray'), plt.show() 

    return light

def marcar_bordes(img_preprocesada, img_original):

    # Contornos de las formas
    edges = cv2.findContours(img_preprocesada,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    canvas = np.zeros_like(img_original)
    cv2.drawContours(canvas,edges, -1, (0,255,0), 2)
    return canvas
    #plt.imshow(canvas, cmap='gray'), plt.show()
    
def deteccion_de_posibles_patentes(imagen_preproc):
    """Filtrado segun areas de las patentes definidas"""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_preproc)
    
    min_ancho=40
    max_ancho=98
    min_area = 562 
    max_area = 2440 

    posibles_patentes=[]

    for contorno in range(1, num_labels):
        x = stats[contorno, cv2.CC_STAT_LEFT]
        y = stats[contorno, cv2.CC_STAT_TOP]
        w = stats[contorno, cv2.CC_STAT_WIDTH]
        h = stats[contorno, cv2.CC_STAT_HEIGHT]
        area = stats[contorno, cv2.CC_STAT_AREA]
        ratio = w/h

        if min_area < area < max_area and 1.739 < ratio < 3.7 and min_ancho < w < max_ancho:  
            posibles_patentes.append([x,y,w,h])

    return posibles_patentes

def mostrar_areas_detectadas(imagen, coordenadas):
    """Dibujar el rectangulo en la foto original"""
    for box in coordenadas:
        x, y, w, h = box
        cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the image with drawn bounding boxes
    plt.imshow(imagen),plt.show()

def contar_pixels_blancos(img_original, coord_de_patentes):
    if len(coord_de_patentes) > 1:
        for recuadro in coord_de_patentes:
            

def main():

    img_patentes = [f'img/patentes/img{i:02}.png' for i in range(1, 13)]
    
    patentes = list()

    for img in img_patentes:

        # Procesado, Transformacionces y Morfología
        img_proc = proc_patentes(img)

        # Detección de componentes
        img_bordes = marcar_bordes(img_proc, img)

        # Lista con posibles patentes
        coord_de_pat = deteccion_de_posibles_patentes(img_proc)

        # Mostrar Procesado
        #copiar imshow para mostrar todas juntas o agregar ey llamar desde la primera y segunda funcion

        # Seleccion del recuadro mayor cantidad de blanco
        coord_de_pat = contar_pixels_blancos(img, coord_de_pat)



        patente.append(procesar_patente(img))

    for pat in patentes:
        plt.figure()
        plt.imshow(
                segmentar_caracteres(pat, box=True),
                cmap='gray')
        plt.show()

main()





