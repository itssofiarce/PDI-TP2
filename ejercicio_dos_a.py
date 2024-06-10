import cv2
from matplotlib import pyplot as plt
import numpy as np

# Defininimos función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)


def proc_patentes(imagen_de_auto):
    """Tratamiento de la imagen a detectar las patentes"""

    source_img = cv2.imread(f'{imagen_de_auto}')

    K_SIZE_GAUSSIAN_BLUR = (1, 19)

    # Blur para deshacernos del ruido y mejorar la detección de bordes
    blur = cv2.GaussianBlur(source_img, K_SIZE_GAUSSIAN_BLUR, 0)

    # Top hat para resaltar la parte clara de las patentes
    # El tamaño del kernel corresponde aprox a las dimesiones de una patente
    filterSize =(17,3) 
    K_TOPHAT = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize) 
    tophat_img = cv2.morphologyEx(blur.copy(),cv2.MORPH_BLACKHAT,K_TOPHAT) 

    # Convierto Escala de Grises
    img_gray = cv2.cvtColor(tophat_img, cv2.COLOR_BGR2GRAY)

    # Aplico cierre para cerrar agujeros en mis elementos
    K_CIERRE = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
    cierre = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, K_CIERRE)
    light = cv2.threshold(cierre, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    return blur,tophat_img, img_gray, light

def marcar_bordes(img_preprocesada, img_original):

    img_original = cv2.imread(img_original)

    # Contornos de las formas
    edges = cv2.findContours(img_preprocesada,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    canvas = np.zeros_like(img_original)
    cv2.drawContours(canvas,edges, -1, (0,255,0), 2)
    return canvas

def contar_pixels_negros(img_original, coord_de_patentes):

    img_original = cv2.imread(img_original)

    region_patente=[]
    max=0
    for recuadro in coord_de_patentes:
        x=recuadro[0]
        y=recuadro[1]
        w=recuadro[2]
        h=recuadro[3]

        patente = img_original[y:y+h, x:x+w]

        number_of_white_pix = np.sum(patente == 0)
        if number_of_white_pix > max:
            region_patente=recuadro

    return region_patente
            

    
def deteccion_de_posibles_patentes(imagen_preproc, img_orginal):
    """Filtrado segun areas de las patentes definidas"""

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_preproc)
    
    min_ancho = 40
    max_ancho = 100
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
    
    # Seleccion del recuadro mayor cantidad de negro, en caso de que se hayan detectado mas de uno
    if len(posibles_patentes) == 1:
        return posibles_patentes[0]
    elif len(posibles_patentes) > 1:
        posibles_patentes = contar_pixels_negros(img_orginal, posibles_patentes)

    return posibles_patentes

def mostrar_areas_detectadas(img_original, coordenadas):
    """Selecciono la parte de la patente en la foto original"""
    x=coordenadas[0]
    y=coordenadas[1]
    w=coordenadas[2]
    h=coordenadas[3]

    patente = img_original[y:y+h, x:x+w]

    return patente

def ploteo_de_etapas(lista_de_img, titulos):

    plt.subplot(241), imshow(lista_de_img[0], new_fig=False, colorbar=False, title=titulos[0])
    plt.subplot(242), imshow(lista_de_img[1], new_fig=False, colorbar=False, title=titulos[1])
    plt.subplot(243), imshow(lista_de_img[2], new_fig=False, colorbar=False, title=titulos[2])
    plt.subplot(244), imshow(lista_de_img[3], new_fig=False, colorbar=False, title=titulos[3])
    plt.subplot(245), imshow(lista_de_img[4], new_fig=False, colorbar=False, title=titulos[4])
    plt.subplot(246), imshow(lista_de_img[5], new_fig=False, colorbar=False, title=titulos[5])
    plt.subplot(247), imshow(lista_de_img[6], new_fig=False, colorbar=False, title=titulos[6])
    plt.show()
    cv2.waitKey(0)


patentes = []
img_patentes = [f'Patentes/img{i:02}.png' for i in range(1, 13)]
    

def main():
    for img in img_patentes:

        # Procesado, Transformacionces y Morfología
        blur, tophat, gris, img_proc = proc_patentes(img)      

        # Detección de componentes
        canvas_con_bordes = marcar_bordes(img_proc, img)

        # Lista con posibles patentes
        coord_de_pat = deteccion_de_posibles_patentes(img_proc, img)
        patentes.append(coord_de_pat)

        # Segmenento de la patente
        img_original = cv2.imread(img)
        patente_img = mostrar_areas_detectadas(img_original, coord_de_pat)

        # Muestro todas las etapas del filtrado
        img_original=cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        todas_imgs=[img_original, blur, tophat, gris, img_proc, canvas_con_bordes, patente_img]
        titulos=["Imagen Original", "Original con blurring", "Top Hat sobre blur", "Top Hat en escala de grises", "Top Hat en grises y con cierre", "Bordes", "Patente"]
        ploteo_de_etapas(todas_imgs, titulos)

#main()





