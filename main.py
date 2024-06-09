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

PATENTES = [f'Patentes/img{i:02}.png' for i in range(1, 13)]
for patente in PATENTES:
    source_img = cv2.imread(f'{patente}')
    #plt.imshow(source_img, cmap='gray'), plt.show()

    # Top hat para resaltar la parte clara de las patentes
    # El tamaño del kernel corresponde aprox a las dimesiones de una patente
    filterSize =(17,3) 
    K_TOPHAT = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize) 
    tophat_img = cv2.morphologyEx(source_img.copy(),cv2.MORPH_BLACKHAT,K_TOPHAT)
    #plt.imshow(tophat_img, cmap='gray'), plt.show() 

    # Convierto Escala de Grises
    img_gray = cv2.cvtColor(tophat_img, cv2.COLOR_BGR2GRAY)
    #plt.imshow(img_gray, cmap='gray'), plt.show() 

    K_CIERRE = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    cierre = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, K_CIERRE)
    light = cv2.threshold(cierre, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #plt.imshow(light, cmap='gray'), plt.show() 

    # Contornos de las formas
    edges = cv2.findContours(light,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    canvas = np.zeros_like(source_img)
    cv2.drawContours(canvas,edges, -1, (0,255,0), 2)
    #plt.imshow(canvas, cmap='gray'), plt.show()

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(light)
    
    min_ancho=30
    max_ancho=50
    min_altura=15
    max_altura=25
    min_area = 562 # sacado de la foto con menor area
    max_area = 2950 
    componen=[]
    posibles_pat_2=[]

    for contorno in range(1, num_labels):
        x = stats[contorno, cv2.CC_STAT_LEFT]
        y = stats[contorno, cv2.CC_STAT_TOP]
        w = stats[contorno, cv2.CC_STAT_WIDTH]
        h = stats[contorno, cv2.CC_STAT_HEIGHT]
        area = stats[contorno, cv2.CC_STAT_AREA]

        if 537 < area < 2549 and 1.739 < w/h < 4.7:
        #if  w < 145 and h > 17 and h < 60 and area > min_area and area < max_area: #and h < 15 and area < min_area: 
            #if h/w > :
            # Chequeo si no se detectó otra patente

            # Agregar comparacion con las otras componentes dentro de la lista
            componen.append([x,y,w,h,area])
            posibles_pat[f'{patente}']=componen    
            posibles_pat_2.append([x,y,w,h])

            #posibles_pat_2.append(((x, y), (x + w, y + h)))

    if posibles_pat_2:
        for box in posibles_pat_2:
            x, y, w, h = box
            cv2.rectangle(source_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

      # Display the image with drawn bounding boxes
    plt.imshow(source_img),plt.show()

# Dibujar el rectangulo en la foto original


# Not in img 6, 5, 4, 11


