import cv2
from matplotlib import pyplot as plt
import numpy as np
from ejercicio_dos_a import img_patentes, patentes

for i in range(0,len(img_patentes)):

    imagen_original = cv2.imread(img_patentes[i])
    x=patentes[i][0]
    y=patentes[i][1]
    w=patentes[i][2]
    h=patentes[i][3]

    patente = imagen_original[y:y+h, x:x+w]

    # Convierto a escala de grises
    img_gray_patente = cv2.cvtColor(patente, cv2.COLOR_BGR2GRAY)

    # Binarizo
    umbralizada = cv2.threshold(img_gray_patente, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(umbralizada)

    contornos_de_interes = []
    
    posibles_patentes=[]

    for contorno in range(1, num_labels):
        x = stats[contorno, cv2.CC_STAT_LEFT]
        y = stats[contorno, cv2.CC_STAT_TOP]
        w = stats[contorno, cv2.CC_STAT_WIDTH]
        h = stats[contorno, cv2.CC_STAT_HEIGHT]
        area = stats[contorno, cv2.CC_STAT_AREA]
        ratio = w/h

        if  h/w > 1: 
            contornos_de_interes.append([x,y,w,h])


    # Marcamos las areas con rectangulos
    for cnt in contornos_de_interes:
        x, y, w, h = cnt
        cv2.rectangle(patente, (x, y), (x + w, y + h), (0, 255, 0), 1)

    plt.imshow(patente, cmap='gray'), plt.show()
    