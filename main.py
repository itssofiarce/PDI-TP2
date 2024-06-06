# Imagen escala de grises
# Detectar bordes
# Aplicar Morfologia, rellenar 
# Suavizar
# Filtar en componentes conectadas
# Por grupos de 3 o de 6
import cv2
from matplotlib import pyplot as plt
import numpy as np

# --- Cargo imagen ----------------------------------------------------------------------
source_img = cv2.imread('Patentes/img014.png')

# --- Convierto Escala de Grises ----------------------------------------------------------------------
img_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
umbral = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY_INV)[1]
#plt.imshow(umbral, cmap='gray'), plt.show()

# Lista de objetos con los puntos que definen cada forma. Contornos de las formas
edges = cv2.findContours(umbral,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0] 
canvas = np.zeros_like(source_img)
cv2.drawContours(canvas,edges, -1, (0,255,0), 2)
#plt.imshow(canvas, cmap='gray'), plt.show()

# Rellenamos los huecos chiquitos con cierre
kernel = np.ones((5,5),np.uint8)
erosion = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
#plt.imshow(erosion, cmap='gray'), plt.show()

# Nos quedamos con los bordes de la diferencia entre la dialtiacion y erosion
gradient = cv2.morphologyEx(erosion, cv2.MORPH_GRADIENT, kernel)
gradient_g = cv2.cvtColor(gradient, cv2.COLOR_BGR2GRAY)
#plt.imshow(gradient, cmap='gray'), plt.show()

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gradient_g)

area=1214
posibles_pat=[]
for contorno in range(1, num_labels):
    x = stats[contorno, cv2.CC_STAT_LEFT]
    y = stats[contorno, cv2.CC_STAT_TOP]
    w = stats[contorno, cv2.CC_STAT_WIDTH]
    h = stats[contorno, cv2.CC_STAT_HEIGHT]
    area = stats[contorno, cv2.CC_STAT_AREA]
    if 537 < area < 2349 and 1.739 < w/h < 3.7:
        posibles_pat.append([x,y,w,h])
        #posibles_pat.append(((x, y), (x + w, y + h)))



# Not in img 6, 5, 4, 11


