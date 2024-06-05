import cv2
import numpy as np
import matplotlib.pyplot as plt


placa = cv2.imread("placa.png", cv2.IMREAD_GRAYSCALE)
plt.imshow(placa, cmap='gray');plt.show()

#Eliminar ruido
imagenBorrosa = blur = cv2.blur(placa, (21, 21))
plt.imshow(imagenBorrosa, cmap='gray'), plt.show()

# Aplicar el operador Sobel en la dirección x e y
gradiente_x = cv2.Sobel(imagenBorrosa, cv2.CV_64F, 1, 0, ksize=3)
gradiente_y = cv2.Sobel(imagenBorrosa, cv2.CV_64F, 0, 1, ksize=3)

# Calcular la magnitud del gradiente combinando gradiente_x y gradiente_y
magnitud_gradiente = cv2.magnitude(gradiente_x, gradiente_y)
# Convertir la magnitud del gradiente a un formato de 8 bits para visualizar
magnitud_gradiente = cv2.convertScaleAbs(magnitud_gradiente)

#Umbralar la imagen
retval, imagenUmbralada = cv2.threshold(magnitud_gradiente, 20, 255, cv2.THRESH_BINARY)
plt.imshow(imagenUmbralada, cmap='gray');plt.show()

#Clausura
B = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
Aclau = cv2.morphologyEx(imagenUmbralada, cv2.MORPH_CLOSE, B)
plt.imshow(Aclau, cmap='gray');plt.show()

#Apertura
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
imgAperturada = cv2.morphologyEx(Aclau, cv2.MORPH_OPEN, kernel)
plt.imshow(imgAperturada, cmap='gray');plt.show()

#Cierre
B2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
Aclau2 = cv2.morphologyEx(imgAperturada, cv2.MORPH_CLOSE, B)
plt.imshow(Aclau2, cmap='gray');plt.show()

#Dilatacion
kernel = np.array([[0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0]],np.uint8)
Fd = cv2.dilate(Aclau2, kernel, iterations=13)
plt.imshow(Fd, cmap='gray');plt.show()

#Erosion
kernel = np.array([[0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0]],np.uint8)
Fd2 = cv2.erode(Fd, kernel, iterations=15)
plt.imshow(Fd2, cmap='gray');plt.show()

kernel = np.array([[0,0,0],[1,1,1],[0,0,0]],np.uint8)
Fd3 = cv2.erode(Fd2, kernel, iterations=4)
plt.imshow(Fd3, cmap='gray');plt.show()

#Cierre
B2 = cv2.getStructuringElement(cv2.MORPH_RECT, (50,50))
Aclau3 = cv2.morphologyEx(Fd3, cv2.MORPH_CLOSE, B)
plt.imshow(Aclau3, cmap='gray');plt.show()

#Componentes
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(Fd3, cv2.CV_32S)

im_color = cv2.applyColorMap(np.uint8(255/num_labels*labels), cv2.COLORMAP_JET)
for centroid in centroids:
    cv2.circle(im_color, tuple(np.int32(centroid)), 9, color=(255,255,255), thickness=-1)
for st in stats:
    cv2.rectangle(im_color,(st[0],st[1]),(st[0]+st[2],st[1]+st[3]),color=(0,255,0),thickness=2)
plt.imshow(im_color), plt.show()

#Clasificacion por proporcion
#plt.imshow(placa), plt.show()

resistencia = 70/236
chipset = 0.58

listStats = stats.tolist()
stats
clasificacion = {}

for componente in listStats:
    if componente[2]/componente[3] >= 4.5 and componente[2]/componente[3] <= 0.7 and componente[3] > 200:
        print(componente)
        clasificacion['Chipset'] = componente
#clasificacion
lista = []
for componete in listStats:
    if componete[2]/componete[3] >= 2.7 and componete[2]/componete[3] <= 4 and componente:
        lista.append(componete)
        print(componete)

im_color = cv2.applyColorMap(np.uint8(255/num_labels*labels), cv2.COLORMAP_JET)

for st in lista:
    cv2.rectangle(im_color,(st[0],st[1]),(st[0]+st[2],st[1]+st[3]),color=(0,255,0),thickness=2)
plt.imshow(im_color), plt.show()


# Separa por componente
for label in range(1, num_labels):
    # Crear una máscara para el componente actual
    mask = (labels == label).astype('uint8') * 255

    # Segmentar el componente
    segmented_component = cv2.bitwise_and(Fd, Fd, mask=mask)

    # Mostrar el componente segmentado (opcional)
    cv2.imshow('Componente ' + str(label), segmented_component)
    cv2.waitKey(0)
    cv2.destroyAllWindows()