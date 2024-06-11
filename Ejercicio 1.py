import cv2
import numpy as np
import matplotlib.pyplot as plt
from Funciones_ejercicio_1 import detectar_chip, detectar_resistencias, detectar_capacitores, resaltar_componentes_con_diccionario, detectar_y_resaltar_componentes

#1-a

#Leemos la imágen de la placa
placa = cv2.imread("placa.png", cv2.IMREAD_GRAYSCALE)
plt.imshow(placa, cmap='gray'); plt.show()

placa_original = cv2.imread("placa.png").cvtColor(placa, cv2.COLOR_BGR2RGB)
plt.imshow(placa_original); plt.show()

#Primero detectamos el CHIP
#Tratamos la imagen:

#Eliminamos ruido
imagenBorrosa = blur = cv2.blur(placa, (21, 21))
#plt.imshow(imagenBorrosa, cmap='gray'), plt.show()

#Aplicamos el operador Sobel en la dirección x e y
gradiente_x = cv2.Sobel(imagenBorrosa, cv2.CV_64F, 1, 0, ksize=3)
gradiente_y = cv2.Sobel(imagenBorrosa, cv2.CV_64F, 0, 1, ksize=3)

#Calculamos la magnitud del gradiente combinando gradiente_x y gradiente_y
magnitud_gradiente = cv2.magnitude(gradiente_x, gradiente_y)

#Convertimos la magnitud del gradiente a un formato de 8 bits para visualizar
magnitud_gradiente = cv2.convertScaleAbs(magnitud_gradiente)

#Umbralar la imagen
retval, imagenUmbralada = cv2.threshold(magnitud_gradiente, 20, 255, cv2.THRESH_BINARY)
#plt.imshow(imagenUmbralada, cmap='gray');plt.show()

#El chip quedó dividido. Probamos: clausura
B = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
imgClausura1 = cv2.morphologyEx(imagenUmbralada, cv2.MORPH_CLOSE, B)
#plt.imshow(Aclau, cmap='gray');plt.show()

#Apertura
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
imgAperturada = cv2.morphologyEx(imgClausura1, cv2.MORPH_OPEN, kernel)
#plt.imshow(imgAperturada, cmap='gray');plt.show()

#Clausura 2
B2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
imgClausura2 = cv2.morphologyEx(imgAperturada, cv2.MORPH_CLOSE, B)
#plt.imshow(imgClausura2, cmap='gray');plt.show()

#Dilatacion
kernel = np.array([[0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0]],np.uint8)
imgDilatada = cv2.dilate(imgClausura2, kernel, iterations=13)
#plt.imshow(imgDilatada, cmap='gray');plt.show()

#Erosion
kernel = np.array([[0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0]],np.uint8)
imgErosionada = cv2.erode(imgDilatada, kernel, iterations=15)
#plt.imshow(imgErosionada, cmap='gray');plt.show()

kernel = np.array([[0,0,0],[1,1,1],[0,0,0]],np.uint8)
imgErosionada2 = cv2.erode(imgErosionada, kernel, iterations=4)
#plt.imshow(imgErosionada2, cmap='gray');plt.show()

#Clausura 3
B2 = cv2.getStructuringElement(cv2.MORPH_RECT, (50,50))
imgClausura3 = cv2.morphologyEx(imgErosionada2, cv2.MORPH_CLOSE, B)
plt.imshow(imgClausura3, cmap='gray');plt.show()

#La imagen nos satisface. Identificamos componentes:
chip = detectar_chip(imgClausura3)

#----------------------------------
#Vamos a detectar las RESISTENCIAS
#En la siguiente imagen vemos que hay una resistencia que
#fue identificada como dos componentes distintas. Debemos corregir eso
plt.imshow(imgErosionada2[1600:1700,1000:1200], cmap='gray'),plt.show()

#Aplicaremos una dilatacion para corregir esto
kernel = np.array([[1,1,1],[1,1,1],[1,1,1]],np.uint8)
imagenParaResistencias = cv2.dilate(imgErosionada2, kernel, iterations=1)
#plt.imshow(imagenParaResistencias, cmap='gray');plt.show()

#Recortamos la imagen para que no tome tanto ruido
imagenParaResistenciasRecortada = imagenParaResistencias[900:,:]
plt.imshow(imagenParaResistenciasRecortada, cmap='gray'),plt.show()

#Calculamos las componentes nuevamente
resistencias = detectar_resistencias(imagenParaResistenciasRecortada)


#Objetivo logrado
#--------------------------------------------
#Queda detectar los CAPACITORES

#Para los capacitores partiremos de la imagen base y realizaremos un umbralado
#que solo permita pasar los valores muy altos.

#Umbralar la imagen
retval, imagenUmbralada = cv2.threshold(placa, 200, 255, cv2.THRESH_BINARY)
#plt.imshow(imagenUmbralada, cmap='gray');plt.show()

#Clausura o cierre
B2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
imagenClausura = cv2.morphologyEx(imagenUmbralada, cv2.MORPH_CLOSE, B)

#Erosion
kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
imgErosionada = cv2.erode(imagenClausura, kernel, iterations=5)
plt.imshow(imgErosionada, cmap='gray');plt.show()

#Nos satisface. Como antes, buscaremos las componentes conectadas de acuerdo a ciertas proporciones. 
#En este caso, extraeremos aquellas que conservan una mayor area
#y un ancho similar al alto
capacitores = detectar_capacitores(imgErosionada)

#-------------------------------------------------
#Mostramos imagen final con TODAS LAS COMPONENTES REMARCADAS
imgFinal = placa_original.copy()
resaltar_componentes_con_diccionario(imgFinal,chip,(255,0,0))
resaltar_componentes_con_diccionario(imgFinal,resistencias,(0,255,0))
resaltar_componentes_con_diccionario(imgFinal,capacitores,(0,0,255))

#Generamos img de salida
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
plt.title('COMPONENTES DETECTADAS', fontsize=9.5)
txt="rojo: chip\nazul: capacitores\nverde: resistencias"
props = dict(boxstyle='round', facecolor='grey', alpha=0.15)  # bbox features
ax.text(1.03, 0.98, txt, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=props)
plt.imshow(imgFinal),plt.show()

#-----------------------------------------
#1-b: Vamos a clasificar capacitores por áreas, adjudicando una tupla con el menor y mayor valor de área a c/u:
clasificacion_capacitores = {
    'capacitor_chico':(0,7500),
    'capacitor_medio':(7500,90000),
    'capacitor_grande': (90000,999999)
}
#Los agrupamos:
capacitores_chicos = []
capacitores_medios = []
capacitores_grandes = []

for capacitor in capacitores.keys():
    if capacitores[capacitor][4] < clasificacion_capacitores['capacitor_chico'][1]:
        capacitores_chicos.append(capacitor)
    elif capacitores[capacitor][4] > clasificacion_capacitores['capacitor_grande'][0]:
        capacitores_grandes.append(capacitor)
    else:
        capacitores_medios.append(capacitor)

#Contamos cuántos hay de cada uno
total_capacitores_chicos = len(capacitores_chicos)
total_capacitores_medios = len(capacitores_medios)
total_capacitores_grandes = len(capacitores_grandes)

total_capacitores_grandes

#Imágen de salida
fig, ax = plt.subplots(figsize=(5, 5))
plt.title('Capacitores encontrados', fontsize=20)
texto = f"""
Chicos = {total_capacitores_chicos}
Medios = {total_capacitores_medios}
Grandes = {total_capacitores_grandes}
"""
plt.text(0.5, 0.5, texto, horizontalalignment='center', verticalalignment='center', fontsize=18)
ax.axis('off')
plt.show()



#-----------------------------------------
#1-c: Contamos resistencias eléctricas:
contador_resistencias = 0
for i in resistencias.keys():
    contador_resistencias += 1
print('Las resistencias eléctricas detectadas en total son: ',contador_resistencias)