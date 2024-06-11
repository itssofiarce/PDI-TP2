import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

def resaltar_componentes_con_diccionario(img, dicc, color):
    for i in dicc.keys():
        cv2.rectangle(img,(dicc[i][0],dicc[i][1]),
                    (
                        dicc[i][0]+dicc[i][2],
                        dicc[i][1]+dicc[i][3]),
                        color=color,
                        thickness=15
                    )
    return img

def detectar_y_resaltar_componentes(imagen):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen, cv2.CV_32S)
    im_color = cv2.applyColorMap(np.uint8(255/num_labels*labels), cv2.COLORMAP_JET)
    img_resaltada = im_color.copy()
    for centroid in centroids:
        cv2.circle(img_resaltada, tuple(np.int32(centroid)), 9, color=(255,255,255), thickness=-1)
    for st in stats:
        cv2.rectangle(img_resaltada,(st[0],st[1]),(st[0]+st[2],st[1]+st[3]),color=(0,255,0),thickness=2)
    return num_labels, labels, stats, centroids, im_color, img_resaltada

def detectar_chip(img):
    num_labels, labels, stats, centroids, im_color, img_resaltada = detectar_y_resaltar_componentes(img)
    chip = {}
    for componente in stats:
        x = componente[0]
        y = componente[1]
        width = componente[2]
        height = componente[3]
        area = componente[4]

        if 0.4 <= width/height <= 0.55 and area > 10000:
            chip['Chipset'] = componente
    plt.imshow(resaltar_componentes_con_diccionario(im_color,chip,(0,255,0))),plt.show()
    return chip

def detectar_resistencias(img):
    num_labels, labels, stats, centroids, im_color, img_resaltada = detectar_y_resaltar_componentes(img)
    resistencias = {}
    resistencias_relativas_img_original = {} #la imagen que se pasa está recortada, y las coordenadas de las componentes se relativizan a ella. Vamos a arreglarlas
    contador = 1
    for componente in stats:
        x = componente[0]
        y = componente[1]
        width = componente[2]
        height = componente[3]
        area = componente[4]

        maxWidth = 230
        maxHeight = 80

        if height < width:
            if width <= maxWidth and height <= maxHeight and area > 4000:
                resistencias[f'Resistencia_{contador}'] = componente
                componente_modificado = copy.deepcopy(componente)
                componente_modificado[1] += 900
                resistencias_relativas_img_original[f'Resistencia_{contador}'] = componente_modificado
                contador+=1
        else:
            if width <= maxHeight and height <= maxWidth and area > 4000:
                resistencias[f'Resistencia_{contador}'] = componente
                componente_modificado = copy.deepcopy(componente)
                componente_modificado[1] += 900
                resistencias_relativas_img_original[f'Resistencia_{contador}'] = componente_modificado
                contador +=1
    plt.imshow(resaltar_componentes_con_diccionario(im_color,resistencias,(0,255,0))),plt.show()
    return resistencias_relativas_img_original

def detectar_capacitores(img):
    num_labels, labels, stats, centroids, im_color, img_resaltada = detectar_y_resaltar_componentes(img)
    capacitores= {}
    contador = 1
    for componente in stats:
        x = componente[0]
        y = componente[1]
        width = componente[2]
        height = componente[3]
        area = componente[4]

        if 0.8 < abs(width/height) < 1.3 and 3100 < area < 140000:
            capacitores[f"Capacitor_{contador}"] = componente
            contador+=1
    plt.imshow(resaltar_componentes_con_diccionario(im_color,capacitores,(0,255,0))),plt.show()
    return capacitores