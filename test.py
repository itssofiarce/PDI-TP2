import cv2
import matplotlib.pyplot as plt

# Load the image
source_img = cv2.imread('Patentes/img01.png')

# Convert to grayscale
img_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

# Apply Canny edge detection to find edges in the image
edges = cv2.Canny(blur, 50, 150)

# Apply morphological closing to fill gaps and enhance edges
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Find contours in the edge-detected image
contornos, _ = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar rectángulos alrededor de los contornos con altura mayor o igual a 140
for contorno in range(1,contornos):

    x, y, w, h = cv2.boundingRect(contorno)
    #if h >= 40:  # Filtrar contornos con altura mayor o igual a 140
    cv2.rectangle(source_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Mostrar la imagen con rectángulos alrededor de los contornos seleccionados
plt.imshow(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


