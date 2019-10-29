
import numpy as np
import cv2
imagem = cv2.imread('convite.jpg')
vermelho = (0, 0, 255)
verde = (0, 255, 0)
azul = (255, 0, 0)
cv2.line(imagem, (0, 0), (100, 200), verde)
cv2.line(imagem, (300, 200), (150, 150), vermelho, 5)
cv2.rectangle(imagem, (20, 20), (120, 120), azul, 10)
cv2.rectangle(imagem, (200, 50), (225, 125), verde, -1)
(X, Y) = (imagem.shape[1] // 2, imagem.shape[0] // 2)
for raio in range(0, 175, 15):
    cv2.circle(imagem, (X, Y), raio, vermelho)
cv2.imshow("Desenhando sobre a imagem", imagem)
cv2.waitKey(0)


# clocando quadrados amarelos na imagem

import cv2
imagem = cv2.imread('convite.jpg')
for y in range(0, imagem.shape[0], 10): #percorre linhas
  for x in range(0, imagem.shape[1], 10): #percorre colunas
    imagem[y:y+5, x: x+5] = (0,255,255)
cv2.imshow("Imagem modificada", imagem)
cv2.waitKey(0)

 import cv2

 img = cv2.imread('convite.jpg')
 cv2.imshow("Original", img)
 flip_horizontal = img[::-1,:] #comando equivalente abaixo
 #flip_horizontal = cv2.flip(img, 1) 
 
 cv2.imshow("Flip Horizontal", flip_horizontal)              
 flip_vertical = img[:,::-1] #comando equivalente abaixo 
 #flip_vertical = cv2.flip(img, 0) 
 
 cv2.imshow("Flip Vertical", flip_vertical) 
 flip_hv = img[::-1,::-1] #comando equivalente abaixo  
#flip_hv = cv2.flip(img, -1) cv2.imshow("Flip Horizontal e Vertical", flip_hv) cv2.waitKey(0)


import numpy as np 
import cv2 
 
img = cv2.imread('convite.jpg')
cv2.imshow("Original", img)
img_redimensionada = img[::2,::2] 
 
cv2.imshow("Imagem redimensionada", img_redimensionada)
cv2.waitKey(0)  