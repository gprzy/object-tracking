import cv2 as cv
import matplotlib.pyplot as plb
import numpy as np
import sys
import json
import time
import imutils
#COISAS QUE POSSAM AJUDAR: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
#FAZER: Encontrar os contur FEITO/ Verificar se ele estiver o formato de uma bola/ Encontrar o centro do contur/ Fazer drawnline de um contor antigo para sua posição atual


#Usar basket.
video = cv.VideoCapture('basket.mp4')

if (video.isOpened()== False):
    print("Erro em abrir o arquivo de video.")
# Vê o video até terminar ou até (q) ser apertado.
while(video.isOpened()):
    #Pega altura e largura do video.
    height = video.get(cv.CAP_PROP_FRAME_HEIGHT)
    width = video.get(cv.CAP_PROP_FRAME_WIDTH)
    ret, frame = video.read()
    if ret == True:
        #Mostra o video um pouco mais lento para ficar mais fácil de ver.
        #time.sleep(0.05)
        #Aplicando Filtros.
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (11, 11), 0)


        thresh, bin = cv.threshold(blurred, 50, 255, cv.THRESH_BINARY_INV)
        #Faz erosion and dilate para remover alguns erros.

        erode = cv.erode(bin, (5,5), iterations=2)
        dilate = cv.dilate(erode, (5,5), iterations=2)

        #Pega os countours.
        contours, img = cv.findContours(dilate,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        center = 0
        #Desenha todos os contours se quise ver.
        #cv.drawContours(frame, contours, -1, (0,255,0), 3)
        #contrs = imutils.grab_contours(contours)

        #Verifica se o contur encontrado e realmente o objeto desejado e se for desenha ele.
        for (i, c) in enumerate(contours):
            (x, y, w, h) = cv.boundingRect(c) 
            # Para mudar o tamanho e tentar encontrar o objeto mude o h(height) e w(width) até que consiga isolar o objeto.
            if w > 20 and h < 30:
                cv.rectangle(
                    img=frame, 
                    pt1=(x, y),
                    pt2=(x + w, y + h),
                    color=(0, 255, 0),
                    thickness=2
                )
            
        #Mostra o video.
        cv.imshow('Binary',dilate)
        cv.imshow('Original',frame)

        # Fecha todas as janelas se 'q' for apertado.
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    else:
            break
video.release()
# Fecha todas as janelas
cv.destroyAllWindows()