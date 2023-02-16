import cv2
import os
from PIL import Image, ImageEnhance

for fileName in os.listdir(r'G:\.shortcut-targets-by-id\1pJlXGXFXW0mhG6UF8qOK_dHbw1yUybKO\Zack Dragos\20029\20029F1\FlameOnly'):
    print(fileName)
    img = cv2.imread(r'G:\.shortcut-targets-by-id\1pJlXGXFXW0mhG6UF8qOK_dHbw1yUybKO\Zack Dragos\20029\20029F1\FlameOnly' + "\\" + fileName)

    (B, G, R) = cv2.split(img)
    #img2 = B - R
    r = cv2.merge((R,R,R))
    cv2.imshow("red",r)

    img2 = cv2.subtract(img,r)
    cv2.imshow("img", img2)

    img3 = cv2.subtract(G,R)
    img4 = cv2.add(G,B)
    #cv2.imshow('img', img2)
    im = img2.load()
    for i in range(img2.size[0]):
        for j in range(img2.size[1]):
            if im[i,j] > 3:
                print("Hi")
    cv2.imwrite(os.path.join(r'G:\.shortcut-targets-by-id\1pJlXGXFXW0mhG6UF8qOK_dHbw1yUybKO\Zack Dragos\20029\20029F1\SubtractedImages', fileName), img2)