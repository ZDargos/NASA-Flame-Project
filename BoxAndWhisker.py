import plotly.express as px
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
import os

img = Image.open(r'G:\.shortcut-targets-by-id\1XOMUuLHiM5Iwz7XVbtov9F6fcQxzI93L\JS Research Students\Ethan Catalanello\Cases\20029\20029F1\20029F1_Images\20029F1B_LLL-UV\20029F1B_00091.tif')
im = img.load()
width, height = img.size
pxIntensityR = []
pxIntensityG = []
pxIntensityB = []
xvals = []
count5 = 0
for i in range(img.size[0]):
    for j in range(img.size[1]):
        if im[i, j] > 3:
            pxIntensityR.append(im[i, j])
        # if im[i, j] > 3:
        #     pxIntensityG.append(im[i, j])
        # if im[i, j] > 3:
        #     pxIntensityB.append(im[i, j])
        count5 += 1
print(im[5,5])
pxIntensity = [pxIntensityR, pxIntensityG, pxIntensityB]
plt.boxplot(pxIntensity, labels= ["Red", "Green", "Blue"])
#print(pxIntensityR)
plt.show()
print("hi")

