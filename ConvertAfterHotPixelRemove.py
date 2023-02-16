import numpy as np
from numpy import asarray
import cv2
from PIL import Image
import struct, time, os


def read_hobj(image_filename):
    start_time = time.time()
    print('Getting HOBJ file >>> {:s}'.format(image_filename))
    with open(image_filename, 'rb') as f:
        ibuffer = f.read(84)

        npixels = struct.unpack('i', ibuffer[72:76][::-1])[0]
        rows = struct.unpack('h', ibuffer[80:82][::-1])[0] + 1
        cols = struct.unpack('h', ibuffer[82:84][::-1])[0] + 1

        print('Image width/height is {:d}/{:d}.'.format(cols, rows))
        print('# of pixels is {:d}.'.format(npixels))

        f.seek(84 + rows * 6 + 17)
        image_str = f.read(npixels * 2)
        bit_format = '<{:d}{:s}'.format(npixels, 'h')
        image_array = struct.unpack(bit_format, image_str)
        print('Read time: {:.2f} seconds.'.format(time.time() - start_time))
    return np.array(image_array).reshape(rows, cols)


imgFolder = os.path.join('removed')
saveDir = os.path.join('removed')
sclSaveDir = os.path.join('removed')

for fileName in os.listdir(imgFolder):
    imgPath = os.path.join(imgFolder, fileName)
    fileNameNoExt = fileName[0:(len(fileName) - 4)]

    fileNameTiff = fileNameNoExt + 'tiff'
    imgArr = read_hobj(imgPath)


    im = Image.fromarray(imgArr)
    px = im.load()
    pixels = im.load()
    coordinate2 = x, y = 0, 0
    # Go through every pixel and read its value
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            coordinate = x, y = i, j
            if im.getpixel(coordinate) >= 50:
                if i <= 1300 and j <= 1020:
                    if (px[i + 2, j] + px[i - 2, j] + px[i, j + 2] + px[i, j - 2]) / 4 <= px[i, j] / 2:
                        pixels[i, j] = 1
    im.save(saveDir + "\\" + fileNameTiff)
    img1 = asarray(im)
    sclImgArr = img1 - np.min(img1)
    sclImgArr = (sclImgArr * float(np.iinfo(np.uint16).max) / np.max(sclImgArr)).astype(np.uint16)

    imgArr = imgArr.astype(np.uint16)

    sclImgArr = cv2.cvtColor(sclImgArr, cv2.COLOR_BayerRG2RGB)

    fileNameNoExt = fileName[0:(len(fileName) - 4)]

    fileNameTiff = fileNameNoExt + 'tiff'

    #cv2.imwrite(os.path.join(saveDir, fileNameTiff), imgArr)
    cv2.imwrite(os.path.join(sclSaveDir, fileNameTiff), sclImgArr)
