import zipfile
import glob
import os
from PIL import Image
from skimage.filters.rank import median
from skimage.morphology import disk
from skimage import data, io
from matplotlib import pyplot as plt
from skimage.filters import gaussian
zip = 'removed'
save = 'removed'



def UnZip(zipFileName, saveDirFolder):
    with zipfile.ZipFile(zipFileName, "r") as zip_ref:
        zip_ref.extractall(saveDirFolder)

def UnZipAll(CaseFolder, FolderType):
    for file in glob.iglob(CaseFolder + '/**/*' + FolderType,recursive=True):
        UnZip(file,file[:-4])

# im1 = io.imread('removed')
# im1 = median(im1, disk(5))
# im1 = gaussian(im1, sigma =1, truncate = 5)
# io.imshow(im1, cmap = 'gray')
# plt.show()
avg = 0
pxtotal = 0
im1 = Image.open('removed')

def avgCalc(im):
    pxtotal = 0
    pX = im.load()
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            pxtotal = pX[i, j] + pxtotal
    avg = pxtotal / (im.size[1] * im.size[0])
    return avg
ints = [1]

x = 1
for i in range(34):
    x = x+1
    ints.append(i)
    print(i, x)
print(x)
print(ints.size)
#print(avgCalc(im1))

#UnZipAll(r'G:\.shortcut-targets-by-id\12r7NVnUc9cMtuw-IqvGIc2uNipmfyODe\Cases\19254', 'INTNS.zip')
