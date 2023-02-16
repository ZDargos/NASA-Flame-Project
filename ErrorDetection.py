from PIL import Image
import os
import math
import glob
pic = 'removed''
MainDir = 'removed'
secDir = 'removed'
def PercentCheck(count, total):
    if count == math.floor(total/20):
        print('completed 5%   ', count, '/', total)
    if count == math.floor(total/8):
        print('completed 12.5%   ', count, '/', total)
    if count == math.floor(total/4):
        print('completed 25%   ', count, '/', total)
    if count == math.floor(total*3/8):
        print('completed 37.5%   ', count, '/', total)
    if count == math.floor(total / 2):
        print('completed 50%   ', count, '/', total)
    if count == math.floor(total*5/8):
        print('completed 62.5%   ', count, '/', total)
    if count == math.floor(total*3/4):
        print('completed 75%   ', count, '/', total)
    if count == math.floor(total*7/8):
        print('completed 87.5%   ', count, '/', total)


def errorCheck(dir):
    print(dir)
    total = len(os.listdir(dir))
    count = 0
    images = [dir]
    for fileName in sorted(os.listdir(dir)):
        if 'tif' in fileName:
            try:
                im = Image.open(dir + '\\' + fileName)
            except:
                images.append(fileName)
                print(fileName)
        count = count + 1
        PercentCheck(count,total)
    return images

def findErrors(mainDir):
    all = []
    count = 0
    if os.path.exists(mainDir + '\\BrokenImages.txt') == False:
        for file in glob.iglob(mainDir + '/**/*INTNS_ALL', recursive = True):
            count = count + 1
            all.append(errorCheck(file))

        with open(mainDir + '\\BrokenImages.txt', 'w') as f:
            f.write(mainDir)
        for i in range(count):
            with open(mainDir + '\\BrokenImages.txt', 'a') as f:
                f.write('\n')
                f.write('\n')
                f.write(all[i][0])
                for j in range(len(all[i])-1):
                    f.write('\n')
                    f.write(all[i][j+1])
# for folder in os.listdir(MainDir):
#     print(folder)
#     findErrors(MainDir + "\\" + folder)


#all.append(errorCheck(secDir))
#print(all[0][1])
#im = Image.open(pic)
