import struct, time, os, cv2, math, glob, matplotlib, pathlib
import numpy as np
from numpy import asarray
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from skimage import io
from skimage.transform import warp_polar
from skimage.filters import gaussian
import pandas as pd
from skimage.filters.rank import median
from skimage.morphology import disk


def ACMEdirectoryCheck(file):
    '''
        This function will check the directory in which you wish to store your
        images of the Scaled HOBJ ACME Images (Scl_ALL), and unscaled HOBJ ACME Images
        (Unscl_ALL) to see if subfolders for each of those divisions already exists.
        If they do, nothing will happen, if the subfolders do not already exist, they
        will be created.

        :param file: A directory in which you want to store your saved converted ACME images
        :return: nothing
    '''
    if os.path.exists(file + '_Scl_ALL'):
        pass
    else:
        os.mkdir(file + '_Scl_ALL')
        print("Scaled Directory Made")
    if os.path.exists(file + '_Unscl_ALL'):
        pass
    else:
        os.mkdir(file + '_Unscl_ALL')
        print("Unscaled Directory Made")


def directoryCheck(file):
    '''
        This function will check the directory in which you wish to store your
        images of the "FlameOnly", "Graphs", and "Images" to see if subfolders
        for each of those divisions already exists. If they do, nothing will
        happen, if the subfolders do not already exist, they will be created.

        :param file: A directory in which you want to store your saved images
        :return: nothing
    '''
    if os.path.exists(file + "_FLAMEONLY"):
        pass
    else:
        os.mkdir(file + "_FLAMEONLY")
        print("Flame only directory made")
    if os.path.exists(file + "_GRAPHS"):
        pass
    else:
        os.mkdir(file + "_GRAPHS")
        print("Graph directory made")
    if os.path.exists(file + "_IMAGES"):
        pass
    else:
        os.mkdir(file + "_IMAGES")
        print('Images directory made')


def read_hobj(image_filename):
    '''
    A helper function that takes in an HOBJ image and converts it into a 2D array of pixel values

    :param image_filename: The fileName of the HOBJ file
    :return: An array storing the pixel data of the HOBJ file
    '''
    start_time = time.time()
    # print('Getting HOBJ file >>> {:s}'.format(image_filename))
    with open(image_filename, 'rb') as f:
        ibuffer = f.read(84)

        npixels = struct.unpack('i', ibuffer[72:76][::-1])[0]
        rows = struct.unpack('h', ibuffer[80:82][::-1])[0] + 1
        cols = struct.unpack('h', ibuffer[82:84][::-1])[0] + 1

        # print('Image width/height is {:d}/{:d}.'.format(cols, rows))
        # print('# of pixels is {:d}.'.format(npixels))

        f.seek(84 + rows * 6 + 17)
        image_str = f.read(npixels * 2)
        bit_format = '<{:d}{:s}'.format(npixels, 'h')
        image_array = struct.unpack(bit_format, image_str)
        # print('Read time: {:.2f} seconds.'.format(time.time() - start_time))
    return np.array(image_array).reshape(rows, cols)


def analizeHOBJALL(imgFolder, saveDir, sclSaveDir):
    '''
    Function used to convert an entire folder full of HOBJ files to viewable .tiff
    formatted images

    :param imgFolder: Folder containing HOBJ files
    :param saveDir: Folder that will store color unscaled images
    :param sclSaveDir: Folder that will store color scaled images
    :return: nothing
    '''
    count = 0
    total = len(os.listdir(imgFolder))
    for fileName in os.listdir(imgFolder):
        analizeHOBJ(imgFolder, fileName, saveDir, sclSaveDir)
        count = count + 1
        PercentCheck(count, total)


def analizeHOBJ(imgFolder, fileName, saveDir, sclSaveDir):
    '''
    Single target conversion of HOBJ image to viewable .tiff formatted images

    :param imgFolder: Folder containing HOBJ images
    :param fileName: FileName of HOBJ Image
    :param saveDir: Location for storing non-color scaled image
    :param sclSaveDir: Location for storing color scaled image
    :return: none
    '''
    if fileName.endswith('HOBJ') or fileName.endswith('hobj'):
        imgPath = os.path.join(imgFolder, fileName)
        fileNameNoExt = fileName[0:(len(fileName) - 4)]

        fileNameTiff = fileNameNoExt + 'tiff'  # Start with converting the HOBJ files to TIFF Files
        imgArr = read_hobj(imgPath)

        im = Image.fromarray(imgArr)
        px = im.load()
        pixels = im.load()

        # Go through every pixel and read its value
        for i in range(im.size[0]):
            for j in range(im.size[1]):
                coordinate = i, j
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

        # cv2.imwrite(os.path.join(saveDir, fileNameTiff), imgArr)
        cv2.imwrite(os.path.join(sclSaveDir, fileNameTiff), sclImgArr)


def rotateNP90(img):
    '''
    Returns "img" rotated 90 degrees
    '''
    src = img
    src = np.rot90(src)
    return src


def rotateNP270(img):
    '''
    Returns "img" rotated 270 degrees
    '''
    src = img
    src = np.rot90(src)
    src = np.rot90(src)
    src = np.rot90(src)
    return src


def rotateImages(dir):
    '''
    Rotates all images in the "dir" directory 90 degrees
    '''
    for fileName in os.listdir(dir):
        if fileName.endswith("tiff"):
            src = Image.open(dir + "\\" + fileName)
            src = src.rotate(90, expand=True)
            src.save(dir + "\\" + fileName)


def avgCalc(im):
    '''

    :param im: Image file
    :return: the average pixel value for the whole image
    '''
    pxtotal = 0
    pX = im.load()
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            pxtotal = pX[i, j] + pxtotal
    avg = pxtotal / (im.size[1] * im.size[0])
    return avg


def flamePics(imgFolder, saveDir):
    '''
    Takes in all images in "imgFolder" and calculates whether or not they contain the flame
    by comparing its average pixel values to the baseline thresholds. It then saves those
    images to a seperate folder located at "saveDir"

    :param imgFolder: Folder containing all images
    :param saveDir: location for saving FlameOnly images
    :return: none
    '''
    count = 0
    total = len(os.listdir(imgFolder))
    print("Identifying Flame Frames in " + imgFolder)
    flameFound = False
    for fileName in sorted(os.listdir(imgFolder)):
        if fileName.endswith(".tiff") or fileName.endswith(
                'tif'):  # Used to stop error from occuring where the last filename after looping through all of them would say "error, no image found"
            try:  # this is done to catch any potentially corrupted files
                im1 = Image.open(imgFolder + "\\" + fileName)
                if 'ACME' in fileName:
                    imOrig = im1
                    im1 = ImageOps.grayscale(im1)
                imgFound = True

            except:
                print('image could not be opened')
                imgFound = False
            if flameFound == False and imgFound == True:  # will ensure that after the flame ignition is found all
                # proceeding images will be saved but wont bother saving all the images
                avg = avgCalc(im1)
                if 'ACME' in fileName and (avg < 10):
                    imOrig.save(saveDir + "\\" + fileName)
                    print(fileName, " was identified as flame start")
                    flameFound = True
                    total = total - count
                    count = 0
                if (avg > 60) and fileName.endswith('.tiff'):
                    im1.save(saveDir + "\\" + fileName)
                    print(fileName, " was identified as flame start")
                    flameFound = True
                    total = total - count
                    count = 0
                if (avg > 950 and fileName.endswith('.tif')):
                    im1.save(saveDir + "\\" + fileName)
                    print(fileName, " was identified as flame start")
                    flameFound = True
                    total = total - count
                    count = 0
                if (
                        count == 1 or count == 100 or count == 200 or count == 300 or count == 400 or count == 500 or count == 600 or count == 700):
                    print(count, "avg is ", avg)
                count = count + 1
            else:
                avg = avgCalc(im1)
                if 'ACME' in fileName:
                    if imgFound == True:
                        imOrig.save(saveDir + "\\" + fileName)
                else:
                    if (avg < 865) and fileName.endswith('.tif'):
                        imgFound = False
                    if (avg < 40) and fileName.endswith('.tiff'):
                        imgFound = False
                    if imgFound == True:
                        im1.save(saveDir + "\\" + fileName)
                count = count + 1
                PercentCheck(count, total)


def combineRTheta(radius, theta):
    '''
    Combines two arrays into one array of order pairs (radius, theta).
    Returns their result

    '''
    r = np.array(radius)
    t = np.array(theta)
    RTheta = np.array((r, t))
    return RTheta


def makeDataFrame(Data1, RTheta, fileName, saveDir):
    '''
    Creates a pickle dataframe of Data1, RTheta, and fileName, saving it under the "saveDir" directory
    :param Data1:
    :param RTheta:
    :param fileName:
    :param saveDir:
    :return: A pickle dataframe containing formatted data from first 3 params
    '''
    d = {'Image Name': fileName, 'Polar Image': Data1, 'R,Theta': RTheta}
    df = pd.DataFrame(data=d)  # {key:pd.Series(value) for key, value in d.items()}
    df.to_pickle(saveDir)
    return df


def darkFrameCalc(FullimgFolder):
    '''
    Calculates and returns the average of the darkframes from the tiff images. This is used to return
    an array containing the pixel data of the background noise (hot pixels) from the camera which can
    then be used to subtract from Flame images later.

    :param FullimgFolder: Folder containing all images
    :return: the image array containing the average values of all darkframes (only background noise left)
    '''
    print(FullimgFolder + "\\" + os.listdir(FullimgFolder)[0])
    x = 0  # This is established to ensure that dark frames are
    # calculated with the image file and not an extraneous text file
    while os.listdir(FullimgFolder)[x].endswith(".tiff") == False:
        x = x + 1
    imgSub = cv2.imread(FullimgFolder + "\\" + os.listdir(FullimgFolder)[x], cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if imgSub.shape == (1392, 1040):
        imgSub = rotateNP90(imgSub)
    totalDone = 1
    for i in range(34):
        try:
            imgSub = np.add(imgSub, cv2.imread(FullimgFolder + "\\" + os.listdir(FullimgFolder)[x + i + 1],
                                               cv2.IMREAD_GRAYSCALE).astype(np.float32))
            totalDone = totalDone + 1
        except:
            pass
    imgSub = np.divide(imgSub, totalDone)
    imgSub = imgSub.astype(np.float32)
    return imgSub


def darkFrameCalcLLUV(FullimgFolder):
    '''
    Same as "darkFrameCalc" but is compatible with LLUV images
    :param FullimgFolder: folder containing LLUV images
    :return:
    '''
    print(FullimgFolder + "\\" + os.listdir(FullimgFolder)[0])
    x = 0  # This is established to ensure that dark frames are
    # calculated with the image file and not an extraneous text file
    while os.listdir(FullimgFolder)[x].endswith(".tif") == False:
        x = x + 1
    imgSub = cv2.imread(FullimgFolder + "\\" + os.listdir(FullimgFolder)[x], cv2.IMREAD_GRAYSCALE).astype(np.float32)
    totalDone = 1
    for i in range(34):
        try:
            imgSub = np.add(imgSub, cv2.imread(FullimgFolder + "\\" + os.listdir(FullimgFolder)[x + i + 1],
                                               cv2.IMREAD_GRAYSCALE).astype(np.float32))
            totalDone = totalDone + 1
        except:
            pass
    imgSub = np.divide(imgSub, totalDone)
    print(totalDone)
    imgSub = imgSub.astype(np.float32)
    return imgSub


def skImageSubtract(imgSub, fileName):
    '''
    Returns the difference between fileName and imgSub (fileName - imgSub)
    :param imgSub: The image you want to subtract from fileName
    :param fileName: The base image you want to subtract from
    :return: Returns the difference between fileName and imgSub (fileName - imgSub)
    '''
    img = fileName
    if img.shape[0] == 1392:
        img = rotateNP90(img)
    temp = np.subtract(img, imgSub)
    temp = temp.astype(np.uint16)
    return temp


def skImageSubtractALL(fullImgFolder, flameOnlyFolder, saveDir):
    '''
    Mass application function for the skImageSubtract function across all images in "flameOnlyFolder"
    Also calculates the darkframe from fullImgFolder to be used in the subtraction
    :param fullImgFolder: Folder containing all images
    :param flameOnlyFolder: FlameOnly folder
    :param saveDir: Directory to save hotPixel removed images
    :return: none
    '''
    count = 0
    imgSub = darkFrameCalc(fullImgFolder)
    for fileName in sorted(os.listdir(flameOnlyFolder)):
        if fileName.endswith(".tiff"):
            skImageSubtract(imgSub, flameOnlyFolder + "\\" + fileName, saveDir)
            count = count + 1
            if count == 1 or count == 20 or count == 40 or count == 50 or count == len(os.listdir(fileName)) - 1:
                print(fileName + " had hotpixels subtracted")


def avgCalcSKI(im, width):  # takes in an image 'im' (read through SKImage) as well as a width which defines the width of each box used to find intensity
    valAvg = []
    length = im.shape[0]
    # Here the amount of boxes that will be used for any given image is calculated and rounded to the lowest amount to ensure no out of bounds errors will occur
    its = math.floor(im.shape[1] / width)
    # The function then iterates through the rest of the image, in boxes of range "width" and will repeat the above process for every box
    for iter in range(its):
        val = 0
        for x in range(width):  # loop through a width of "width" pixels
            x = x + (width * iter)
            for y in range(length):
                val = val + im[y, x]
        # round up the average to the next highest integer and then store that value in a list
        # average out each value with respect to the number of total pixels added
        val = val / (im.shape[0] * width)
        # then adds said value to a list of avgs
        valAvg.append(val)
    return valAvg  # will return a list of values that can be accessed through "avgCalc()[0]" for red, "avgCalc()[1]" for green, and "avgCalc()[2]" for blue


def cart2polar(im, center):
    '''
    Converts "im" to polar format based around "center"
    '''
    warped = warp_polar(im, center)
    return warped


def maxLoc(image):
    '''

    :param image: Image
    :return: coordinates of the pixel with highest value
    '''
    max = image.max()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] == max:
                return i, j


def removeHP(image, numPix):
    '''
    Removes "numPix" number of hotPixels from "image
    '''
    for i in range(numPix):
        image[maxLoc(image)] = 0


def peakIdentifyALL(imgFolder, GraphdirSave):
    '''
    Applies peakIdentify function to all images in imgFolder
    '''
    for fileName in os.listdir(imgFolder):
        peakIdentify(imgFolder + "\\" + fileName, GraphdirSave)


def dist(point, center):
    '''

    :param point: Coordinate in image array
    :param center: center point
    :return: the distance between the two points
    '''
    x = point[0] - center[0]
    y = point[1] - center[1]
    z = (x ** 2 + y ** 2)

    dist = math.sqrt(z)
    return dist


def thetaCalc(point, center):
    '''

    :param point: Coordinate in image array
    :param center: center point
    :return: the angle between the line made by the point-center line and the x-axis
    '''
    x = point[0] - center[0]
    y = point[1] - center[1]
    theta = math.atan(y / x)
    return theta


def peakIdentify(fileName, GraphdirSave, imgName, windowLength, polyDegree, maxMultiple):
    '''
    Identifies and returns a list of x and y values of the locations of all spikes in pixel values.
    Used to determine location of flame front coordinates.
    :param fileName: fileName of image
    :param GraphdirSave: Directory to save the graph made
    :param imgName: Name of image to save alongside the graph
    :param windowLength: Range at which the find_peaks function operates under
    :param polyDegree: the highest order polynomial used to create the line of bestfit for the created graph
    :param maxMultiple: the mulitplier used with the average pixel value of the image to use as the threshold for
    identifying peaks
    :return: list of x vals, list of y vals
    '''
    if fileName.dtype != 'str' or fileName.endswith(".tiff") or fileName.endswith('.tif'):
        peaks_outer = []
        if fileName.dtype != 'str':
            dst = fileName
        else:
            dst = io.imread(fileName)
        avgmax = max(avgCalcSKI(dst, 20))  # Calculate the highest avg pixel value using above function
        pixmax = dst.max()

        # print(pixmax,avgmax)  just needed for testing purposes

        for i in range(dst.shape[0]):  # Loops through every row (or y value)
            rowimg = dst[i, :]  # pulls one row from dst to form a 1-d array
            smoothdata = savgol_filter(rowimg, windowLength,
                                       polyDegree)  # uses savgol filter to smooth data to minimize false positive peaks
            peaks = find_peaks(smoothdata, height=avgmax * maxMultiple)  # finds the peaks in smoothdata
            height = peaks[1]['peak_heights']  # pulls the heights from the peaks function into a usable array
            peak_pos = peaks[0]  # pulls the peak position from peaks array

            if len(peaks[0]) != 0:  # ensures a value is only added to the list when a peak is actually identified
                row = [i, peak_pos[-1]]
                peaks_outer.append(row)
        peaks_outer = np.array(peaks_outer)
        # print(peaks_outer.shape)
        # print(peaks_outer)
        xvals = []
        yvals = []

        for i in range(len(peaks_outer)):
            yvals.append(peaks_outer[i][0])
            xvals.append(peaks_outer[i][1])

        plt.xlim(0, 500)
        plt.ylim(360, 0)
        matplotlib.use('Agg')  # using Agg ensures that only the data of each matplot is stored instead of the entire gui
        # this frees up an enourmous amount of ram and stops the error from too many matplots being open
        plt.scatter(xvals, yvals, s=3)
        plt.gca().invert_yaxis()  # ensures values go from 0 at bottom to 360 degrees at top
        if fileName.dtype == 'str':
            plt.title(fileName)
            plt.savefig(GraphdirSave + "\\" + fileName)
            # print(fileName, "  saved") only needed for testing purposes
        else:
            plt.title(imgName)
            plt.savefig(GraphdirSave + "\\" + imgName)
            # print(imgName, "  saved") just needed for testing purposes

        plt.clf()
        plt.close('all')  # extra insurance that matplot won't take up too much ram
        return xvals, yvals


def flamePlot(flameImage, pointVals, dir, fileName):
    '''Plots and saves graph of flameImage under the title fileName in the "dir" directory'''
    fig = plt.figure(dpi=600)
    plt.imshow(
        flameImage,
        cmap='gray',
        origin='lower'
    )
    plt.scatter(
        pointVals[0],
        pointVals[1],
        s=0.75,
        marker='.'
    )
    plt.title(fileName)
    plt.xlabel('Radius(px)')
    plt.ylabel('Angle' + '(' + u'\N{DEGREE SIGN}' + ')')
    fig.savefig(dir + "\\" + fileName[:-4] + 'matplotlib_dpi600.png')
    plt.close()


def makeVideo(GraphDir, shape, Type):
    '''
    Converts a directory of images into a .mp4 video
    :param GraphDir: Directory of graphs
    :param shape: dimensions of files
    :param Type: Type of file, ACME, LLLUV, INTNS
    :return: none
    '''
    print("making video")
    imgs = sorted(glob.glob(GraphDir + '\*.png'))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    video = cv2.VideoWriter(GraphDir + '\\' + Type + '.mp4', fourcc, 5, shape)

    for i in imgs:
        video.write(cv2.imread(i))

    cv2.destroyAllWindows()
    video.release()


def isFull(Folder):
    '''
    Tests whether "folder" has already been filled with images in a previous run to avoid
    re-analyzing folders

    :param Folder: Folder containing images
    :return: True or false (full or not)
    '''
    initial_count = 0
    for path in pathlib.Path(Folder).iterdir():
        if path.is_file():
            initial_count += 1
    if initial_count >= 5:
        return True
    else:
        return False


def PercentCheck(count, total):
    '''

    :param count: Number of completed files
    :param total: Total number of files
    :return: % completed
    '''
    if count == math.floor(total / 20):
        print('completed 5%   ', count, '/', total)
    if count == math.floor(total / 8):
        print('completed 12.5%   ', count, '/', total)
    if count == math.floor(total / 4):
        print('completed 25%   ', count, '/', total)
    if count == math.floor(total * 3 / 8):
        print('completed 37.5%   ', count, '/', total)
    if count == math.floor(total / 2):
        print('completed 50%   ', count, '/', total)
    if count == math.floor(total * 5 / 8):
        print('completed 62.5%   ', count, '/', total)
    if count == math.floor(total * 3 / 4):
        print('completed 75%   ', count, '/', total)
    if count == math.floor(total * 7 / 8):
        print('completed 87.5%   ', count, '/', total)


"""

Below is the cumulative code that will take a folder full of LLL-UV, ACME, or INTNS images and develop:
1) A folder full of all identified images holding a flame front
2) A folder full of all graphs with points on a polar coordinate of the flame front
3) A folder full of all polar converted images with the identified flame front points overlayed onto it
4) A video file comprised of #3
5) A pandas dataframe comprising of [Images, radii and thetas, filename]
0) Will preemptively scan the file location for graph, flameOnly, and images folders. Should they not be found
a new folder in the required names will be created

If at any point you need to redo an individual folder full of LLL-UV or INTNS images i.e. 20006B2_INTNS_ALL,
then you must empty all related folders (the 20006B2_INTNS_GRAPHS and 20006B2_INTNS_IMAGES)
You can empty the flame only folder if you wish to recalculate the flame images as well

Parameters for Different Image Types [Sigma, Truncate, Center]
- INTNS : 7,3,(720,383)
- LLL-UV : 1,5,(259,268)
- ACME_Scl_ALL : 9,3,(532,758)


"""


def fullAnalyze(MainDir, imgType, Sigma, Truncate, Center):  # sigma and truncate refer to the gaussian blur
    for file in glob.iglob(MainDir + '/**/*' + imgType, recursive=True):
        if 'Fiber' in file:
            pass
        else:
            print("Currently analyzing " + file)
            if imgType == 'INTNS_ALL':
                fname = file[:-4]
            else:
                fname = file
            directoryCheck(fname)
            # make lists to store dataframe data
            PandaSave = fname + "_PANDAS_dataframe.pkl"
            rThetas = []
            fileNames = []
            Images = []

            graphdir = fname + "_GRAPHS"
            flameOnlydir = fname + "_FLAMEONLY"
            imgdir = fname + "_IMAGES"
            print("Searching for Flame Pics")
            if isFull(flameOnlydir) == False:
                flamePics(file,
                          flameOnlydir)  # There is a catch inside this function if file is LLL-UV (name is deceiving)
            print("Found Flame Pics")

            if isFull(imgdir) == False:
                if 'LLL-UV' in imgType:
                    print("Calculating Dark frames")
                    sub = darkFrameCalcLLUV(file)
                    print("Dark Frames found")
                if 'INTNS' in imgType:
                    print("Calculating Dark frames")
                    sub = darkFrameCalc(file)
                    print("Dark Frames found")

                Size = len(os.listdir(flameOnlydir))
                print(Size, ' total images')
                imgsDone = 0
                for fileName in sorted(os.listdir(flameOnlydir)):  # Change this back to flameOnlydir
                    # print(fileName) just needed for testing purposes
                    if fileName.endswith('.tiff') or fileName.endswith('.tif'):

                        if 'ACME' in fileName:
                            img = io.imread(flameOnlydir + "\\" + fileName)[:, :, 2]

                        else:
                            img = io.imread(flameOnlydir + "\\" + fileName)
                            img = skImageSubtract(sub, img)
                        if 'LLL-UV' in imgType:
                            img = median(img, disk(5))
                        # if imgType == 'INTNS':
                        img = gaussian(img, sigma=Sigma,
                                       truncate=Truncate)  # Values for INTNS (sigma = 7, truncate = 3) LLUV = (9,3)
                        if 'INTNS' in imgType:
                            img = rotateNP90(img)  # comment out for LLUV

                        img = cart2polar(img, Center)
                        img = np.flipud(img)  # Comment
                        img3 = img
                        if 'INTNS' in imgType:
                            fFront = peakIdentify(img, graphdir, fileName, 21, 9, 1.2)
                        if 'LLL-UV' in imgType:
                            fFront = peakIdentify(img, graphdir, fileName, 19, 7, 1.1)
                        if 'ACME' in imgType:
                            fFront = peakIdentify(img, graphdir, fileName, 19, 7, 1.03)
                        rTheta = combineRTheta(fFront[0], fFront[1])
                        rThetas.append(rTheta)
                        fileNames.append(fileName)
                        Images.append(img3)
                        flamePlot(img, fFront, imgdir, fileName)
                    imgsDone = imgsDone + 1
                    PercentCheck(imgsDone, Size)
                makeDataFrame(Images, rThetas, fileNames, PandaSave)
                makeVideo(imgdir, (3840, 2880), imgType)

            else:
                print("Image folder already full, skipping folder")


center = (0, 0)
if __name__ == '__main__':
    for file in glob.iglob(r'E:\.shortcut-targets-by-id\12r7NVnUc9cMtuw-IqvGIc2uNipmfyODe\Cases\20029/**/*LLL-UV',
                           recursive=True):
        print("Currently analyzing " + file)
        directoryCheck(file)

        # make lists to store dataframe data
        PandaSave = file[:-4] + "_PANDAS_dataframe.pkl"
        rThetas = []
        fileNames = []
        Images = []

        graphdir = file[:-4] + "_GRAPHS"
        flameOnlydir = file[:-4] + "_FLAMEONLY"
        imgdir = file[:-4] + "_IMAGES"
        print("Searching for Flame Pics")
        flamePics(file, flameOnlydir)  # There is a catch inside this function if file is LLL-UV (name is deceiving)
        print("Found Flame Pics")
        graphs1 = []
        graphs2 = []
        print("Calculating Dark frames")
        sub = darkFrameCalcLLUV(file)
        print("Dark Frames found")

        for fileName in os.listdir(flameOnlydir):  # Change this back to flameOnlydir
            print(fileName)
            if fileName.endswith('.tiff') or fileName.endswith('.tif'):
                img = io.imread(flameOnlydir + "\\" + fileName)

                img = skImageSubtract(sub, img)
                img = gaussian(img, sigma=7, truncate=3)  # Values for INTNS (sigma = 7, truncate = 3) LLUV = (9,3)
                img = rotateNP90(img)  # comment out for LLUV
                img2 = img  # just used for testing purposes
                img = cart2polar(img, center)
                img = np.flipud(img)  # Comment
                img3 = img
                # removeHP(img,20) #supposed to remove x number of highest valued pixels
                fFront = peakIdentify(img, graphdir, fileName)
                # fig = graphOverlay(img,fFront,fileName)
                rTheta = combineRTheta(fFront[0], fFront[1])
                rThetas.append(rTheta)
                fileNames.append(fileName)
                Images.append(img3)
                flamePlot(img, fFront, imgdir, fileName)

        makeDataFrame(Images, rThetas, fileNames, PandaSave)
        makeVideo(graphdir)
