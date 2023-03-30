import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def bitReduce(image, table):
    newImage = np.zeros(shape=(image.shape[0], image.shape[1]), dtype='uint8')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            newImage[i][j] = table[tuple(image[i][j])]
    return newImage

def constructTable(image):
    colorMap = set([tuple(j) for i in image for j in i])
    table = {}
    code = 0
    for color in colorMap:
        table[color] = code
        table[code] = color
        code += 1
    return table

def saveImage(image, fileName):
    outDir = 'out'
    try:
        os.mkdir(outDir)
    except:
        pass
    cv2.imwrite(os.path.join(outDir, fileName), image)

def solve(image, flattenImage, depth):
    # end of recursion, quantize the image
    if depth == 0:
        bAverage = np.mean(flattenImage[:,2])
        gAverage = np.mean(flattenImage[:,3])
        rAverage = np.mean(flattenImage[:,4])
        
        for pixel in flattenImage:
            image[pixel[0]][pixel[1]] = [bAverage, gAverage, rAverage]
        return
    
    # calculate BGR ranges
    bRange = np.max(flattenImage[:,2]) - np.min(flattenImage[:,2])
    gRange = np.max(flattenImage[:,3]) - np.min(flattenImage[:,3])
    rRange = np.max(flattenImage[:,4]) - np.min(flattenImage[:,4])

    maxRange = np.argmax([-1, -1, bRange, gRange, rRange])

    flattenImage = flattenImage[flattenImage[:, maxRange].argsort()]
    medianIndex = int((len(flattenImage) + 1) / 2)

    solve(image, flattenImage[0:medianIndex], depth - 1)
    solve(image, flattenImage[medianIndex:], depth - 1)

def medianCut(image, n):
    # flatten the image
    flattenImage = []
    for rowIndex, rows in enumerate(image):
        for colIndex, pixel in enumerate(rows):
            flattenImage.append([rowIndex, colIndex, pixel[0], pixel[1], pixel[2]])
    flattenImage = np.array(flattenImage)

    # run recursion
    solve(image, flattenImage, n)

    # save image
    saveImage(image, f'median_cut{str(n)}.jpg')

    return image

def calDistance(color1, color2):
    return np.sum(np.square(color1 - color2))

def findClosestColor(pixel, colorMap):
    closestColor = pixel
    minDistance = np.inf

    # try every color in map
    for color in colorMap:
        newDistance = calDistance(np.array(pixel), np.array(color))
        if newDistance < minDistance:
            minDistance = newDistance
            closestColor = color

    return closestColor

def addError(pixel, error):
    # for BGR seperately
    for i in range(3):
        # clamp to 0-255
        pixel[i] = max(0, min(pixel[i] + error[i], 255))

def errorDiffusionDithering(image, colorMap, n):
    # define height and width for the image
    h, w = image.shape[:2]

    # tranform to float
    image = np.array(image, dtype=float)

    # dither every pixels
    for rowIndex in range(h):
        for colIndex in range(w):
            # find new color with shortest distance
            oldColor = image[rowIndex][colIndex].copy()
            newColor = findClosestColor(image[rowIndex][colIndex], colorMap)

            # calculate error
            image[rowIndex][colIndex] = newColor
            error = oldColor - newColor

            # apply error
            if colIndex + 1 < w:
                addError(image[rowIndex][colIndex + 1], (7 / 16) * error)
            if rowIndex + 1 < h:
                addError(image[rowIndex + 1][colIndex], (5 / 16) * error)
                if colIndex - 1 >= 0:
                    addError(image[rowIndex + 1][colIndex - 1], (3 / 16) * error)
                if colIndex + 1 < w:
                    addError(image[rowIndex + 1][colIndex + 1], (1 / 16) * error)

    # tranform back
    image = np.array(image, dtype='uint8')

    # save image
    saveImage(image, f'error_diffusion_dithering_{str(n)}.jpg')

    return image

def nearestNeighborInterpolation(image):
    # define height and width for the image
    h, w = image.shape[:2]

    # initialize new image
    image_NN = np.zeros(shape = (4 * h, 4 * w, 3), dtype='uint8')

    # calculate each pixel in new image
    for i in range(4 * h):
        for j in range(4 * w):
            image_NN[i][j] = image[int(i / 4)][int(j / 4)]

    # save image
    saveImage(image_NN, 'bee_near.jpg')

    return image_NN

def biLinearInterpolation(image):
    # define height and width for the image
    h, w = image.shape[:2]

    # initialize new image
    image_BL = np.zeros(shape=(4 * h, 4 * h, 3))

    # calculate each pixel in new image
    for i in range(4 * h):
        for j in range(4 * w):
            # find position in original image
            x = (i + 1) / 4 - 1
            y = (j + 1) / 4 - 1

            # find referencing point
            xIndex = int(x)
            yIndex = int(y)

            # calculate bias
            u = x - xIndex
            v = y - yIndex

            # # calculate by value * area
            image_BL[i][j] += (1 - u) * (1 - v) * image[xIndex][yIndex]
            if xIndex + 1 < h:
                image_BL[i][j] += u * (1 - v) * image[xIndex + 1][yIndex]
            if yIndex + 1 < w:
                image_BL[i][j] += (1 - u) * v * image[xIndex][yIndex + 1]
                if xIndex + 1 < h:
                    image_BL[i][j] += u * v * image[xIndex + 1][yIndex + 1]

            # fix value
            for c in range(3):
                if image_BL[i][j][c] < 0:
                    image_BL[i][j][c] = 0

    image_BL = image_BL.astype('uint8')

    # save image
    saveImage(image_BL, 'bee_linear.jpg')

    return image_BL

def saveHist(data, fileName):
    outDir = 'out'
    try:
        os.mkdir(outDir)
    except:
        pass
    plt.clf()
    plt.hist(data)
    plt.xlim(0, 256)
    plt.title(fileName)
    plt.savefig(os.path.join(outDir, fileName))

def RGB2YIQ(image):
    # define height and width for the image
    h, w = image.shape[:2]

    # initialize new image
    image_YIQ = np.zeros(shape=(h, w, 3), dtype='float64')

    # calculate each channel
    for i in range(h):
        for j in range(w):
            image_YIQ[i][j][0] += 0.299 * image[i][j][0] + 0.587 * image[i][j][1] + 0.114 * image[i][j][2]
            image_YIQ[i][j][1] += 0.596 * image[i][j][0] - 0.275 * image[i][j][1] - 0.321 * image[i][j][2]
            image_YIQ[i][j][2] += 0.212 * image[i][j][0] - 0.523 * image[i][j][1] + 0.311 * image[i][j][2]

    return image_YIQ

def YIQ2RGB(image):
    # define height and width for the image
    h, w = image.shape[:2]

    # initialize new image
    image_RGB = np.zeros(shape=(h, w, 3), dtype='float64')

    # calculate each channel
    for i in range(h):
        for j in range(w):
            image_RGB[i][j][0] += image[i][j][0] + 0.956 * image[i][j][1] + 0.621 * image[i][j][2]
            image_RGB[i][j][1] += image[i][j][0] - 0.272 * image[i][j][1] - 0.647 * image[i][j][2]
            image_RGB[i][j][2] += image[i][j][0] - 1.106 * image[i][j][1] + 1.703 * image[i][j][2]

    image_RGB = image_RGB.astype('uint8')

    return image_RGB

def gammaCorrect(image, gamma):
    # convert image to YIQ
    image_YIQ = RGB2YIQ(image.copy())
    y_channel = image_YIQ[:,:,0].copy()

    # standarize to 0-1
    image_YIQ[:,:,0] /= 255.0

    # gamma tranformation
    image_YIQ[:,:,0] = np.power(image_YIQ[:,:,0], gamma)

    # de-standarize
    image_YIQ[:,:,0] *= 255
    y_channel_gamma = image_YIQ[:,:,0].copy()

    # convert image to RGB
    image_RGB_gamma = YIQ2RGB(image_YIQ).astype('uint8')

    return image_RGB_gamma, y_channel, y_channel_gamma