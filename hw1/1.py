import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

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
        # clamp to 0-1
        pixel[i] = max(0, min(pixel[i] + error[i], 1))

def errorDiffusionDithering(image, colorMap, n):
    # define height and width for the image
    h, w = image.shape[:2]

    # tranform to float
    image = np.array(image, dtype=float) / 255

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
    image = np.array(image * 255, dtype=np.uint8)

    # save image
    saveImage(image, f'error_diffusion_dithering_{str(n)}.jpg')

    return image

### A part ###
def A(image):
    # median cut
    image3 = medianCut(image.copy(), 3)
    image6 = medianCut(image.copy(), 6)

    # calculate MSE
    print(f'MSE of quantization with n=3: {np.mean((image3 - image) ** 2)}')
    print(f'MSE of quantization with n=6: {np.mean((image6 - image) ** 2)}')

    return image3, image6

### B part ###
def B(image, colorMap3, colorMap6):
    # error diffusion dithering
    image3 = errorDiffusionDithering(image.copy(), colorMap3, 3)
    image6 = errorDiffusionDithering(image.copy(), colorMap6, 6)

    # calculate MSE
    print(f'MSE of error diffusion dithering with n=3: {np.mean((image3 - image) ** 2)}')
    print(f'MSE of error diffusion dithering with n=6: {np.mean((image6 - image) ** 2)}')

    return image3, image6

if __name__ == '__main__':
    # open input image
    imageFileName = 'img/Lenna.jpg'
    image = cv2.imread(imageFileName)

    # part A
    image3, image6 = A(image.copy())

    # generate color map
    colorMap3 = set([tuple(j / 255) for i in image3 for j in i])
    colorMap6 = set([tuple(j / 255) for i in image6 for j in i])

    # part B
    image3_d, image6_d = B(image.copy(), colorMap3, colorMap6)

    # plot
    f, ax = plt.subplots(2, 3, subplot_kw={'xticks': [], 'yticks': []})
    ax[0][0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0][0].set_title('Original')
    ax[0][1].imshow(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB))
    ax[0][1].set_title('Quantization (n=3)')
    ax[0][2].imshow(cv2.cvtColor(image3_d, cv2.COLOR_BGR2RGB))
    ax[0][2].set_title('Dithering (n=3)')
    ax[1][0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[1][0].set_title('Original')
    ax[1][1].imshow(cv2.cvtColor(image6, cv2.COLOR_BGR2RGB))
    ax[1][1].set_title('Quantization (n=6)')
    ax[1][2].imshow(cv2.cvtColor(image6_d, cv2.COLOR_BGR2RGB))
    ax[1][2].set_title('Dithering (n=6)')
    plt.show()
