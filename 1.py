import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

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
    # print(f'ranges: R: {rRange}, G: {gRange}, B: {bRange}')

    maxRange = np.argmax([-1, -1, bRange, gRange, rRange])
    # print(f'max range: {maxRange}')

    flattenImage = flattenImage[flattenImage[:, maxRange].argsort()]
    medianIndex = int((len(flattenImage) + 1) / 2)
    # print(f'median index: {medianIndex}')

    solve(image, flattenImage[0:medianIndex], depth - 1)
    solve(image, flattenImage[medianIndex:], depth - 1)

def medianCut(image, n):
    # flatten the image
    flattenImage = []
    for rowIndex, rows in enumerate(image):
        for colIndex, pixel in enumerate(rows):
            flattenImage.append([rowIndex, colIndex, pixel[0], pixel[1], pixel[2]])
    flattenImage = np.array(flattenImage)
    # print(flattenImage[0])

    # run recursion
    solve(image, flattenImage, n)

    # save image
    outDir = 'out'
    fileName = 'median_cut' + str(n) + '.jpg'
    try:
        os.mkdir(outDir)
    except:
        pass
    cv2.imwrite(os.path.join(outDir, fileName), image)

    return image

if __name__ == '__main__':

    # open input image
    imageFileName = 'img/Lenna.jpg'
    image = cv2.imread(imageFileName)

    ### A part ###
    # median cut
    image6 = medianCut(image.copy(), 6)
    image3 = medianCut(image.copy(), 3)

    # show image
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original')
    ax[1].imshow(cv2.cvtColor(image6, cv2.COLOR_BGR2RGB))
    ax[1].set_title('n = 6')
    ax[2].imshow(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB))
    ax[2].set_title('n = 3')
    plt.show()

    # calculate MSE
    print(f'MSE of quantization with n=6: {np.mean((image6 - image) ** 2)}')
    print(f'MSE of quantization with n=3: {np.mean((image3 - image) ** 2)}')

    ### B part ###
