import cv2
import numpy as np
import os

def saveImage(image, fileName):
    outDir = 'output'
    try:
        os.mkdir(outDir)
    except:
        pass
    cv2.imwrite(os.path.join(outDir, fileName), image)

def macroblockDivide(image, size):
    # shape of input image
    row, col = image.shape[:2]

    # shape of output mbs
    rowMB, colMB = row // size, col // size

    # initialize macro blocks
    mb = np.zeros(shape=(rowMB, colMB, size, size, 3))

    # divide into macroblocks
    for i in range(rowMB):
        for j in range(colMB):
            mb[i, j] = image[i*size:(i+1)*size, j*size:(j+1)*size]

    return mb

def calSAD(a, b):
    return np.sum(np.abs(a - b))

def fullSearch(tar, ref, rowIndex, colIndex, searchRange):
    best = None
    minSAD = np.inf
    for i in range(-searchRange, searchRange + 1):
        for j in range(-searchRange, searchRange + 1):
            if rowIndex + i >= 0 and rowIndex + i < ref.shape[0] and colIndex + j >= 0 and colIndex + j < ref.shape[1]:
                nowSAD = calSAD(ref[rowIndex + i, colIndex + j], tar[rowIndex, colIndex])
                if nowSAD < minSAD:
                    minSAD = nowSAD
                    best = (i, j)
    return np.array(best)

def logSearch(tar, ref, rowIndex, colIndex, searchRange):
    # step 1. initialize
    center = (0, 0)
    best = None
    minSAD = np.inf
    step = searchRange // 2
    M = []
    updateM = True

    while step >= 1:
        # step 2. M(n)
        if updateM:
            M = [(0, 0), (step, 0), (0, step), (-step, 0), (0, -step)]
            updateM = False

        # step 3. find minimum
        for (i, j) in M:
            if 0 <= rowIndex + center[0] + i < ref.shape[0] and 0 <= colIndex + center[1] + j < ref.shape[1]:
                nowSAD = calSAD(ref[rowIndex + center[0] + i, colIndex + center[1] + j], tar[rowIndex, colIndex])
                if nowSAD < minSAD:
                    minSAD = nowSAD
                    best = (i, j)

        # step 4. q <- q + i, l <- l + j, M <- M - (-i, -j), go to step 3
        if best[0] != 0 or best[1] != 0:
            print(best[0], best[1])
            center = (center[0] + best[0], center[1] + best[1])
            M = [(m[0] + best[0], m[1] + best[1]) for m in M]
            continue

        # step 5. n <- n / 2
        step //= 2
        updateM = True
        print('---------------- step size: ', step)

    # step 6. find minimum in N(1)
    for i in [-step, 0, step]:
        for j in [-step, 0, step]:
            if 0 <= rowIndex + center[0] + i < ref.shape[0] and 0 <= colIndex + center[1] + j < ref.shape[1]:
                nowSAD = calSAD(ref[rowIndex + center[0] + i, colIndex + center[1] + j], tar[rowIndex, colIndex])
                if nowSAD < minSAD:
                    minSAD = nowSAD
                    best = (i, j)
    center = (center[0] + best[0], center[1] + best[1])

    return np.array(center)

def calMotionVector(ref, tar, searchRange, searchMethod):
    # shape of macroblocks
    row, col = ref.shape[:2]

    # initialize motion vector
    mv = np.zeros(shape=(row, col, 2))

    # calculate each block
    for i in range(row):
        for j in range(col):
            mv[i, j] = searchMethod(tar, ref, i, j, searchRange)

    return mv

def predict(tar, mv, size):
    imagePdt = np.zeros_like(tar)
    for i in range(tar.shape[0] // size):
        for j in range(tar.shape[1] // size):
            m, n = mv[i, j]
            imagePdt[int(i*size):int((i+1)*size), int(j*size):int((j+1)*size)] = tar[int((i+m)*size):int((i+m+1)*size), int((j+n)*size):int((j+n+1)*size)]

    return imagePdt

def PSNR(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    psnr = 10 * np.log10((255 ** 2) / mse)

    return psnr

if __name__ == '__main__':
    # get input
    image40 = cv2.imread('img/40.jpg')
    image42 = cv2.imread('img/42.jpg')
    image51 = cv2.imread('img/51.jpg')
    # convert BGR to RGB
    image40 = image40[:,:,::-1]
    image42 = image42[:,:,::-1]
    image51 = image51[:,:,::-1]

    # part 1
    # d = {'full': fullSearch, '2d': logSearch}
    d = {'2d': logSearch}
    for blockSize in [8, 16]:
        mbRef = macroblockDivide(image40, blockSize)
        mbTar = macroblockDivide(image42, blockSize)
        for searchMethod in d.keys():
            for searchRange in [8, 16]:
                mv = calMotionVector(mbRef, mbTar, searchRange, d[searchMethod])
                imagePdt = predict(image40, mv, blockSize)
                saveImage(imagePdt[:,:,::-1], f'{searchMethod}_predicted_r{searchRange}_b{blockSize}.jpg')
                print('{}_r{}_b{} -- SAD: {}, PSNR: {:2.4f}'.format(searchMethod, searchRange, blockSize, calSAD(image42, imagePdt), PSNR(image42, imagePdt)))

    # part 2
    mbRef = macroblockDivide(image40, 8)
    mbTar = macroblockDivide(image51, 8)
    mv = calMotionVector(mbRef, mbTar, 8, logSearch)
    imagePdt = predict(image40, mv, 8)
    saveImage(imagePdt[:,:,::-1], f'40to51.jpg')
    print('40 to 51 -- SAD: {}, PSNR: {:2.4f}'.format(calSAD(image51, imagePdt), PSNR(image51, imagePdt)))
