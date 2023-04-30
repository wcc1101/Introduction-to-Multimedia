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
    best = None
    minSAD = np.inf
    step = searchRange // 2
    while step > 0:
        for i in range(-searchRange, searchRange + 1, step):
            for j in range(-searchRange, searchRange + 1, step):
                if rowIndex + i >= 0 and rowIndex + i < ref.shape[0] and colIndex + j >= 0 and colIndex + j < ref.shape[1]:
                    nowSAD = calSAD(ref[rowIndex + i, colIndex + j], tar[rowIndex, colIndex])
                    if nowSAD < minSAD:
                        minSAD = nowSAD
                        best = (i, j)
        searchRange = step
        step //= 2
    return np.array(best)

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
    imageRef = cv2.imread('img/40.jpg')
    imageTar = cv2.imread('img/42.jpg')
    # convert BGR to RGB
    imageRef = imageRef[:,:,::-1]
    imageTar = imageTar[:,:,::-1]

    d = {'full': fullSearch, '2d': logSearch}

    for blockSize in [8, 16]:
        mbRef = macroblockDivide(imageRef, blockSize)
        mbTar = macroblockDivide(imageTar, blockSize)
        for searchMethod in d.keys():
            for searchRange in [8, 16]:
                mv = calMotionVector(mbRef, mbTar, searchRange, d[searchMethod])
                imagePdt = predict(imageRef, mv, blockSize)
                saveImage(imagePdt[:,:,::-1], f'{searchMethod}_predicted_r{searchRange}_b{blockSize}.jpg')
                print('{}_r{}_b{} -- SAD: {}, PSNR: {:2.4f}'.format(searchMethod, searchRange, blockSize, calSAD(imageTar, imagePdt), PSNR(imageTar, imagePdt)))
