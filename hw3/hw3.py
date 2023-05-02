import cv2
import numpy as np
import os
import time

def saveImage(image, fileName):
    outDir = 'out'
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
    return np.sum(np.abs(a.astype('int32') - b.astype('int32')))

def fullSearch(tar, ref, rowIndex, colIndex, searchRange):
    best = None
    minSAD = np.inf
    for i in range(-searchRange, searchRange + 1):
        for j in range(-searchRange, searchRange + 1):
            if 0 <= rowIndex + i < ref.shape[0] and 0 <= colIndex + j < ref.shape[1]:
                nowSAD = calSAD(ref[rowIndex + i, colIndex + j], tar[rowIndex, colIndex])
                if nowSAD < minSAD:
                    minSAD = nowSAD
                    best = (i, j)
    return np.array(best)

def updateCenter(center, best, rowIndex, colIndex, imgShape):
    newCenter = (center[0] + best[0], center[1] + best[1])

    if rowIndex + newCenter[0] < 0:
        newCenter = (-rowIndex, newCenter[1])
    elif rowIndex + newCenter[0] >= imgShape[0]:
        newCenter = (imgShape[0] - 1 - rowIndex, newCenter[1])
    if colIndex + newCenter[1] < 0:
        newCenter = (newCenter[0], -colIndex)
    elif colIndex + newCenter[1] >= imgShape[1]:
        newCenter = (newCenter[0], imgShape[1] - 1 - colIndex)

    return newCenter

def logSearch(tar, ref, rowIndex, colIndex, searchRange):
    best = None
    minSAD = np.inf
    step = searchRange // 2
    center = (0, 0)

    while step >= 1:
        for i in [-step, 0, step]:
            for j in [-step, 0, step]:
                if i * j != 0 and (i != 0 or j != 0):
                    continue
                if 0 <= rowIndex + center[0] + i < ref.shape[0] and 0 <= colIndex + center[1] + j < ref.shape[1]:
                    nowSAD = calSAD(ref[rowIndex + center[0] + i, colIndex + center[1] + j], tar[rowIndex, colIndex])
                    if nowSAD < minSAD:
                        minSAD = nowSAD
                        best = (i, j)
        center = updateCenter(center, best, rowIndex, colIndex, ref.shape)
        best = (0, 0)
        step //= 2

    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if 0 <= rowIndex + center[0] + i < ref.shape[0] and 0 <= colIndex + center[1] + j < ref.shape[1]:
                nowSAD = calSAD(ref[rowIndex + center[0] + i, colIndex + center[1] + j], tar[rowIndex, colIndex])
                if nowSAD < minSAD:
                    minSAD = nowSAD
                    best = (i, j)
    center = updateCenter(center, best, rowIndex, colIndex, ref.shape)

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

def drawMotion(image, mv, size):
    # get arrows index
    arrows = np.zeros_like(mv)
    for i in range(mv.shape[0]):
        for j in range(mv.shape[1]):
            arrows[i,j] = (j*size+size/2+mv[i,j,0], i*size+size/2+mv[i,j,1])

    # draw arrows
    for i in range(arrows.shape[0]):
        for j in range(arrows.shape[1]):
            x1, y1 = j*size+size/2, i*size+size/2
            x2, y2 = arrows[i,j]
            angle = np.arctan2(y2-y1, x2-x1)
            end_x = int(x2 - 3*np.cos(angle))
            end_y = int(y2 - 3*np.sin(angle))
            cv2.arrowedLine(image, (int(x1), int(y1)), (end_x, end_y), (255, 0, 255), 1)

    return image

def residual(image, imagePdt):
    image = image.astype('int32')

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(3):
                image[i, j, c] = image[i, j, c] - imagePdt[i, j, c]
                if image[i, j, c] < 0:
                    image[i, j, c] = 0
    
    image = image.astype('uint8')

    return image

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
    d = {'full': fullSearch, '2d': logSearch}
    for searchMethod in d.keys():
        for blockSize in [8, 16]:
            mbRef = macroblockDivide(image40, blockSize)
            mbTar = macroblockDivide(image42, blockSize)
            for searchRange in [8, 16]:
                start = time.time()
                mv = calMotionVector(mbRef, mbTar, searchRange, d[searchMethod])
                end = time.time()
                imagePdt = predict(image40, mv, blockSize)
                imageMV = drawMotion(image40.copy(), mv, blockSize)
                imageRes = residual(image40.copy(), imagePdt)
                saveImage(imagePdt[:,:,::-1], f'{searchMethod}_predicted_r{searchRange}_b{blockSize}.jpg')
                saveImage(imageMV[:,:,::-1], f'{searchMethod}_motion_vector_r{searchRange}_b{blockSize}.jpg')
                saveImage(imageRes[:,:,::-1], f'{searchMethod}_residual_r{searchRange}_b{blockSize}.jpg')
                print('{}_r{:02d}_b{:02d} -- SAD: {}, PSNR: {:2.4f}'.format(searchMethod, searchRange, blockSize, calSAD(image42, imagePdt), PSNR(image42, imagePdt)))
                print('{}_r{:02d}_b{:02d} -- runtime: {:3.4f}s'.format(searchMethod, searchRange, blockSize, end - start))

    # part 2
    mbRef = macroblockDivide(image40, 8)
    mbTar = macroblockDivide(image51, 8)
    mv = calMotionVector(mbRef, mbTar, 8, fullSearch)
    imagePdt = predict(image40, mv, 8)
    print('Part2 -- SAD: {}, PSNR: {:2.4f}'.format(calSAD(image51, imagePdt), PSNR(image51, imagePdt)))
