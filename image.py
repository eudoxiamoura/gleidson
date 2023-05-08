import os
import time
import math

import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries

from medpy.filter.smoothing import anisotropic_diffusion

from matplotlib import cm

# nurbs
from geomdl import BSpline, NURBS
from geomdl import utilities
from geomdl import exchange

from geomdl.visualization import VisMPL

from geomdl import operations
# from geomdl.visualization import VisMPL
from geomdl.visualization import VisPlotly


# carregar imagens

def loadImage(pathes, channels):
    imgs = []
    for i in pathes:
        imgs.append(cv2.imread(i, channels))
    return imgs


def saveFile(name, array):
    f = open("Processamento/Result/Objetos/"+name+".cpt", "w")
    for i in array:
        f.write(str(i[0])+','+str(i[1])+','+str(i[2])+'\n')
    f.close()


def saveFileFullPts(path, name, array, w):
    f = open(path+name+"_ctrlpoints.cptw", "w")
    for i in range(len(array[0])):
        for j in range(len(array)):
            f.write(str(array[j][i][0])+','+str(array[j][i]
                    [1])+','+str(array[j][i][2])+','+str(w[j]))
            if j < len(array)-1:
                f.write(";")
        f.write("\n")
    f.close()


def loadFile(pathes):
    imgs = []
    start_time = time.time()
    for i in pathes:
        ini = time.time()
        file = open(i, 'r')
        txt = file.read()
        file.close()
        imgs.append([float(val) for val in txt.split()])
        print(str(pathes.index(i))+'/'+str(np.size(pathes))+'\t'+i +
              '\t'+str(time.time()-ini)+'\t'+str(time.time()-start_time))
    return imgs


def saveImagesDefinedPath(image, folder, names):
    pathes = []
    for i in range(len(image)):
        filename = names[i].split('/')[-1][:-4]
        pathes.append(folder+'/'+filename+'.jpg')
        cv2.imwrite(folder+'/'+filename+'.jpg', image[i])
    return pathes


def saveImages(image, folder, processo):
    pathes = []
    for i in range(len(image)):
        pathes.append(folder+'/'+processo+'_'+str(i)+'.jpg')
        cv2.imwrite(folder+'/'+processo+'_'+str(i)+'.jpg', image[i])
    return pathes


def saveImagesSided(image, folder, processo, onerow=True):
    pathes = []
    if onerow:
        newimg = np.concatenate(
            (image[2], image[1], image[0], image[3], image[4]), axis=1)
        cv2.imwrite(folder+'/'+processo+'_full.jpg', newimg)
        return pathes
    else:
        imgs = []
        for i in image:
            imgs.append(np.hstack((i[2], i[1], i[0], i[3], i[4])))
        newimg = imgs[0]
        for i in range(1, len(imgs), 1):
            newimg = np.vstack((newimg, imgs[i]))
        newimg = np.vstack(
            (newimg, np.zeros(imgs[0].shape), np.zeros(imgs[0].shape)))
        cv2.imwrite(folder+'/'+processo+'_full2.jpg', newimg)
        return pathes


def convertToImage(arr, size):
    array = np.array(arr)
    shape = array.shape
    for i in range(shape[0]):
        min = np.min(array[i])
        max = np.max(array[i])
        for j in range(shape[1]):
            array[i][j] = (int)(((array[i][j]-min)/(max-min))*255)
    array = np.reshape(array, (shape[0], size[0], size[1]))

    return array


def gray2Color(images):
    newimg = []

    rain = [[0, 0, 0], [15, 0, 15], [31, 0, 31], [47, 0, 47], [63, 0, 63], [79, 0, 79], [95, 0, 95], [111, 0, 111], [127, 0, 127], [143, 0, 143], [159, 0, 159], [175, 0, 175], [191, 0, 191], [207, 0, 207], [223, 0, 223], [239, 0, 239], [255, 0, 255], [239, 0, 250], [223, 0, 245], [207, 0, 240], [191, 0, 236], [175, 0, 231], [159, 0, 226], [143, 0, 222], [127, 0, 217], [111, 0, 212], [95, 0, 208], [79, 0, 203], [63, 0, 198], [47, 0, 194], [31, 0, 189], [15, 0, 184], [0, 0, 180], [0, 15, 184], [0, 31, 189], [0, 47, 194], [0, 63, 198], [0, 79, 203], [0, 95, 208], [0, 111, 212], [0, 127, 217], [0, 143, 222], [0, 159, 226], [0, 175, 231], [0, 191, 236], [0, 207, 240], [0, 223, 245], [0, 239, 250], [0, 255, 255], [0, 245, 239], [0, 236, 223], [0, 227, 207], [0, 218, 191], [0, 209, 175], [0, 200, 159], [0, 191, 143], [0, 182, 127], [0, 173, 111], [0, 164, 95], [0, 155, 79], [0, 146, 63], [
        0, 137, 47], [0, 128, 31], [0, 119, 15], [0, 110, 0], [15, 118, 0], [30, 127, 0], [45, 135, 0], [60, 144, 0], [75, 152, 0], [90, 161, 0], [105, 169, 0], [120, 178, 0], [135, 186, 0], [150, 195, 0], [165, 203, 0], [180, 212, 0], [195, 220, 0], [210, 229, 0], [225, 237, 0], [240, 246, 0], [255, 255, 0], [251, 240, 0], [248, 225, 0], [245, 210, 0], [242, 195, 0], [238, 180, 0], [235, 165, 0], [232, 150, 0], [229, 135, 0], [225, 120, 0], [222, 105, 0], [219, 90, 0], [216, 75, 0], [212, 60, 0], [209, 45, 0], [206, 30, 0], [203, 15, 0], [200, 0, 0], [202, 11, 11], [205, 23, 23], [207, 34, 34], [210, 46, 46], [212, 57, 57], [215, 69, 69], [217, 81, 81], [220, 92, 92], [222, 104, 104], [225, 115, 115], [227, 127, 127], [230, 139, 139], [232, 150, 150], [235, 162, 162], [237, 173, 173], [240, 185, 185], [242, 197, 197], [245, 208, 208], [247, 220, 220], [250, 231, 231], [252, 243, 243]]
    iron = [[0, 0, 0], [0, 0, 36], [0, 0, 51], [0, 0, 66], [0, 0, 81], [2, 0, 90], [4, 0, 99], [7, 0, 106], [11, 0, 115], [14, 0, 119], [20, 0, 123], [27, 0, 128], [33, 0, 133], [41, 0, 137], [48, 0, 140], [55, 0, 143], [61, 0, 146], [66, 0, 149], [72, 0, 150], [78, 0, 151], [84, 0, 152], [91, 0, 153], [97, 0, 155], [104, 0, 155], [110, 0, 156], [115, 0, 157], [122, 0, 157], [128, 0, 157], [134, 0, 157], [139, 0, 157], [146, 0, 156], [152, 0, 155], [157, 0, 155], [162, 0, 155], [167, 0, 154], [171, 0, 153], [175, 1, 152], [178, 1, 151], [182, 2, 149], [185, 4, 149], [188, 5, 147], [191, 6, 146], [193, 8, 144], [195, 11, 142], [198, 13, 139], [201, 17, 135], [203, 20, 132], [206, 23, 127], [208, 26, 121], [210, 29, 116], [212, 33, 111], [214, 37, 103], [217, 41, 97], [219, 46, 89], [221, 49, 78], [223, 53, 66], [224, 56, 54], [226, 60, 42], [228, 64, 30], [229, 68, 25], [231, 72, 20], [232, 76, 16], [
        234, 78, 12], [235, 82, 10], [236, 86, 8], [237, 90, 7], [238, 93, 5], [239, 96, 4], [240, 100, 3], [241, 103, 3], [241, 106, 2], [242, 109, 1], [243, 113, 1], [244, 116, 0], [244, 120, 0], [245, 125, 0], [246, 129, 0], [247, 133, 0], [248, 136, 0], [248, 139, 0], [249, 142, 0], [249, 145, 0], [250, 149, 0], [251, 154, 0], [252, 159, 0], [253, 163, 0], [253, 168, 0], [253, 172, 0], [254, 176, 0], [254, 179, 0], [254, 184, 0], [254, 187, 0], [254, 191, 0], [254, 195, 0], [254, 199, 0], [254, 202, 1], [254, 205, 2], [254, 208, 5], [254, 212, 9], [254, 216, 12], [255, 219, 15], [255, 221, 23], [255, 224, 32], [255, 227, 39], [255, 229, 50], [255, 232, 63], [255, 235, 75], [255, 238, 88], [255, 239, 102], [255, 241, 116], [255, 242, 134], [255, 244, 149], [255, 245, 164], [255, 247, 179], [255, 248, 192], [255, 249, 203], [255, 251, 216], [255, 253, 228], [255, 254, 239], [255, 255, 249]]
    thermal = [[0, 5, 0], [0, 10, 0], [0, 16, 0], [0, 23, 0], [0, 31, 0], [0, 39, 0], [0, 47, 0], [0, 56, 0], [0, 66, 0], [0, 75, 0], [0, 85, 0], [0, 96, 0], [0, 106, 0], [0, 117, 0], [0, 128, 0], [0, 138, 0], [0, 149, 0], [0, 159, 0], [0, 170, 0], [0, 180, 0], [0, 189, 0], [0, 199, 0], [0, 208, 0], [0, 216, 0], [0, 224, 0], [0, 232, 0], [0, 239, 0], [0, 245, 0], [0, 250, 0], [0, 255, 0], [5, 255, 0], [10, 255, 0], [16, 255, 0], [23, 255, 0], [31, 255, 0], [39, 255, 0], [47, 255, 0], [56, 255, 0], [66, 255, 0], [75, 255, 0], [85, 255, 0], [96, 255, 0], [106, 255, 0], [117, 255, 0], [128, 255, 0], [138, 255, 0], [149, 255, 0], [159, 255, 0], [170, 255, 0], [180, 255, 0], [189, 255, 0], [199, 255, 0], [208, 255, 0], [216, 255, 0], [224, 255, 0], [232, 255, 0], [239, 255, 0], [245, 255, 0], [250, 255, 0], [255, 255, 0], [255, 250, 0], [255, 244, 0], [255, 238, 0], [
        255, 230, 0], [255, 221, 0], [255, 212, 0], [255, 202, 0], [255, 192, 0], [255, 181, 0], [255, 170, 0], [255, 159, 0], [255, 147, 0], [255, 135, 0], [255, 123, 0], [255, 112, 0], [255, 100, 0], [255, 89, 0], [255, 78, 0], [255, 67, 0], [255, 57, 0], [255, 47, 0], [255, 38, 0], [255, 30, 0], [255, 22, 0], [255, 16, 0], [255, 10, 0], [255, 6, 0], [255, 3, 0], [255, 1, 0], [255, 0, 0], [255, 1, 5], [255, 3, 10], [255, 6, 16], [255, 10, 23], [255, 16, 31], [255, 22, 39], [255, 30, 47], [255, 38, 56], [255, 47, 66], [255, 57, 75], [255, 67, 85], [255, 78, 96], [255, 89, 106], [255, 100, 117], [255, 112, 128], [255, 123, 138], [255, 135, 149], [255, 147, 159], [255, 159, 170], [255, 170, 180], [255, 181, 189], [255, 192, 199], [255, 202, 208], [255, 212, 216], [255, 221, 224], [255, 230, 232], [255, 238, 239], [255, 244, 245], [255, 250, 250], [255, 255, 255]]

    for i in images:
        newimg.append(np.zeros([i.shape[0], i.shape[1], 3], dtype=np.uint8))

    for i in range(len(images)):
        for j in range(len(images[i])):
            for k in range(len(images[i][j])):

                t = images[i][j][k]
                r = 0
                g = 0
                b = 0
                # proposta do artigo
                # if(t>=0 and t<= 11/20):
                #    r = 0
                # elif(t>11/20 and t <4/5):
                #    r = 4*t-(11/5)
                # elif(t>=4/5 and t <=1):
                #    r = 1

                # if(t>=0 and t<= 1/5):
                #    g = 5*t
                # elif(t>1/5 and t <7/20):
                #    g = -(20/9)*t+(13/9)
                # elif(t>=7/20 and t <=4/5):
                #    g = -(20/27)*t+(11/27)
                # elif(t>=4/5 and t <=14/15):
                #    g = -(15/2)*t+7
                # elif(t>=14/15 and t <=19/20):
                #    g = 60*t-56
                # elif(t>=19/20 and t <=1):
                #    g = 1

                # if(t>=0 and t<= 1/5):
                #    b = 1
                # elif(t>1/5 and t <2/5):
                #    b = -4*t+9/5
                # elif(t>=2/5 and t <=14/15):
                #    b = 1/5
                # elif(t>14/15 and t <19/20):
                #    b = 60*t-56
                # elif(t>19/20 and t <1):
                #    b = 1

                # item C
                # r = (-0.5*math.cos(math.pi*t)+0.5)
                # g = (math.sin(math.pi*t))
                # b = (0.5*math.cos(math.pi*t)+0.5)

                # rainpallete
                c = int(((120-50)*((t-0)/(255-0)))+50)-1
                r = rain[c][0]
                g = rain[c][1]
                b = rain[c][2]

                newimg[i][j][k][0] = r
                newimg[i][j][k][1] = g
                newimg[i][j][k][2] = b

    return newimg


def showImage(image, window_name):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)


def showImageMatplot(image):
    plt.imshow(image, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
    plt.show()


def histogram(images):
    h = []
    for i in images:
        l = [0]*256
        for val in i.flatten():
            l[int(val)] += 1
        h.append(l)
    return h


def showHistogram(histogram):
    plt.bar(list(range(256)), list(histogram))
    plt.show()


def showRotatePoints(points):
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xticks(np.arange(-500, 500, 50))
    ax.set_yticks(np.arange(-200, 200, 50))
    ax.set_zticks(np.arange(-700, 700, 50))

    for j in range(0, 180, 30):
        tpoints = translatePts(points, x=0, y=0, z=0)
        ptContRotate1 = rotatePts(tpoints, angle=j, axis='y')[0]
        ax.plot([i[0] for i in ptContRotate1],
                [i[1] for i in ptContRotate1],
                [i[2] for i in ptContRotate1], label=j)

    plt.legend()
    plt.show()


def showpt(ref):
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xticks(np.arange(-500, 500, 20))
    ax.set_yticks(np.arange(-500, 500, 20))
    ax.set_zticks(np.arange(-500, 500, 20))

    # visao normal
    ax.plot([i[0] for i in ref[0]],
            [i[1] for i in ref[0]],
            [i[2] for i in ref[0]], label='base')

    plt.legend()
    plt.show()


def showPoints(points, ref, ell):
    v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xticks(np.arange(-500, 500, 50))
    ax.set_yticks(np.arange(-500, 500, 50))
    ax.set_zticks(np.arange(-500, 500, 50))

    arr = ['frontal', 'obliquo direito(45)', 'direito total(90)',
           'obliquo esquerdo(-45)', 'esquerdo total(-90)']

    for i in [1, 2, 3, 4]:
        ax.plot([i[0] for i in points[i][i]],
                [i[1] for i in points[i][i]],
                [i[2] for i in points[i][i]], marker='x', label=arr[i] if i < len(arr) else 'eita')

    # ax.plot([i[0] for i in ell[0][0]],
    #            [i[1] for i in ell[0][0]],
    #            [i[2] for i in ell[0][0]], label='ellipse')

    ax.plot([i[0] for i in points[5][1]],
            [i[1] for i in points[5][1]],
            [i[2] for i in points[5][1]], label='45 auxiliar')

    ax.plot([i[0] for i in points[6][3]],
            [i[1] for i in points[6][3]],
            [i[2] for i in points[6][3]], label='-45 auxiliar')

    ax.plot([i[0] for i in points[7][0]],
            [i[1] for i in points[7][0]],
            [i[2] for i in points[7][0]], label='central')

    ax.plot([i[0] for i in points[8]],
            [i[1] for i in points[8]],
            [i[2] for i in points[8]], marker='x', label='esquerda')

    ax.plot([i[0] for i in points[9]],
            [i[1] for i in points[9]],
            [i[2] for i in points[9]], marker='x', label='direita')

    plt.legend()
    plt.show()


def saveHistogram(histogram, folder, processo):
    for i in range(len(histogram)):
        plt.bar(list(range(256)), histogram[i])
        plt.savefig(folder+'/'+processo+'_'+str(i)+'.jpg', dpi=300)
        plt.close()


def equalizeHistogram(histogram, images, ref_pos):
    new_histograma = []
    for i in histogram:
        acum = [0]*len(i)
        for j in range(len(i)):
            if j == 0:
                acum[j] = i[j]
            else:
                acum[j] = acum[j-1]+i[j]
        new_histograma.append([(255/acum[-1])*acum[j] for j in range(len(i))])
    newimg = []
    for i in images:
        newimg.append(i.copy())
    for img in range(len(images)):
        for i in range(len(images[img])):
            for j in range(len(images[img][i])):
                newidx = images[img][i][j]
                newimg[img][i][j] = new_histograma[ref_pos][int(newidx)]
    return newimg


def otsuHistogram(images, histogram):
    l_threshold = []
    for im in range(len(images)):
        n = images[im].shape[0]*images[im].shape[1]
        threshold = var_max = -1
        sum = sumb = q1 = q2 = mi1 = mi2 = 0
        max_intensity = 255

        for i in range(len(histogram[im])):
            sum += i*histogram[im][i]

        for t in range(250):
            q1 += histogram[im][t]
            if q1 == 0:
                continue
            q2 = n-q1

            sumb += t*histogram[im][t]
            mi1 = sumb/q1
            mi2 = (sum-sumb)/q2

            var = q1*q2*pow((mi1-mi2), 2)

            if var > var_max:
                threshold = t
                var_max = var

        l_threshold.append(threshold)
    return l_threshold


def binarizeImg(images, threshold):
    newimg = []
    for i in images:
        newimg.append(i.copy())

    for i in range(len(images)):
        for j in range(len(images[i])):
            for k in range(len(images[i][j])):
                if (images[i][j][k] >= threshold[i]):
                    newimg[i][j][k] = 255
                else:
                    newimg[i][j][k] = 0
    return newimg


def segmentImg(images, mask):
    newimg = []
    for i in images:
        newimg.append(i.copy())
    for i in range(len(images)):
        for j in range(len(images[i])):
            for k in range(len(images[i][j])):
                if (mask[i][j][k] == 0):
                    newimg[i][j][k] = 0
    return newimg


def cannyImg(images, th1, th2):
    newimg = []
    for i in images:
        newimg.append(i.copy())
    start_time = time.time()
    for i in range(len(images)):
        ini = time.time()
        blur = cv2.medianBlur(newimg[i].astype(np.uint8), 11)
        newimg[i] = cv2.Canny(blur, th1, th2)
        print('Canny:'+'\t'+str(i)+'\t'+str(time.time()-ini) +
              '\t'+str(time.time()-start_time))
    return newimg


def drawContour(images):
    newimg = []
    start_time = time.time()
    for i in range(len(images)):
        ini = time.time()
        contours, hierarchy = cv2.findContours(images[i].astype(
            np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        auximg = np.zeros(
            (images[i].shape[0], images[i].shape[1], 1), np.uint8)
        for j in range(len(c)-1):
            cv2.line(auximg, (c[j][0][0], c[j][0][1]),
                     (c[j+1][0][0], c[j+1][0][1]), 255, 1)
        cv2.line(auximg, (c[len(c)-1][0][0], c[len(c)-1]
                 [0][1]), (c[0][0][0], c[0][0][1]), 255, 1)
        cv2.circle(auximg, (cX, cY), 5, (255, 255, 255), -1)
        newimg.append(auximg)
        # newimg.append(cv2.drawContours(np.zeros((images[i].shape[0],images[i].shape[1],3), np.uint8), c, -1, (0,255,0), 3))
        # print('Contorno:'+'\t'+str(i)+'\t'+str(time.time()-ini)+'\t'+str(time.time()-start_time))
    return newimg


def drawPoints(points, size, color, img=None):
    newimg = []
    start_time = time.time()
    for i in range(len(points)):
        ini = time.time()
        pts = points[i]
        auximg = None
        if img != None:
            auximg = img[i]
        else:
            auximg = np.zeros((size[0], size[1], 3), np.uint8)

        for j in range(len(pts)-1):
            cv2.line(auximg, ((int)(pts[j][0]), (int)(pts[j][1])), ((
                int)(pts[j+1][0]), (int)(pts[j+1][1])), color, 3)
        cv2.line(auximg, ((int)(pts[len(pts)-1][0]), (int)(pts[len(pts)-1][1])),
                 ((int)(pts[0][0]), (int)(pts[0][1])), color, 3)
        newimg.append(auximg)
        # newimg.append(cv2.drawContours(np.zeros((images[i].shape[0],images[i].shape[1],3), np.uint8), c, -1, (0,255,0), 3))
        # print('Contorno:'+'\t'+str(i)+'\t'+str(time.time()-ini)+'\t'+str(time.time()-start_time))
    return newimg


def getContour(images):
    newimg = []
    start_time = time.time()
    for i in range(len(images)):
        ini = time.time()
        contours, hierarchy = cv2.findContours(images[i].astype(
            np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)
        lp = []
        for j in range(len(c)):
            lp.append((c[j][0][0], c[j][0][1], 0))
        newimg.append(lp)

    return newimg


def translatePts(points, x=10, y=0, z=0):
    arrmat = np.array([[1, 0, 0, x],
                       [0, 1, 0, y],
                       [0, 0, 1, z],
                       [0, 0, 0, 1]])
    newpts = []
    for lpt in points:
        listpts = []
        for pt in lpt:
            arrpts = np.array([pt[0], pt[1], pt[2], 1])
            listpts.append(np.matmul(arrmat, arrpts)[0:3])
        newpts.append(listpts)

    return newpts


def moveToDistancePts(points, topoint):
    return np.array(topoint)-np.array(points)


def scalePts(points, x=10, y=0, z=0):
    arrmat = np.array([[x, 0, 0, 0],
                       [0, y, 0, 0],
                       [0, 0, z, 0],
                       [0, 0, 0, 1]])
    newpts = []
    for lpt in points:
        listpts = []
        for pt in lpt:
            arrpts = np.array([pt[0], pt[1], pt[2], 1])
            listpts.append(np.matmul(arrmat, arrpts)[0:3])
        newpts.append(listpts)

    return newpts


def rotatePts(points, angle=90, axis='x'):
    angle = (angle*math.pi)/180
    c = math.cos(angle)
    s = math.sin(angle)
    # z
    arrmat = np.array([[c, -s, 0, 0],
                       [s, c, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], float)
    if axis == 'x':
        arrmat = np.array([[1, 0, 0, 0],
                           [0, c, -s, 0],
                           [0, s, c, 0],
                           [0, 0, 0, 1]], float)
    elif axis == 'y':
        arrmat = np.array([[c, 0, s, 0],
                           [0, 1, 0, 0],
                           [-s, 0, c, 0],
                           [0, 0, 0, 1]], float)

    newpts = []
    for lpt in points:
        listpts = []
        for pt in lpt:
            arrpts = np.array([pt[0], pt[1], pt[2], 1], float)
            listpts.append(np.matmul(arrmat, arrpts))
        newpts.append(listpts)

    return newpts


def perspectivePts(points):
    for i in range(len(points)):
        for j in range(len(points[i])):
            if points[i][j][3] == 0:
                points[i][j] = [points[i][j][0],
                                points[i][j][1], points[i][j][2]]
            else:
                points[i][j] = [points[i][j][0],
                                points[i][j][1], points[i][j][2]]

    return points


def buildEllipse(points):
    lpts = []
    for p in points:
        pts = []
        for i in range(0, 360, 10):
            x = (p[1]/2)*math.cos((i*math.pi)/180)
            y = 0
            z = (p[2]/2)*math.sin((i*math.pi)/180)
            pts.append([(float)(x), (float)(y), (float)(z)])

        x = (p[1]/2)*math.cos((-45*math.pi)/180)
        y = 0
        z = (p[2]/2)*math.sin((-45*math.pi)/180)
        pts.append([(float)(x), (float)(y), (float)(z)])
        pts.append([(float)(x), -450, (float)(z)])
        pts.append([(float)(x), (float)(y), (float)(z)])
        x = (p[1]/2)*math.cos((-90*math.pi)/180)
        y = 0
        z = (p[2]/2)*math.sin((-90*math.pi)/180)
        pts.append([(float)(x), (float)(y), (float)(z)])
        pts.append([(float)(x), -450, (float)(z)])
        pts.append([(float)(x), (float)(y), (float)(z)])
        x = (p[1]/2)*math.cos((-135*math.pi)/180)
        y = 0
        z = (p[2]/2)*math.sin((-135*math.pi)/180)
        pts.append([(float)(x), (float)(y), (float)(z)])
        pts.append([(float)(x), -450, (float)(z)])
        lpts.append(pts)
    return lpts


def getEllipsePoint(points, angle, torotate=True):
    pts = []
    rotate = 0

    if torotate:
        x = (points[0][1]/2)*math.cos((-90*math.pi)/180)
        y = 0
        z = (points[0][2]/2)*math.sin((-90*math.pi)/180)
        pts.append([(float)(x), (float)(y), (float)(z)])
        rotate = rotatePts([pts], angle=-angle, axis='y')
    else:
        x = (points[0][1]/2)*math.cos((angle*math.pi)/180)
        y = 0
        z = (points[0][2]/2)*math.sin((angle*math.pi)/180)
        pts.append([(float)(x), (float)(y), (float)(z)])
        rotate = [[np.array([(float)(x), (float)(y), (float)(z)])]]

    return rotate


def coefAngular(x1, y1, x2, y2):
    # m= (y-y0)/(x-x0)
    resp = 0
    if ((x2 - x1) != 0):
        resp = (y2 - y1)/(x2 - x1)

    return resp


def coefLinear(x, y, ca):
    # n = y-mx
    return y-(ca*x)


def getLineSegment(x1, y1, x2, y2, val):
    # vector<float> val;
    # if(toimg){
    #    b.y *= -1;
    #    a.y *= -1;
    # }

    # val.push_back(coefAngular(a, b));
    # val.push_back(coefLinear(a, val[0]));

    return 1  # val;


def anisotropicFilter(img, n, k, g):
    imgs = []
    for i in img:
        imgs.append(i.copy())

    newimg = []
    for i in range(len(img)):

        newimg.append(anisotropic_diffusion(
            imgs[i], niter=n, kappa=k, gamma=g, option=2))
        # print('Contorno:'+'\t'+str(i)+'\t'+str(time.time()-ini)+'\t'+str(time.time()-start_time))

    return newimg


def imgROI(images):
    newimg = []
    for i in images:
        newimg.append(i.copy())

    roi = []*len(newimg)

    # get clean image
    for i in range(len(images)):
        h = images[i].shape[0]
        w = images[i].shape[1]
        down = np.zeros((h, w, 1), np.uint8)
        left = np.zeros((h, w, 1), np.uint8)
        right = np.zeros((h, w, 1), np.uint8)
        # Down
        for j in range(w):
            found = False
            for k in range(h-1, 0, -1):
                if (found):
                    down[k][j] = 0
                else:
                    if (images[i][k][j] != 0):
                        found = True
                        down[k][j] = 255
                    else:
                        down[k][j] = 0

        # Left
        for j in range(h):
            found = False
            for k in range(w-1, 0, -1):
                if (found):
                    left[j][k] = 0
                else:
                    if (images[i][j][k] != 0):
                        found = True
                        left[j][k] = 255
                    else:
                        left[j][k] = 0

        # Right
        for j in range(h):
            found = False
            for k in range(w):
                if (found):
                    right[j][k] = 0
                else:
                    if (images[i][j][k] != 0):
                        found = True
                        right[j][k] = 255
                    else:
                        right[j][k] = 0

        newimg[i] = cv2.subtract(cv2.subtract(down, left), right)
        # newimg[i] = cv2.add(cv2.add(cv2.bitwise_and(down,left),cv2.bitwise_and(down,right)),cv2.subtract(cv2.subtract(down,left),right))

    return newimg


def imgLeftRightRoi(images):
    newimgleft = []
    newimgright = []
    for i in images:
        newimgleft.append(i.copy())
        newimgright.append(i.copy())

    # get clean image
    for i in range(len(images)):
        h = images[i].shape[0]
        w = images[i].shape[1]
        down = np.zeros((h, w, 1), np.uint8)
        left = np.zeros((h, w, 1), np.uint8)
        right = np.zeros((h, w, 1), np.uint8)
        # Down
        for j in range(w):
            found = False
            for k in range(h-1, 0, -1):
                if (found):
                    down[k][j] = 0
                else:
                    if (images[i][k][j] != 0):
                        found = True
                        down[k][j] = 255
                    else:
                        down[k][j] = 0

        # Left
        for j in range(h):
            found = False
            for k in range(w-1, 0, -1):
                if (found):
                    left[j][k] = 0
                else:
                    if (images[i][j][k] != 0):
                        found = True
                        left[j][k] = 255
                    else:
                        left[j][k] = 0

        # Right
        for j in range(h):
            found = False
            for k in range(w):
                if (found):
                    right[j][k] = 0
                else:
                    if (images[i][j][k] != 0):
                        found = True
                        right[j][k] = 255
                    else:
                        right[j][k] = 0

        newimgleft[i] = cv2.subtract(left, down)
        newimgright[i] = cv2.subtract(right, down)

    pos_left = []
    pos_right = []
    for i in range(len(newimgleft)):
        h = newimgleft[i].shape[0]
        w = newimgleft[i].shape[1]
        b = []
        for j in range(w):
            for k in range(h):
                if (newimgleft[i][k][j] != 0):
                    b.append((j, k))
        pos_left.append(b)
    for i in range(len(newimgright)):
        h = newimgright[i].shape[0]
        w = newimgright[i].shape[1]
        b = []
        for j in range(w):
            for k in range(h):
                if (newimgright[i][k][j] != 0):
                    b.append((j, k))
        pos_right.append(b)

    return pos_left, pos_right


def ptsRoi(images, other):
    rois = [None]*len(images)

    newimg = []
    for i in other:
        newimg.append(i.copy())

    h = images[0].shape[0]
    w = images[0].shape[1]

    # 0, 45, -45
    for i in [0, 1, 3]:
        # Down
        intersec = [h, w]
        minleft = [0, 0]
        minright = [0, 0]

        # intercessao
        for j in range(w):
            found = False
            for k in range(h-1, 0, -1):
                if (images[i][k][j] != 0):
                    if (k < intersec[1]):
                        intersec = [j, k]
        # menor ponto pra direita
        for j in range(intersec[0], 0, -1):
            found = False
            for k in range(h-1, 0, -1):
                if (images[i][k][j] != 0):
                    if (k > minright[1]):
                        minright = [j, k]
        # menor ponto pra esquerda
        for j in range(intersec[0], w, 1):
            found = False
            for k in range(h-1, 0, -1):
                if (images[i][k][j] != 0):
                    if (k > minleft[1]):
                        minleft = [j, k]

        rois[i] = [(minright[0], minright[1]), (intersec[0],
                                                intersec[1]), (minleft[0], minleft[1])]

        cv2.circle(newimg[i], (intersec[0], intersec[1]), 3, 255, -1)
        cv2.circle(newimg[i], (minright[0], minright[1]), 3, 255, -1)
        cv2.circle(newimg[i], (minleft[0], minleft[1]), 3, 150, -1)
    # 90, -90
    for i in [2, 4]:
        start = [0, 0]
        min = [0, 0]
        end = [0, 0]

        # primeiro ponto pra direita
        found = False
        for j in range(w):
            for k in range(h-1, 0, -1):
                if (images[i][k][j] != 0 and not found):
                    if (i == 4):
                        start = [j, k]
                    else:
                        end = [j, k]
                    found = True

        # primeiro ponto para a esquerda
        found = False
        for j in range(w-1, 0, -1):
            for k in range(h-1, 0, -1):
                if (images[i][k][j] != 0 and not found):
                    if (i == 4):
                        end = [j, k]
                    else:
                        start = [j, k]
                    found = True
        # minimo entre eles
        for j in range(start[0], end[0], 1 if i == 4 else -1):
            found = False
            for k in range(h-1, 0, -1):
                if (images[i][k][j] != 0):
                    if (k > min[1]):
                        min = [j, k]

        rois[i] = [(start[0], start[1]), (min[0], min[1]), (end[0], end[1])]

        cv2.circle(newimg[i], (start[0], start[1]), 5, 255, -1)
        cv2.circle(newimg[i], (min[0], min[1]), 5, 255, -1)
        cv2.circle(newimg[i], (end[0], end[1]), 5, 150, -1)

    return newimg, rois


def getContourThreshold(contour, t):
    # pontos acima do limiar de z <= 0
    newcnt = []
    for i in contour:
        b = []
        for j in i:
            if j[2] <= t:
                b.append(j)
        newcnt.append(b)
    return newcnt


def findBSpline(path):
    # Create a B-Spline curve instance
    curve = BSpline.Curve()

    # Set up curve
    curve.degree = 4
    curve.ctrlpts = exchange.import_txt(path)

    # Auto-generate knot vector
    curve.knotvector = utilities.generate_knot_vector(
        curve.degree, len(curve.ctrlpts))

    # Set evaluation delta
    curve.delta = 0.01

    # Evaluate curve
    curve.evaluate()

    return curve.evalpts

# p0 = findBSpline("Processamento/Result/Objetos/0.cpt")

# print(p0)


def showSurface(path, name, u, v, texturepath):
    # Create a BSpline surface instance
    surf = NURBS.Surface()

    # Set degrees
    surf.degree_u = 8
    surf.degree_v = 2

    # Set control points
    surf.set_ctrlpts(*exchange.import_txt(path+name +
                     "_ctrlpoints.cptw", two_dimensional=True))

    # Set knot vectors
    surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, u)
    surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, v)

    # Set sample size and evaluate surface
    surf.sample_size = 50
    # surf.delta = 0.01
    # Evaluate surface tangent and normal at the given u and v

    surf.evaluate()

    datapath = path+name+"_data.txt"
    exchange.export_obj(surf, path+name+"_object.obj")
    f = open(datapath, "w")
    result = ""
    ctd = 0
    for i in surf.faces:
        for j in i.vertices:
            ctd = ctd+1
            color = [1.0, 1.0, 1.0]
            tx = [j[0]/256, j[1]/256]
            result = result+str(j[0])+' '+str(j[1])+' '+str(j[2])+' '+str(color[0])+' '+str(
                color[1])+' '+str(color[2])+' '+str(tx[0])+' '+str(tx[1])+' '

    f.write(texturepath+'\n')
    f.write(str(8*ctd)+'\n')
    f.write(result[:-1])
    f.close()

    # Import colormaps

    # Plot the surface
    vis_config = VisMPL.VisConfig(ctrlpts=False, axes=True, legend=False)
    vis_comp = VisMPL.VisSurface(vis_config)
    surf.vis = vis_comp
    surf.render(colormap=cm.coolwarm)

    # Plot the control point grid and the evaluated surface
    # vis_comp = VisPlotly.VisVolume()
    # surf.vis = vis_comp
    # surf.render()

    return datapath


"""
datapath = showSurface("Processamento/Result/Objetos/", objname, len(
    p0), 9, os.path.abspath(os.getcwd())+"/"+pathcolored[0])

print(datapath)
"""


def addNewPoint2Array(array, ref, ):
    if array[0][1] > ref:
        array.insert(0, np.array([array[0][0], ref, 0], float))
    return array


def quantImage(images, n):
    newimg = images.copy()
    r = (2**8)/(2**n)
    for i in range(len(images)):
        for j in range(len(images[i])):
            for k in range(len(images[i][j])):
                newimg[i][j][k] = (int)(images[i][j][k]/r)*r
    return newimg


def superpixel(images):
    newimg = []
    start_time = time.time()
    for i in range(len(images)):
        ini = time.time()
        newimg.append(mark_boundaries(images[i], felzenszwalb(
            images[i], scale=200, sigma=0.5, min_size=480)))
    return newimg


# verificar depois
# surf
def surf(images):
    newimg = []
    for i in range(len(images)):
        # (edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=500)
        orb = cv2.ORB_create()
        kp = orb.detect(images[i].astype(np.uint8), None)
        kp, des = orb.compute(images[i].astype(np.uint8), kp)
        newimg.append(cv2.drawKeypoints(images[i].astype(
            np.uint8), kp, None, color=(0, 255, 0), flags=0))
        # showImageMatplot(cv2.drawKeypoints(images[i].astype(np.uint8), kp, None, color=(0,255,0), flags=0))
    return newimg

# surf


def detectAndDescribe(image, method=None):
    """
    Compute key points and feature descriptors using an specific method
    """

    assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf'"

    # detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()

    # get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)

    return (kps, features)


def createMatcher(method, crossCheck):
    "Create and return a Matcher Object"

    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf


def matchKeyPointsBF(featuresA, featuresB, method):
    bf = createMatcher(method, crossCheck=True)

    # Match descriptors.
    best_matches = bf.match(featuresA, featuresB)

    # Sort the features in order of distance.
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_matches, key=lambda x: x.distance)
    print("Raw matches (Brute force):", len(rawMatches))
    return rawMatches


def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
    # compute the raw matches and initialize the list of actual matches
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    print("Raw matches (knn):", len(rawMatches))
    matches = []

    # loop over the raw matches
    for m, n in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches


def merge(images1, images2):
    feature_extractor = 'orb'  # one of 'sift', 'surf', 'brisk', 'orb'
    feature_matching = 'bf'
    trainImg = images1.astype(np.uint8)
    queryImg = images2.astype(np.uint8)
    kpsA, featuresA = detectAndDescribe(trainImg, method=feature_extractor)
    kpsB, featuresB = detectAndDescribe(queryImg, method=feature_extractor)

    fig = plt.figure(figsize=(20, 8))

    if feature_matching == 'bf':
        matches = matchKeyPointsBF(
            featuresA, featuresB, method=feature_extractor)
        img3 = cv2.drawMatches(trainImg, kpsA, queryImg, kpsB, matches[:100],
                               None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    elif feature_matching == 'knn':
        matches = matchKeyPointsKNN(
            featuresA, featuresB, ratio=0.75, method=feature_extractor)
        img3 = cv2.drawMatches(trainImg, kpsA, queryImg, kpsB, np.random.choice(matches, 100),
                               None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(img3)
    plt.show()
    return
