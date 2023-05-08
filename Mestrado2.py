import image
import folder
import config as c
import cv2
import numpy as np
import string
import os

import subprocess

import math


# to get paths in original folder
# get paths
# img_paths = folder.getFiles('dmruff\\*\\', 'txt')
# get just the paths wanted (5 imagens)
# workpaths = [i for i in img_paths if i.split("\\")[-1].split('.')[3]=='S'][0:50];
# get the files content
# temperaturas = image.loadFile(workpaths)
# convert temperatures into image
# imagens = image.convertToImage(temperaturas, [480,640])
# newpathes = image.saveImagesDefinedPath(imagens, 'dmruffProc2\\', workpaths)


# to get paths in matriz to image done
# get paths
print("Loading Files")
img_paths = folder.getFiles('Processamento/Result/Real', 'jpg')
seg_paths = folder.getFiles('Processamento/Result/Segmentado', 'jpg')
# get just the paths wanted (5 imagens)
img_workpaths = img_paths[100:105]
seg_workpaths = seg_paths[100:105]
# get the files content
imagens = image.loadImage(img_workpaths, 0)
seg = image.loadImage(seg_workpaths, 0)


img_name = ''.join(img_workpaths[0].split('/')[-1].split('.')[0])

img_newpathes = image.saveImages(
    imagens, 'Processamento/Result/Proc', '0-original')
seg_newpathes = image.saveImages(
    seg, 'Processamento/Result/Proc', '1-segmentadao')

print("Processing")
bin = image.binarizeImg(seg, [50]*len(seg))
# imagens segmentadas
segmentada = image.segmentImg(imagens, bin)

# contorno
cont = image.drawContour(bin)

loi = image.imgROI(cont)
pleft, pright = image.imgLeftRightRoi(cont)

roi_img, roi_pts = image.ptsRoi(loi, segmentada)

print("Saving Images")
# save results
pathbin = image.saveImages(bin, 'Processamento/Result/Proc', '2-binarizada')
pathseg = image.saveImages(
    segmentada, 'Processamento/Result/Proc', '3-segmentado')
pathcont = image.saveImages(cont, 'Processamento/Result/Proc', '5-contorno')
pathloi = image.saveImages(loi, 'Processamento/Result/Proc', '5-interesse')
pathroi_img = image.saveImages(
    roi_img, 'Processamento/Result/Proc', '6-interessepts')

# image.saveImagesSided(bin, 'Result\\ProcFull\\', img_name+'_2-binarizada')
# image.saveImagesSided(segmentada, 'Result\\ProcFull\\', img_name+'_3-segmentado')
# image.saveImagesSided(cont, 'Result\\ProcFull\\', img_name+'_5-contorno')
# image.saveImagesSided(loi, 'Result\\ProcFull\\', img_name+'_5-interesse')
# image.saveImagesSided(roi_img, 'Result\\ProcFull\\', img_name+'_6-interessepts')

##################################
######## ESPACIONALIZAÇÃO##########
##################################

ptContorno = image.getContour(bin)

Dw = abs(roi_pts[0][0][0]-roi_pts[0][2][0])
dw = abs(roi_pts[2][0][0]-roi_pts[2][2][0]) if abs(roi_pts[2][0][0]-roi_pts[2][2][0]
                                                   ) > abs(roi_pts[4][0][0]-roi_pts[4][2][0]) else abs(roi_pts[4][0][0]-roi_pts[4][2][0])
h = (roi_pts[0][1][1]+roi_pts[0][0][1]+roi_pts[0][2][1])/2


ptsellipse = [[(0, 0, 0), Dw, dw, -90]]*5
ptEllipse = image.buildEllipse(ptsellipse)

# converte os pontos de interesse em array
newroi_pts = []
for i in roi_pts:
    b = []
    for j in i:
        b.append(np.array([j[0], j[1], 0], float))
    newroi_pts.append(b)

print("Volume")
# calcula a distancia dos pontos para as suas respectivas posições para o centro
move0 = image.moveToDistancePts(
    [roi_pts[0][1][0], roi_pts[0][1][1], 0], [0, 0, 0])
move45 = image.moveToDistancePts(
    [roi_pts[1][2][0], roi_pts[1][2][1], 0], [0, 0, 0])
move90 = image.moveToDistancePts(
    [roi_pts[2][1][0], roi_pts[2][1][1], 0], [0, 0, 0])
movem45 = image.moveToDistancePts(
    [roi_pts[3][0][0], roi_pts[3][0][1], 0], [0, 0, 0])
movem90 = image.moveToDistancePts(
    [roi_pts[4][1][0], roi_pts[4][1][1], 0], [0, 0, 0])

# move todo mundo para o centro
ptCont0 = image.translatePts(ptContorno, x=move0[0], y=move0[1], z=move0[2])
ptCont45 = image.translatePts(
    ptContorno, x=move45[0], y=move45[1], z=move45[2])
ptCont90 = image.translatePts(
    ptContorno, x=move90[0], y=move90[1], z=move90[2])
ptContm45 = image.translatePts(
    ptContorno, x=movem45[0], y=movem45[1], z=movem45[2])
ptContm90 = image.translatePts(
    ptContorno, x=movem90[0], y=movem90[1], z=movem90[2])
# move todos os pontos de interesse para o centro
ptRoi0 = image.translatePts(newroi_pts, x=move0[0], y=move0[1], z=move0[2])
ptRoi45 = image.translatePts(newroi_pts, x=move45[0], y=move45[1], z=move45[2])
ptRoi90 = image.translatePts(newroi_pts, x=move90[0], y=move90[1], z=move90[2])
ptRoim45 = image.translatePts(
    newroi_pts, x=movem45[0], y=movem45[1], z=movem45[2])
ptRoim90 = image.translatePts(
    newroi_pts, x=movem90[0], y=movem90[1], z=movem90[2])

# rotaciona todo mundo em graus
ptContRotate45 = image.rotatePts(ptCont45, angle=45, axis='y')
ptContRotate45aux = image.rotatePts(ptCont45, angle=135, axis='y')
ptContRotate90 = image.rotatePts(ptCont90, angle=90, axis='y')
ptContRotatem45 = image.rotatePts(ptContm45, angle=-45, axis='y')
ptContRotatem45aux = image.rotatePts(ptContm45, angle=-135, axis='y')
ptContRotatem90 = image.rotatePts(ptContm90, angle=-90, axis='y')
# rotaciona todos os pontos de interesse em 45 graus
ptRoiRotate45 = image.rotatePts(ptRoi45, angle=45, axis='y')
ptRoiRotate45aux = image.rotatePts(ptRoi45, angle=135, axis='y')
ptRoiRotate90 = image.rotatePts(ptRoi90, angle=90, axis='y')
ptRoiRotatem45aux = image.rotatePts(ptRoim45, angle=-135, axis='y')
ptRoiRotatem90 = image.rotatePts(ptRoim90, angle=-90, axis='y')

# rotaciona a elipse em graus
ptEllipseRotate45 = image.rotatePts(ptEllipse, angle=-45, axis='y')
ptEllipseRotate90 = image.rotatePts(ptEllipse, angle=-90, axis='y')
ptEllipseRotatem45 = image.rotatePts(ptEllipse, angle=45, axis='y')
ptEllipseRotatem90 = image.rotatePts(ptEllipse, angle=90, axis='y')


# calcula a distancia dos pontos para as suas respectivas posições
movepts45 = image.moveToDistancePts(
    [0, 0, 0], [ptRoi0[0][2][0], ptRoi0[0][2][1], 0])
movepts90 = image.moveToDistancePts(
    [0, 0, 0], [ptRoi0[0][0][0], ptRoi0[0][0][1], 0])
moveptsm45 = image.moveToDistancePts(
    [0, 0, 0], [ptRoi0[0][0][0], ptRoi0[0][0][1], 0])
moveptsm90 = image.moveToDistancePts(
    [0, 0, 0], [ptRoi0[0][2][0], ptRoi0[0][2][1], 0])

# move a distancia dos pontos para as suas respectivas posições
ptContRotate45 = image.translatePts(
    ptContRotate45, x=movepts45[0], y=movepts45[1], z=movepts45[2])
ptContRotate45aux = image.translatePts(
    ptContRotate45aux, x=movepts45[0], y=movepts45[1], z=movepts45[2])
ptContRotate90 = image.translatePts(
    ptContRotate90, x=movepts90[0], y=movepts90[1], z=movepts90[2])
ptContRotatem45 = image.translatePts(
    ptContRotatem45, x=moveptsm45[0], y=moveptsm45[1], z=moveptsm45[2])
ptContRotatem45aux = image.translatePts(
    ptContRotatem45aux, x=moveptsm45[0], y=moveptsm45[1], z=moveptsm45[2])
ptContRotatem90 = image.translatePts(
    ptContRotatem90, x=moveptsm90[0], y=moveptsm90[1], z=moveptsm90[2])


# roi_pts[0][1][0],roi_pts[0][1][1],0]

# voltar a posicao da imagem
ptCont0 = image.translatePts(ptCont0, x=-move0[0], y=-move0[1], z=-move0[2])
ptRoi0 = image.translatePts(ptRoi0, x=-move0[0], y=-move0[1], z=-move0[2])
########################################

b = []
left = []
right = []
for i in ptCont0[0]:
    if i[0] < ptRoi0[0][0][0] or i[0] > ptRoi0[0][2][0]:
        b.append(i)
    if i[0] < ptRoi0[0][0][0]:
        min = 5
        for j in pright[0]:
            if np.linalg.norm(i-np.array((j[0], j[1], 0))) < min:
                min = np.linalg.norm(i-np.array((j[0], j[1], 0)))
        if min < 5:
            left.append(i)
    if i[0] > ptRoi0[0][2][0]:
        min = 5
        for j in pleft[0]:
            if np.linalg.norm(i-np.array((j[0], j[1], 0))) < min:
                min = np.linalg.norm(i-np.array((j[0], j[1], 0)))
        if min < 5:
            right.append(i)

ptCont0[0] = b

miny = ptCont0[0][0]
maxy = ptCont0[0][0]

for i in ptCont0[0]:
    if (i[1] < miny[1]):
        miny = i
    if (i[1] > maxy[1]):
        maxy = i

middle = [[np.array([ptRoi0[0][1][0], miny[1], 0]), np.array(
    [ptRoi0[0][1][0], miny[1], 0]), ptRoi0[0][1], ptRoi0[0][1], ptRoi0[0][1]]]*5

# pontos acima do limiar de z <= 0
newptContRotate45 = image.getContourThreshold(ptContRotate45, 0)
newptContRotate45aux = image.getContourThreshold(ptContRotate45aux, 0)
newptContRotate90 = image.getContourThreshold(ptContRotate90, 0)
newptContRotatem45 = image.getContourThreshold(ptContRotatem45, 0)
newptContRotatem45aux = image.getContourThreshold(ptContRotatem45aux, 0)
newptContRotatem90 = image.getContourThreshold(ptContRotatem90, 0)

# voltar a posicao da imagem
newptContRotate45 = image.translatePts(
    newptContRotate45, x=-move0[0], y=-move0[1], z=-move0[2])
newptContRotate45aux = image.translatePts(
    newptContRotate45aux, x=-move0[0], y=-move0[1], z=-move0[2])
newptContRotate90 = image.translatePts(
    newptContRotate90, x=-move0[0], y=-move0[1], z=-move0[2])
newptContRotatem45 = image.translatePts(
    newptContRotatem45, x=-move0[0], y=-move0[1], z=-move0[2])
newptContRotatem45aux = image.translatePts(
    newptContRotatem45aux, x=-move0[0], y=-move0[1], z=-move0[2])
newptContRotatem90 = image.translatePts(
    newptContRotatem90, x=-move0[0], y=-move0[1], z=-move0[2])
########################################

# mostrar os pontos encontrados matplot
# image.showPoints([ptCont0, newptContRotate45, newptContRotate90, newptContRotatem45, newptContRotatem90, newptContRotate45aux, newptContRotatem45aux, middle, left, right],
#                  [None, None, None, None, None],
#                 [ptEllipse, ptEllipseRotate45, ptEllipseRotate90, ptEllipseRotatem45, ptEllipseRotatem90])


#### normalizar pontos###################
left = image.addNewPoint2Array(left, middle[0][0][1])

newptContRotate45 = [newptContRotate45[1][i]
                     for i in range(len(newptContRotate45[1])-1, -1, -1)]
newptContRotate45 = image.addNewPoint2Array(newptContRotate45, middle[0][0][1])

newptContRotate90 = [newptContRotate90[2][i]
                     for i in range(len(newptContRotate90[2])-1, -1, -1)]
newptContRotate90 = image.addNewPoint2Array(newptContRotate90, middle[0][0][1])

newptContRotate45aux = [newptContRotate45aux[1][i]
                        for i in range(len(newptContRotate45aux[1])-1, -1, -1)]
newptContRotate45aux = image.addNewPoint2Array(
    newptContRotate45aux, middle[0][0][1])

newptContRotatem45aux = image.addNewPoint2Array(
    newptContRotatem45aux[3], middle[0][0][1])

newptContRotatem90 = image.addNewPoint2Array(
    newptContRotatem90[4], middle[0][0][1])

newptContRotatem45 = image.addNewPoint2Array(
    newptContRotatem45[3], middle[0][0][1])

right = [right[i] for i in range(len(right)-1, -1, -1)]
right = image.addNewPoint2Array(right, middle[0][0][1])
########################################

#################################################################
########### REPRESENTAR GRAFICAMENTE OS PONTOS####################
#################################################################

# SALVAR OS PONTOS ENCONTRADOS ATÉ ENTAO
image.saveFile(str(0), left)
image.saveFile(str(1), newptContRotate45)
image.saveFile(str(2), newptContRotate90)
image.saveFile(str(3), newptContRotate45aux)
image.saveFile(str(4), middle[0])
image.saveFile(str(5), newptContRotatem45aux)
image.saveFile(str(6), newptContRotatem90)
image.saveFile(str(7), newptContRotatem45)
image.saveFile(str(8), right)


print("Spline")
# EXECUCAO DE BSPLINE PARA ENCONTRAR OS PONTOS INTERPOLADOS
p0 = image.findBSpline("Processamento/Result/Objetos/0.cpt")
p1 = image.findBSpline("Processamento/Result/Objetos/1.cpt")
p2 = image.findBSpline("Processamento/Result/Objetos/2.cpt")
p3 = image.findBSpline("Processamento/Result/Objetos/3.cpt")
p4 = image.findBSpline("Processamento/Result/Objetos/4.cpt")
p5 = image.findBSpline("Processamento/Result/Objetos/5.cpt")
p6 = image.findBSpline("Processamento/Result/Objetos/6.cpt")
p7 = image.findBSpline("Processamento/Result/Objetos/7.cpt")
p8 = image.findBSpline("Processamento/Result/Objetos/8.cpt")

# SALVAR OS PONTOS DA SUPERFICIE COMPLETAMENTE
objname = ''.join(img_workpaths[0].split('/')[-1].split('.')[:-1])

print("NURBS Surface")
image.saveFileFullPts("Processamento/Result/Objetos/", objname, [
                      p0, p7, p2, p5, p4, p3, p6, p1, p8], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

print("Coloring")
colored = image.gray2Color(imagens)
pathcolored = image.saveImages(
    colored, 'Processamento/Result/Proc', img_name+'_7-fakecolor')
# image.saveImagesSided(colored, 'Result\\ProcFull\\', img_name+'_7-fakecolor')

image.saveImagesSided([imagens, segmentada],
                      'Processamento/Result/ProcFull', img_name, False)


# MOSTRAR A SUPERFÍCIE
"""
var = len(p0)
print(var)
print("OBJNAME:"+objname)
print("IMGNAME:"+img_name)
print("pathcolored:")
print(pathcolored)
print("os.path.abspath:")
print(os.path.abspath(os.getcwd())+"/"+pathcolored[0])
"""
print("Rendering")
datapath = image.showSurface("Processamento/Result/Objetos/", objname, len(
    p0), 9, os.path.abspath(os.getcwd())+"/"+pathcolored[0])
print(datapath)

"""
subprocess.Popen(
    ["Processamento/RenderizarObj/RenderizarObjetos.exe", datapath])
"""

# quantificacao
# quant = image.quantImage(segmentada, 3)
# image.saveImages(quant, 'Result\\', '6-quantificacao')
# surf
# surf = image.surf(segmentada)
# image.saveImages(surf, 'Result\\', '8-keypoints')
# superpixel
# super = image.superpixel(segmentada)
# image.saveImages(super, 'Result\\', '7-superpixel')
# mergeimages
# merge1 = image.merge(segmentada[0], segmentada[1])
# merge2 = image.merge(segmentada[1], segmentada[2])
# merge3 = image.merge(segmentada[0], segmentada[3])
# merge4 = image.merge(segmentada[3], segmentada[4])
