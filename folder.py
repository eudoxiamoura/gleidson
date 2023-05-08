import os
import glob
import cv2
def getFiles(database, ext):
  
  path = database + '/*.' + ext
  files = glob.glob(path)
  """
  files = []
  for n in os.listdir(database):
    if(not os.path.isdir(database+n)):
      files.append(database+n)
  """
  return files

img_paths = getFiles('Processamento/Result/Real', 'jpg')
print(img_paths)