import os
import glob


def getFiles(database, ext):
    path = database+'*.'+ext
    files = glob.glob(path)
    filesorted = sorted(files)

    """
  files = []
  for n in os.listdir(database):
    if(not os.path.isdir(database+n)):
      files.append(database+n)
  """
    return filesorted
