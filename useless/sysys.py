# coding: utf-8
import os

def syrename(path, newname):
    d = list_files_paths(path)
    for file in files:
        os.rename(file, newname)

def list_files_paths(path):
    files = os.listdir(path)
    paths = [os.path.join(path, file) for file in files]
    # os.path.basename(path)
    # os.path.dirname(path)
    # os.path.splitext()
    # os.path.abspath(name)
    # os.path.split()
    # os.path.isfile()
    # os.path.isdir()
    # os.path.exists()
    # os.system()
    # os.remove()
    return {"f": files, "p": paths}

def sy_mkdir_in_out(path):
    # import shutil
    # os.mkdirs()
    os.mkdir(os.path.join(path, "/in"))
    os.mkdir(os.path.join(path, "/out"))
    os.mkdir(os.path.join(path, "/data"))

def r_pickle(fullpath):
    import pickle
    file = open(fullpath,'rb')
    data = pickle.load(file)
    file.close()
    return data

def w_pickle(fullpath, data):
    import pickle
    file = open(fullpath,'wb')
    pickle.dump(data, file, -1)
    file.close()