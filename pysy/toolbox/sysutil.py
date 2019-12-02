import yaml
import zipfile
import multiprocessing
import sys
import itertools
import numbers
from pathlib import Path

class Yaml(object):
    def __init__(self, path):
        super()
        self.path = path
        if isinstance(path, Path):
            self.path = self.path.as_posix() 

    def load(self):
        with open(self.path, "r") as stream:
            try:
                yamlfile = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                assert(exc)
        return yamlfile
    
    def dump(self, data_dict):
        with open(self.path, 'w') as f:
            yaml.dump(data_dict, f, default_flow_style = False)

# extract a .zip (p) into a folder at tar_dir of a name p.stem
def unzip(p, tar_dir, new_folder = True, delete = False):
    # print(p)
    if not isinstance(p, Path):
        p = Path(p)
    if not isinstance(tar_dir, Path):
        tar_dir = Path(tar_dir)
    if new_folder:
        out_dir = tar_dir.joinpath(p.stem)
    else:
        out_dir = tar_dir
    # if not out_dir.exists(): out_dir.mkdir()
    create_all_parents(out_dir)
    with zipfile.ZipFile(p, "r") as zip_ref:
        zip_ref.extractall(out_dir)
    if delete:
        p.unlink()

# if parent or grand parent dirs not exist, 
# make them including current dir (if its not a file)
def create_all_parents(directory):
    if not isinstance(directory, Path):
        directory = Path(directory)
    parents = list(directory.parents)
    parents.reverse()
    # NOTICE: sometimes is_dir returns false, e.g., a dir of onedrive
    if not directory.is_file():
        parents.append(directory)
    for p in parents:
      if not p.exists():
        p.mkdir()

# search current dir and list the subdirs
def searching_all_files(directory):
    if not isinstance(directory, Path):
        dirpath = Path(directory)
    assert(dirpath.is_dir())
    file_list = []
    for x in dirpath.iterdir():
        if x.is_file():
            file_list.append(x)
        elif x.is_dir():
            file_list.extend(searching_all_files(x))
    return file_list

# print progress bar at one line.
def pbar(idx, total, auto_check = False):
    if auto_check:
        if not isinstance(idx, numbers.Number):
            idx = float(idx)
        if not isinstance(total, numbers.Number):
            total = float(total)
    if (100*(idx + 1)/total).is_integer():
        sys.stdout.write(f"progress reaches {idx + 1} of {total}, {100*(idx + 1)/total}% ...\r")

# class Parallel(object):
#     def __init__(self, core_num = None, *args, **kwargs):
#         """
#         example:
#         def func(x):
#             return x * x
#         seq = range(1500)
#         """
#         super()
#         # initialize multicores
#         if not core_num:
#             cores = multiprocessing.cpu_count()
#         self.pool  = multiprocessing.Pool(processes=cores)

#     def map(self, func, seq):
#         # method 1: map
#         self.res = self.pool.map(func, seq)  # prints [0, 1, 4, 9, 16]

#     def imap(self, func, seq):
#         # method 2: imap
#         for y in self.pool.imap(func, seq):
#             print(y)           # 0, 1, 4, 9, 16, respectively

#     def imap_unordered(self, func, seq):
#         # method 3: imap_unordered
#         for y in self.pool.imap_unordered(func, seq):
#             print(y)           # may be in any order

#     def imap_unordered_print(self, func, seq):
#         cnt = 0
#         for _ in self.pool.imap_unordered(func, seq):
#             # print(_)
#             sys.stdout.write('done %d/%d\r' % (cnt, len(seq)))
#             cnt += 1
#     def starmap(self, func, seq):
#         self.res = self.pool.starmap(func, seq)