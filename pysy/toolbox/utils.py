import yaml
import zipfile
import multiprocessing
import sys
import itertools
import numbers
from pathlib import Path
from datetime import datetime, timedelta, date # date type is not datetime, it only accepts year, month and day.
from time import gmtime, strftime, ctime

class Montre(object):
	def __init__(self):
		super()

	def to_date(self, date_str, format = r"%Y-%m-%d"):
		return datetime.strptime(date_str, format)

	def to_str(self, cur_date, format = r"%Y-%m-%d"):
		return cur_date.strftime(format)

	# Check if the int given year is a leap year
	# return true if leap year or false otherwise
	def is_leap_year(self, year):
		if(year % 4) == 0:
			if(year % 100) == 0:
				if(year % 400) == 0:
					return True
				else:
					return False
			else:
				return True
		else:
			return False

	def manage_time(self, cur_date, years = 0, months = 0, weeks = 0, days = 0, hours = 0, minutes = 0, seconds = 0):
		# the finest scale is second.
		# input time must be datetime type
		if not isinstance(cur_date, datetime):
			raise(ValueError)
		# set output format
		format = r"%Y-%m-%d %H:%M:%S.%f"
		# disintegrate input time into subitems
		cur_year = cur_date.year
		cur_month = cur_date.month
		cur_day = cur_date.day
		cur_hour = cur_date.hour
		cur_minute = cur_date.minute
		cur_second = cur_date.second
		cur_ms = cur_date.microsecond
		# manage year add/substract
		if years != 0:
			cur_year = cur_year + years
		# mange month add/substract
		cur_month = cur_month + months
		if cur_month > 12:
			cur_year = int(cur_year + cur_month // 12)
			cur_month = int(cur_month % 12)

		if (cur_month == 2) and (cur_day > 28):
			if self.is_leap_year(cur_year):
				cur_day = 29
			else:
				cur_day = 28
		str_date = f"{cur_year}-{cur_month}-{cur_day} {cur_hour}:{cur_minute}:{cur_second}.{cur_ms}" 
		cur_date = datetime.strptime(str_date, format)
		# manage the rest
		delta_seconds = 0
		if weeks != 0:
			delta_seconds = delta_seconds +  weeks * 7 * 24 * 60 * 60
		if days != 0:
			delta_seconds = delta_seconds + days * 24 * 60 * 60
		if hours != 0:
			delta_seconds = delta_seconds + hours * 60 * 60
		if minutes != 0:
			delta_seconds = delta_seconds + minutes * 60
		if seconds != 0:
			delta_seconds = delta_seconds + seconds
		cur_date = cur_date + timedelta(seconds = delta_seconds)
		return cur_date

class Yaml(object):
    def __init__(self, path):
        if isinstance(path, str):
            path = Path(path)
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
def unzip(p, tar_dir, new_folder = True, folder_name = None, delete = False):
    # print(p)
    if not isinstance(p, Path):
        p = Path(p)
    if not isinstance(tar_dir, Path):
        tar_dir = Path(tar_dir)
    if new_folder:
        if not folder_name:
            out_dir = tar_dir.joinpath(p.stem)
        else:
            out_dir = tar_dir.joinpath(folder_name)
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
def create_all_parents(directory, flag = "a"):
    # flag: if directory is a dir ("d") or file ("f") or automatically desice ("a")
    if not isinstance(directory, Path):
        directory = Path(directory)
    parents = list(directory.parents)
    parents.reverse()
    # NOTICE: sometimes is_dir returns false, e.g., a dir of onedrive
    if flag == "a":
        if not directory.is_file():
            parents.append(directory)
    elif flag == "d":
        parents.append(directory)
    else:
        pass
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


# mount the drive
def mount_drive():
    from google.colab import drive
    from pathlib import Path
  
    drive.mount('/content/drive')
    return Path.cwd().joinpath('drive/My Drive')