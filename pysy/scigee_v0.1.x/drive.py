from google.colab import drive
from pathlib import Path

# mount the drive
def mount_drive():
  drive.mount('/content/drive')
  return Path.cwd().joinpath('drive/My Drive')