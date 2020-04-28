import os
import shutil

if __name__ == "__main__":
    # publish_folders = [
    #     "build",
    #     "dist",
    #     "pysy.egg-info"
    # ]
    # print("checking dirs...")
    cur_dirs = os.listdir()
    # build   
    os.system("python setup.py sdist bdist_wheel")
    print("package is built...")
    # push
    os.system("twine upload --repository-url https://upload.pypi.org/legacy/ dist/*")
    print("package is publised...")
    print("clearing up...")
    new_dirs = [p for p in os.listdir() if p not in cur_dirs]
    for p in new_dirs:
        # print(p)
        shutil.rmtree(p)
    print("all done.")