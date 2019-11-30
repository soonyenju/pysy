import os

if __name__ == "__main__":
    # build
    os.system("python setup.py sdist bdist_wheel")
    print("package is built...")
    # push
    os.system("twine upload --repository-url https://upload.pypi.org/legacy/ dist/*")
    print("package is publised...")