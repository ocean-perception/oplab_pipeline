# How to release the package to Pypi

The first thing you’ll need to do is register an account on PyPI (pypi.org). Then, install dependencies for releasing the package:

    python3 -m pip install --user --upgrade setuptools wheel

Now run this command from the same directory where setup.py is located:

    python3 setup.py sdist bdist_wheel

Now that you are registered im PyPI, you can use twine to upload the distribution packages. You’ll need to install Twine:

    python3 -m pip install --user --upgrade twine

Once installed, run Twine to upload all of the archives under dist:
 
    twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

You will be prompted for the username and password you registered with PyPI.