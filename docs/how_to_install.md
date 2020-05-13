# How to install

To download the code, go to directory you want download the code to, open a terminal/command prompt there and type
```
git clone https://github.com/ocean-perception/oplab_pipeline.git
```

To push updates you made to the repository on github (assuming you are using the master branch, which is the default), type
```
git add -u
git commit -m "Some message about the change"
git push origin master
```

## Running the code ##
Requires [Python3.6.2](https://www.python.org/downloads/release/python-362/) or ([latest](https://www.python.org/downloads/release/python3)).  
You can also use [Anaconda](https://www.anaconda.com/download/), which comes with the Spyder IDE, that in turn uses the IPython shell.

### When using Python from standard terminal or the Anaconda prompt ###
The oplab_pipeline project is compiled into binaries. Navigate to the oplab_pipeline folder and execute `pip install .`.  
`auv_nav`, `auv_cal` and `correct_images` can now be used as a command from any folder on your computer. You can test that by trying to display the help by calling `auv_nav -h`.  
To run the unit tests, execute `pytest` from within the oplab_pipeline folder.  
To uninstall the oplab_pipeline programs run `pip uninstall oplab_pipeline`.

### When using Spyder IDE in Anaconda ###
When installing the oplab_pipeline as described above, it is not possilbe to call `auv_nav`, `auv_cal` and `correct_images` directly from within the Spyder IDE (which uses IPython console), the reason being that IPython does not support execution of binaries.  
To get the programs to work in the the Spyder IDE, first install oplab_pipeline, as described above, in order to install all the required packages. If you don't want to do that, you can also install the required packages listed in requirements.txt manually.
Then open Spyder and use its shell (it is the IPython shell) to navigate to the oplab_pipeline folder and execute
```
import auv_nav
run auv_nav/auv_nav.py -h
```
This should display the help. To use auv_nav, replace `-h` with the command you want to run.  
This way you can run auv_nav from within Spyder, but you always need to navigate to the oplab_pipeline directory to execute the code.  
`auv_cal` and `correct_images` can be exectuted in the same way.
