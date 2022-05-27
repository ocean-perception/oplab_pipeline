[![oplab_pipeline](https://github.com/ocean-perception/oplab_pipeline/actions/workflows/oplab_pipeline.yml/badge.svg)](https://github.com/ocean-perception/oplab_pipeline/actions/workflows/oplab_pipeline.yml)
[![Code Coverage](https://codecov.io/gh/ocean-perception/oplab_pipeline/branch/master/graph/badge.svg?token=PJBfl6qhp5)](https://codecov.io/gh/ocean-perception/oplab_pipeline) [![Documentation Status](https://readthedocs.org/projects/oplab-pipeline/badge/?version=latest)](https://oplab-pipeline.readthedocs.io/en/latest/?badge=latest) [![Docker Image CI](https://github.com/ocean-perception/oplab_pipeline/actions/workflows/docker_image.yml/badge.svg)](https://github.com/ocean-perception/oplab_pipeline/actions/workflows/docker_image.yml)


# oplab_pipeline

oplab_pipeline is a python toolchain to process AUV dives from raw data into navigation and imaging products. The software is capable of:

- Process navigation: fuses AUV or ROV sensor data using state of the art filters and geolocalises recorded imagery.
- Camera and laser calibration: performs automatic calibration pattern detection to calibrate monocular or stereo cameras. Also calibrates laser sheets with respect to the cameras.
- Image correction: performs pixel-wise image corrections to enhance colour and contrast in underwater images.

Please review the latest changes in the [CHANGELOG.md](CHANGELOG.md). 


## Installation
`cd` into the oplab-pipeline folder and run `pip3 install .`, resp. if you are using Anaconda run `pip install .` from the Anaconda Prompt (Anaconda3).  
This will make the commands `auv_nav`, `auv_cal` and `correct_images` available in the terminal. For more details refer to the documentation.


## Documentation
The documentation is hosted in [read the docs](https://oplab-pipeline.readthedocs.io).


## Citation
If you use this software, please cite the following article:

> Yamada, T, Prügel‐Bennett, A, Thornton, B. Learning features from georeferenced seafloor imagery with location guided autoencoders. J Field Robotics. 2020; 1– 16. https://doi.org/10.1002/rob.21961


## License
Copyright (c) 2020, University of Southampton. All rights reserved.

Licensed under the BSD 3-Clause License. 
See LICENSE.md file in the project root for full license information.  

## Contributing
Please document the code using [Numpy Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html).
If you are using VSCode, there is a useful extension that helps named [Python Docstring Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring). Once installed, make sure you select Numpy documentation in the settings.

Run `pre-commit install` to install [pre-commit](https://pre-commit.com/) into your git hooks. pre-commit will now run on every commit. If you don't have `pre-commit` installed, run `pip install pre-commit`.
