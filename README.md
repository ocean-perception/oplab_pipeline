[![Build Status](https://travis-ci.com/ocean-perception/oplab_pipeline.svg?token=UkLpgZyKjs3prWpXePir&branch=master)](https://travis-ci.com/ocean-perception/oplab_pipeline) [![Code Coverage](https://codecov.io/gh/ocean-perception/oplab_pipeline/branch/master/graph/badge.svg?token=PJBfl6qhp5)](https://codecov.io/gh/ocean-perception/oplab_pipeline)[![Documentation Status](https://readthedocs.org/projects/oplab-pipeline/badge/?version=latest)](https://oplab-pipeline.readthedocs.io/en/latest/?badge=latest)


# oplab_pipeline

oplab_pipeline is a python toolchain to process AUV dives from raw data into navigation and imaging products. The software is capable of:

- Process navigation: fuses AUV or ROV sensor data using state of the art filters and geolocalises recorded imagery.
- Camera and laser calibration: performs automatic calibration pattern detection to calibrate monocular or stereo cameras. Also calibrates laser sheets with respect to the cameras.
- Image correction: performs pixel-wise image corrections to enhance colour and contrast in underwater images.

Please review the latest changes in the [CHANGELOG.md](CHANGELOG.md). 

## Documentation
The documentation is hosted in [read the docs](oplab-pipeline.readthedocs.io).

## Citation
If you use this software, please cite the following article:

> Yamada, T, Prügel‐Bennett, A, Thornton, B. Learning features from georeferenced seafloor imagery with location guided autoencoders. J Field Robotics. 2020; 1– 16. https://doi.org/10.1002/rob.21961


## License
Copyright (c) 2020, University of Southampton. All rights reserved.

Licensed under the BSD 3-Clause License. 
See LICENSE.md file in the project root for full license information.  

## Developers
Please document the code using [Numpy Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html).
If you are using VSCode, there is a useful extension that helps named [Python Docstring Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring). Once installed, make sure you select Numpy documentation in the settings.