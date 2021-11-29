# Docker and Singularity

This are the steps to create a Singularity Image File (SIF) for oplab_pipeline.

## 1. Build the docker image. 
This assumes that you have installed Docker previously. If you haven't, follow [these installation instructions for Docker](https://docs.docker.com/get-docker/). At the root of oplab_pipeline, run

```
docker build --tag oceanperception/oplab_pipeline .
```

Check that your image works correcty by running `auv_nav` help using the following command:

```sh
docker run -it oceanperception/oplab_pipeline auv_nav -h
```

this command should return the following output (as per version 0.1.10):

```
     ● ●  Ocean Perception
     ● ▲  University of Southampton

 Copyright (C) 2020 University of Southampton
 This program comes with ABSOLUTELY NO WARRANTY.
 This is free software, and you are welcome to
 redistribute it.

INFO ▸ Running auv_nav version unknown-commit
usage: auv_nav [-h] {parse,process,convert} ...

positional arguments:
  {parse,process,convert}
    parse               Parse raw data and converting it to an intermediate dataformat for further processing. Type
                        auv_nav parse -h for help on this target.
    process             Process and/or convert data. Data needs to be saved in the intermediate data format generated
                        using auv_nav.py parse. Type auv_nav process -h for help on this target.
    convert             Converts data.

optional arguments:
  -h, --help            show this help message and exit
```

# 2. Convert the docker image to SIF. 
This assumes you have installed Singularity previously. If you havent, follow [these installation instructions for Singularity.](https://sylabs.io/guides/master/user-guide/quick_start.html#quick-installation-steps)

```sh
singularity build oplab_pipeline.sif docker-daemon://oceanperception/oplab_pipeline:latest
```

# 3. Run the SIF image.
The previous instruction will have saved a file `oplab_pipeline.sif` in this same directory. This is your containerised image file. You can use this SIF file in any Linux computer with Singularity installed. To use it, run

```sh
singularity run oplab_pipeline.sif auv_nav -h
```