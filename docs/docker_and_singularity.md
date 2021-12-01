# Docker and Singularity

This are the steps to create a Singularity Image File (SIF) for oplab_pipeline.
There exists a Dockerfile as well, but it's recommended to use the SIF image
instead.

## 1. Build the Singularity image. 
This assumes you have installed Singularity previously. If you havent, follow [these installation instructions for Singularity.](https://sylabs.io/guides/master/user-guide/quick_start.html#quick-installation-steps)

```sh
sudo singularity build oplab_pipeline.sif oplab_pipeline.def
```

## 2. Run the SIF image.
The previous instruction will have saved a file `oplab_pipeline.sif` in this same directory. This is your containerised image file. You can use this SIF file in any Linux computer with Singularity installed. To see the help, run

```sh
singularity run oplab_pipeline.sif
```

Run the pipeline using the following command:
 - `singularity exec oplab_pipeline.sif auv_nav ARGS`
 - `singularity exec oplab_pipeline.sif auv_cal ARGS`
 - `singularity exec oplab_pipeline.sif correct_images ARGS`

## FAQ

If you see the following error:

```sh
/opt/oplab_pipeline_env/lib/python3.9/site-packages/joblib/_multiprocessing_helpers.py:45: UserWarning: [Errno 2] No such file or directory.  joblib will operate in serial mode
  warnings.warn('%s.  joblib will operate in serial mode' % (e,))
```

run the SIF image with the flag `-B /run/shm:/run/shm` Example:
- `singularity exec -B /run/shm:/run/shm oplab_pipeline.sif auv_nav ARGS`
- `singularity exec -B /run/shm:/run/shm oplab_pipeline.sif auv_cal ARGS`
- `singularity exec -B /run/shm:/run/shm oplab_pipeline.sif correct_images ARGS`