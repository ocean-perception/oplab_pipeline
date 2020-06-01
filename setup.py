# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License. 
See LICENSE.md file in the project root for full license information.  
"""

import importlib
import os
from setuptools import setup
from setuptools import find_packages
import subprocess
from oplab.console import Console


classifiers = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development']


def git_command(args):
    prefix = ['git']
    return subprocess.check_output(prefix + args).decode().strip()


def git_pep440_version():
    # Is Git installed?
    try:
        subprocess.call(['git', '--version'],
                        stdout=subprocess.PIPE)
    except OSError:
        return None
    version_full = git_command(['describe', '--tags', '--dirty=.dirty'])
    version_tag = git_command(['describe', '--tags', '--abbrev=0'])
    version_tail = version_full[len(version_tag):]
    return version_tag + version_tail.replace('-', '.dev', 1).replace('-', '+', 1)


def run_setup():
    """Get version from git, then install."""
    # load long description from README.md
    readme_file = 'README.md'
    if os.path.exists(readme_file):
        long_description = open(readme_file, encoding='utf-8', errors='ignore').read()
    else:
        print('Could not find readme file to extract long_description.')
        long_description = ''
    # If .git directory is present, create commit_hash.txt accordingly
    # to indicate version information
    if os.path.exists('.git'):
        # Provide commit hash or empty file to indicate release
        sha1 = git_pep440_version()
        if sha1 is None:
            sha1 = 'unknown-commit'
        elif sha1 == 'release':
            sha1 = ''
        commit_hash_header = (
            '# DO NOT EDIT!  '
            'This file was automatically generated by setup.py of oplab_pipeline')
        with open('oplab/commit_hash.txt', 'w') as f:
            f.write(commit_hash_header + '\n')
            f.write(sha1 + '\n')
    # Import oplab/version.py without importing oplab_pipeline
    version_specs = importlib.util.find_spec('oplab.version')
    version = importlib.util.module_from_spec(version_specs)
    version_specs.loader.exec_module(version)
    oplab_pipeline_version = version.version
    setup(
        name="oplab_pipeline",
        version=oplab_pipeline_version,
        author="Ocean Perception - University of Southampton",
        author_email="miquel.massot-campos@soton.ac.uk",
        description="Toolchain for AUV dive processing, camera calibration and image correction",
        long_description=long_description,
        url="https://github.com/ocean-perception/oplab_pipeline",
        bugtrack_url='http://github.com/ocean-perception/oplab_pipeline/issues',
        packages=find_packages(),
        classifiers=classifiers,
        license='BSD',
        entry_points={  # Optional
            'console_scripts': [
                'auv_nav = auv_nav.auv_nav:main',
                'auv_cal = auv_cal.auv_cal:main',
                'debayer_folder = correct_images.debayer_folder:main',
                'correct_images = correct_images.correct_images:main'
            ],
        },
        package_data={
            '': ['default_yaml/*.yaml'],
            'auv_nav': ['default_yaml/*/*/*.yaml'],
            'correct_images': ['default_yaml/*/*.yaml'],
            'oplab_pipeline_version': ['commit_hash.txt']},
        install_requires=[
            'prettytable>=0.7.2',
            'PyYAML>=3.12',
            'argparse>=1.1',
            'colour_demosaicing>=0.1.4',
            'opencv-python>=4.1.0',
            'imageio>=2.5.0',
            'joblib>=0.13.2',
            'matplotlib>=3.1.0',
            'numpy>=1.17.3',
            'pandas>=0.23.4',
            'pillow>=5.1.0',
            'plotly>=3.6.1',
            'plyfile>=0.7.2',
            'prettytable>=0.7.2',
            'pynmea2>=1.15.0',
            'pytz>=2018.9',
            'scipy>=1.2.1',
            'tqdm>=4.32.2'
        ]
    )

if __name__ == '__main__':
    run_setup()
