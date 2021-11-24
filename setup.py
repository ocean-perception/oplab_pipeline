# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import os
import os.path
import subprocess

from setuptools import find_packages, setup

classifiers = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Topic :: Scientific/Engineering",
    "Topic :: Software Development",
]


def return_version():
    """Append annotation to version string to indicate development versions.

    An empty (modulo comments and blank lines) commit_hash.txt is used
    to indicate a release, in which case nothing is appended to version
    string as defined above.
    """
    version = ""
    path_to_hashfile = os.path.join(os.path.dirname(__file__), "commit_hash.txt")
    if os.path.exists(path_to_hashfile):
        commit_version = ""
        with open(path_to_hashfile, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0 or line[0] == "#":
                    # Ignore blank lines and comments, the latter being
                    # any line that begins with #.
                    continue

                # First non-blank line is assumed to be the commit hash
                commit_version = line
                break

        if len(commit_version) > 0:
            version = commit_version
    else:
        version += ".dev0+unknown.commit"
    return version


def git_command(args):
    prefix = ["git"]
    return subprocess.check_output(prefix + args).decode().strip()


def git_pep440_version():
    # Is this called from Github Actions?
    if "GITHUB_REF_NAME" in os.environ:
        return os.environ["GITHUB_REF_NAME"]
    # Is Git installed?
    try:
        subprocess.call(["git", "--version"], stdout=subprocess.PIPE)
    except OSError:
        return None
    version_full = git_command(["describe", "--tags", "--dirty=.dirty"])
    version_tag = git_command(["describe", "--tags", "--abbrev=0"])
    version_tail = version_full[len(version_tag) :]  # noqa
    return version_tag + version_tail.replace("-", ".dev", 1).replace("-", "+", 1)


def run_setup():
    """Get version from git, then install."""
    # load long description from README.md
    readme_file = "README.md"
    if os.path.exists(readme_file):
        long_description = open(readme_file, encoding="utf-8", errors="ignore").read()
    else:
        print("Could not find readme file to extract long_description.")
        long_description = ""
    # If .git directory is present, create commit_hash.txt accordingly
    # to indicate version information
    if os.path.exists(".git"):
        # Provide commit hash or empty file to indicate release
        sha1 = git_pep440_version()
        if sha1 is None:
            sha1 = "unknown-commit"
        elif sha1 == "release":
            sha1 = ""
        commit_hash_header = (
            "# DO NOT EDIT!  "
            "This file was automatically generated by setup.py of oplab_pipeline"  # noqa
        )
        with open("commit_hash.txt", "w") as f:
            f.write(commit_hash_header + "\n")
            f.write(sha1 + "\n")
    oplab_pipeline_version = return_version()
    setup(
        name="oplab_pipeline",
        version=oplab_pipeline_version,
        author="Ocean Perception - University of Southampton",
        author_email="miquel.massot-campos@soton.ac.uk",
        description="Toolchain for AUV dive processing, camera calibration and image correction",  # noqa
        long_description=long_description,
        url="https://github.com/ocean-perception/oplab_pipeline",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        classifiers=classifiers,
        license="BSD",
        entry_points={  # Optional
            "console_scripts": [
                "auv_nav = auv_nav.auv_nav:main",
                "auv_cal = auv_cal.auv_cal:main",
                "correct_images = correct_images.correct_images:main",
            ],
        },
        scripts=[
            "src/scripts/debayer_folder.py",
            "src/scripts/extract_rosbag_images.py",
            "src/scripts/merge_dataset_csv.py",
            "src/scripts/pixel_stats_folder.py",
            "src/scripts/auv_cd.sh",
        ],
        include_package_data=True,
        package_data={
            "": [
                "default_yaml/*/*/*.yaml",
                "default_yaml/*/*.yaml",
                "default_yaml/*.yaml",
                "commit_hash.txt",
            ]
        },
        install_requires=[
            "argcomplete>=1.12.3",
            "argparse>=1.1",
            "colour_demosaicing>=0.1.5",
            "geographiclib>=1.50",
            "imageio>=2.6.1",
            "joblib>=0.14.1",
            "matplotlib>=3.2.1",
            "numba>=0.51.2",
            "numpy>=1.17.3",
            "opencv-python>=4.1.2",
            "pandas>=0.25.3",
            "pillow>=7.2.0",
            "plotly>=4.7.1",
            "plyfile>=0.7.2",
            "prettytable>=0.7.2",
            "psutil>=5.8.0",
            "pynmea2>=1.15.0",
            "pytz>=2019.3",
            "PyYAML>=3.12",
            "scikit_image>=0.17",
            "scipy>=1.4.1",
            "tqdm>=4.40.2",
            "wheel>=0.30.0",
        ],
    )


if __name__ == "__main__":
    run_setup()
