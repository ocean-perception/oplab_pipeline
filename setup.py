"""
Copyright (c) 2019, University of Southampton
All rights reserved.
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="auv_nav",
    version='0.0.3.0',
    author="Miquel Massot, Takaki Yamada, Subhra Das, Jin Lim & Blair Thornton",
    author_email="miquel.massot-campos@soton.ac.uk",
    description="Toolchain for auv dive processing",
    long_description=long_description,
    url="https://github.com/ocean-perception/auv_nav",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={  # Optional
        'console_scripts': [
            'auv_nav = auv_nav.auv_nav:main',
            'auv_cal = auv_cal.auv_cal:main',
            'debayer_folder = correct_images.debayer_folder:main',
            'correct_images = correct_images.draft_image_correct:main'
        ],
    },
    package_data={
        'auv_nav': ['default_yaml/*.yaml'],
        'auv_cal': ['default_yaml/*.yaml'],
        'correct_images': ['default_yaml/*.yaml']},
    install_requires=[
        'prettytable>=0.7.2',
        'PyYAML>=3.12',
        'argparse>=1.1',
        'colour_demosaicing>=0.1.4',
        'opencv-python>=4.1.0',
        'imageio>=2.5.0',
        'joblib>=0.13.2',
        'matplotlib>=3.1.0',
        'numpy>=1.16.4',
        'pandas>=0.23.4',
        'pillow>=5.1.0',
        'plotly>=3.6.1',
        'prettytable>=0.7.2',
        'pynmea2>=1.15.0',
        'pytz>=2018.9',
        'scipy>=1.2.1',
        'tqdm>=4.32.2'
    ]
)
