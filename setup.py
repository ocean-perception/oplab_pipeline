import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="auv_nav",
    version='0.0.1.5',
    author="Jin Lim, Blair Thornton",
    author_email="author@example.com",
    description="Parsers for navigation data for oplab and acfr formats",
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
        ],
    },
    package_data={'auv_nav': ['default_yaml/*.yaml']},
    install_requires=[
        'matplotlib>=2.1.2',
        'numpy>=1.14.5',
        'pandas>=0.22.0',
        'plotly>=2.5.1',
        'prettytable>=0.7.2',
        'PyYAML>=3.12'
    ]
)
