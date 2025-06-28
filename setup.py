from setuptools import setup, find_packages

setup(
    name="weather_prediction",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'pandas',
        'numpy',
        'matplotlib',
    ],
) 