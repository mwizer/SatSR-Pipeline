from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='SATSR-PIPELINE',
    version='1.0.1',
    description='A pipeline for processing satellite images',
    author='Maciej Wizerkaniuk',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.10',
)