from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='SATSR-PIPELINE',
    version='1.2.0',
    description='A pipeline for processing satellite images',
    author='Dawid Kopeć, Dawid Krutul, Maciej Wizerkaniuk',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.10',
)