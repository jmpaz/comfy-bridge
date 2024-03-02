from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="comfy-bridge",
    version="0.0.1",
    author="jmpaz",
    packages=find_packages(),
    install_requires=required,
)
