from setuptools import setup, find_packages

setup(
    name="PupilToolkit",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy", "pandas", "scipy", "pyyaml", "tqdm",
        "XdetectionCore" # Essential for Session objects
    ],
)