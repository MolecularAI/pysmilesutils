from setuptools import setup, find_packages

setup(
    name="PySMILESutils",
    version="1.0.0",
    description="Utilities for working with SMILES based encodings of molecules for deep learning (PyTorch oriented). ",
    author="Molecular AI group",
    author_email="esben.bjerrum@astrazeneca.com",
    license="Apache 2.0",
    packages=find_packages(exclude=("tests",)),
    url="https://github.com/MolecularAI/pysmilesutils",
)
