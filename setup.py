from setuptools import find_packages, setup

# Version Number:
version = '0.0.1'

setup(
    name="dss",
    version=version,
    url="",
    description="Diffusion structure search",
    install_requires=[
        "numpy",
        "pyyaml",
        "torch",
        "tensorboard",
        "schnetpack",
        "matplotlib",
        "pandas",
        "scipy",
        "seaborn",
        "wandb",
        "mace-torch",
        "torch-dftd",
        "ase>=3.21,<3.26",
    ],
    python_requires=">=3.5",
    packages=find_packages(),
)
