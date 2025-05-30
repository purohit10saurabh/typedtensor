from setuptools import find_packages, setup

setup(
    name="typedtensor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "beartype", 
        "jaxtyping",
        "einops"
    ],
    python_requires=">=3.8"
) 