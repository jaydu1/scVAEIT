from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="scVAEIT",
    description="scVAEIT is a Python module of Variational autoencoder for single-cell mosaic integration and transfer learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jaydu1/scVAEIT",
    project_urls={
        "Bug Tracker": "https://github.com/jaydu1/scVAEIT/issues",
        "Changelog": "https://github.com/jaydu1/scVAEIT/releases",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
    ],
    packages=find_packages(exclude=["*example*", "*Reproducibility*"]),
    python_requires=">=3.9",
    install_requires=[
        "scikit-learn",
        "matplotlib",
        "pandas",
        "jupyter",
        "numpy",
        "tensorflow >= 2.12, < 2.16",
        "tensorflow-probability >= 0.12, < 0.24",
        "scanpy >= 1.9.2",
    ]
)