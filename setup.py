from setuptools import setup

setup(
    name="libesmart",
    version="0.1",
    description="A builder for eSmart",
    url="",
    author="eSmart Systems",
    packages=["esmart"],
    install_requires=[
        "numpy==1.23.1",
        "tensorflow==2.8.0",
        "tensorflow-addons==0.16.1",
        "pyyaml",
        "pandas",
        "path",
        "matplotlib",

        "torch==1.12.0",
        "ax-platform==0.1.19", "botorch==0.4.0", "gpytorch==1.4.2",
        "sqlalchemy",
        "torchviz",
        "argparse",
    ],
    python_requires=">=3.8.10", # You need Python 3.8 or later to run Ax.
    zip_safe=False,
    entry_points={"console_scripts": ["esmart = esmart.cli:main",],},
)