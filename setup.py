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
        "torch==1.12.0",
        "tensorflow==2.9.1",
        "tensorflow-addons==0.17.1",
        "pyyaml",
        "pandas",
        "path",
        "ax-platform==0.2.5.1",
    ],
    python_requires=">=3.9.12",
    zip_safe=False,
    entry_points={"console_scripts": ["esmart = esmart.cli:main",],},
)