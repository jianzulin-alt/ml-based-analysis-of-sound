from setuptools import find_packages, setup


setup(
    name="ml-based-analysis-of-sound",
    version="0.1.0",
    description="Machine learning-based analysis of music and sound",
    packages=find_packages(include=["src", "src.*"]),
)
