from setuptools import setup, find_packages

setup(
    name="nndc",
    version="0.1",
    description="Neighbourhood Retrieval with Distance Correlation",
    author="Surya Kant Sahu",
    author_email="surya.oju@pm.me",
    packages=find_packages(),
    install_requires=[line.strip() for line in open("requirements.txt").readlines()]
)