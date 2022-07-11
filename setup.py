import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kerasswint",
    version="0.0.1",
    author="Jussi Leinonen",
    author_email="jussi.leinonen@meteoswiss.ch",
    description="Keras implementations of Swin Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/meteoswiss-mdr/coalition-4-dl",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
