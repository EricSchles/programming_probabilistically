import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="programming_probabilistically",
    version="0.1",
    description="A set of reference implementations for machine learning",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/EricSchles/programming_probabilistically",
    author="Eric Schles",
    author_email="ericschles@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=[],
    include_package_data=True,
    install_requires=["pytest"],
)