from setuptools import setup, find_packages

# Most config lives in pyproject.toml.
# This file exists for editable installs (pip install -e .) compatibility.
setup(
    name="llmanim",
    version="0.1.0",
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.8",
    install_requires=[
        "manim>=0.18.0",
        "numpy>=1.21.0",
    ],
    author="Rishabh Bhartiya",
    author_email="rishabh.bhartiya.in@gmail.com",
    description="Reusable Manim components for creating LLM and Transformer explanation videos.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rishabhbhartiya/LLManim",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Video",
    ],
)