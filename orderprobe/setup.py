#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="orderprobe",
    version="1.0.0",
    description="OrderProbe: Can LLMs recognize correct internal structure when input order is scrambled?",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Zhaolu-K/OrderProbe",
    packages=find_packages(where='.'),
    package_dir={'': '.'},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "torch>=1.9.0",
        "transformers>=4.21.0",
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.0.0",
        "openpyxl>=3.0.0",
        "tqdm>=4.62.0",
        "jieba>=0.42.1",
        "bert-score>=0.3.11",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "plotting": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
        "stats": [
            "scipy>=1.7.0",
        ],
    },
    keywords="orderprobe structural reconstruction llm evaluation chinese japanese korean idioms acl benchmark",
    project_urls={
        "Bug Reports": "https://github.com/Zhaolu-K/OrderProbe/issues",
        "Source": "https://github.com/Zhaolu-K/OrderProbe",
        "Documentation": "https://orderprobe.readthedocs.io/",
    },
)
