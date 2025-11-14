"""
Setup configuration for uht-DMSlibrarian package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Version
__version__ = "0.1.0"

setup(
    name="uht-DMSlibrarian",
    version=__version__,
    description="Extension of the UMIC-seq Pipeline - Complete pipeline for dictionary building and NGS count inetgration, with fitness calculations, error modelling and mutation analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Matt Penner",
    author_email="mp957@cam.ac.uk",
    url="https://github.com/Matt115A/uht-DMSlibrarian-package",  # Update with actual URL if available
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "biopython>=1.79,<2.0",
        "scikit-bio>=0.5.5,<0.8",
        "numpy>=1.21.0,<2.0",
        "pandas>=1.3.0,<3.0",
        "matplotlib>=3.5.0,<4.0",
        "seaborn>=0.11.0,<1.0",
        "scipy>=1.7.0,<2.0",
        "scikit-allel>=1.2.1,<2.0",
        "tqdm>=4.62.0",
        "psutil>=5.8.0",
    ],
    entry_points={
        "console_scripts": [
            "umic-seq-pacbio=uht_DMSlibrarian.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="bioinformatics pacbio umi sequencing variant calling",
)

