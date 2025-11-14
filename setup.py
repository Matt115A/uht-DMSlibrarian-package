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
    description="Extension to the UMIC-seq PacBio Pipeline - Complete pipeline for processing PacBio data from raw FASTQ to detailed mutation analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Matt Penner",
    author_email="mp957@cam.ac.uk",
    url="https://github.com/yourusername/uht-DMSlibrarian",  # Update with actual URL if available
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "biopython==1.74",
        "scikit-bio==0.5.5",
        "numpy==1.17.2",
        "pandas==0.25.1",
        "matplotlib==3.1.1",
        "seaborn==0.9.0",
        "scipy==1.3.1",
        "scikit-allel==1.2.1",
        "tqdm==4.54.1",
        "psutil==5.6.3",
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
        "License :: OSI Approved :: MIT License",  # Update if different
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="bioinformatics pacbio umi sequencing variant calling",
)

