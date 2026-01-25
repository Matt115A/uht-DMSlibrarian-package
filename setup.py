"""
Setup configuration for uht-DMSlibrarian package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Version
__version__ = "0.2.0"

setup(
    name="uht-dmslibrarian",
    version=__version__,
    description="Extension of the UMIC-seq Pipeline - Complete pipeline for dictionary building and NGS count inetgration, with fitness calculations, error modelling and mutation analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Matt Penner",
    author_email="mp957@cam.ac.uk",
    url="https://github.com/Matt115A/uht-DMSlibrarian-package",  
    packages=find_packages(),
    python_requires=">=3.10,<3.11",
    install_requires=[
        "biopython==1.86",
        "scikit-bio==0.7.0",
        "numpy==2.2.1",
        "pandas==2.3.3",
        "matplotlib==3.10.7",
        "seaborn==0.13.2",
        "scipy==1.15.3",
        "scikit-allel==1.3.13",
        "tqdm==4.67.1",
        "psutil==7.1.3",
        "gradio>=4.0.0",
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
        "Programming Language :: Python :: 3.10",
    ],
    keywords="bioinformatics pacbio umi sequencing variant calling",
)
