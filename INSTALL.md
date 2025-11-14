# Installation Instructions for uht-DMSlibrarian

## Installing via pip

### From source:

1. Clone or download this repository
2. Navigate to the repository directory
3. Install using pip:

```bash
pip install .
```

Or install in editable mode (for development):

```bash
pip install -e .
```

### From a wheel file:

If you have a built wheel file:

```bash
pip install uht-DMSlibrarian-0.1.0-py3-none-any.whl
```

## Dependencies

All dependencies are automatically installed when installing the package. The package requires:

- Python 3.7 or higher
- See `requirements.txt` for full list of Python dependencies

### External Tools

The following external tools are required and must be installed separately:

1. **cd-hit-est** - For fast UMI clustering
   - Install via conda: `conda install -c bioconda cd-hit`
   - Or from source: http://weizhong-lab.ucsd.edu/cd-hit/

2. **abpoa** - For consensus sequence generation
   - Install via conda: `conda install -c bioconda abpoa`
   - Or from source: https://github.com/yangao07/abPOA

3. **minimap2** - For sequence alignment
   - Install via conda: `conda install -c bioconda minimap2`
   - Or from source: https://github.com/lh3/minimap2

4. **PEAR** (optional, for NGS pool counting) - For merging paired-end reads
   - Install via conda: `conda install -c bioconda pear`

## Usage

After installation, you can use the package via the command-line:

```bash
umic-seq-pacbio --help
```

Example usage:

```bash
umic-seq-pacbio all \
  --input reads.fastq.gz \
  --probe probe.fasta \
  --reference reference.fasta \
  --output_dir /path/to/output
```

## Verification

To verify installation, run:

```bash
umic-seq-pacbio --help
```

This should display the help message if installation was successful.

