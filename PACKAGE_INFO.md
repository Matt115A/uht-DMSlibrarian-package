# Package Information for uht-DMSlibrarian v0.1.0

## Package Structure

```
uht-DMSlibrarian/
├── setup.py                  # Package setup configuration
├── pyproject.toml            # Modern Python packaging configuration
├── requirements.txt          # Frozen dependencies (pinned versions)
├── MANIFEST.in               # Files to include in package distribution
├── README.md                 # Original README
├── INSTALL.md                # Installation instructions
└── uht_DMSlibrarian/         # Main package directory
    ├── __init__.py           # Package initialization
    ├── cli.py                # Main command-line interface
    ├── UMIC_seq.py           # UMI extraction and clustering
    ├── UMIC_seq_helper.py    # Helper functions for demultiplexing
    ├── simple_consensus_pipeline.py    # Consensus generation
    ├── sensitive_variant_pipeline.py   # Variant calling
    ├── vcf2csv_detailed.py   # VCF to CSV conversion
    ├── ngs_count.py          # NGS pool counting
    └── fitness_analysis.py   # Fitness analysis
```

## Installation

### Standard Installation

```bash
pip install .
```

### Development Installation (Editable Mode)

```bash
pip install -e .
```

### Install from requirements.txt

```bash
pip install -r requirements.txt
```

## Usage

After installation, the main command-line tool is available as:

```bash
umic-seq-pacbio --help
```

### Example Usage

```bash
# Run complete pipeline
umic-seq-pacbio all \
  --input reads.fastq.gz \
  --probe probe.fasta \
  --reference reference.fasta \
  --output_dir /path/to/output

# Run individual steps
umic-seq-pacbio extract --input reads.fastq.gz --probe probe.fasta --output umis.fasta
umic-seq-pacbio cluster --input_umi umis.fasta --input_reads reads.fastq.gz --output_dir clusters/
umic-seq-pacbio consensus --input_dir clusters/ --output_dir consensus/
umic-seq-pacbio variants --input_dir consensus/ --reference reference.fasta --output_dir variants/
umic-seq-pacbio analyze --input_vcf combined.vcf --reference reference.fasta --output detailed.csv
```

## Dependencies

All dependencies are frozen at specific versions for reproducibility:

- biopython==1.74
- scikit-bio==0.5.5
- numpy==1.17.2
- pandas==0.25.1
- matplotlib==3.1.1
- seaborn==0.9.0
- scipy==1.3.1
- scikit-allel==1.2.1
- tqdm==4.54.1
- psutil==5.6.3

### External Tools Required

These tools must be installed separately (not Python packages):

1. **cd-hit-est** - For fast UMI clustering
   - Install: `conda install -c bioconda cd-hit`

2. **abpoa** - For consensus sequence generation
   - Install: `conda install -c bioconda abpoa`

3. **minimap2** - For sequence alignment
   - Install: `conda install -c bioconda minimap2`

4. **PEAR** (optional) - For merging paired-end reads in NGS pool counting
   - Install: `conda install -c bioconda pear`

## Building Distribution Packages

### Build source distribution and wheel:

```bash
pip install build
python -m build
```

This will create:
- `dist/uht-DMSlibrarian-0.1.0.tar.gz` (source distribution)
- `dist/uht_DMSlibrarian-0.1.0-py3-none-any.whl` (wheel)

### Install from built wheel:

```bash
pip install dist/uht_DMSlibrarian-0.1.0-py3-none-any.whl
```

## Version Information

- Package version: 0.1.0
- Python requirement: >=3.7
- Author: Paul Jannis Zurek (pjz26@cam.ac.uk)

## Notes

- The package name uses a hyphen (`uht-DMSlibrarian`) but the Python module name uses an underscore (`uht_DMSlibrarian`) due to Python naming conventions.
- All dependencies are frozen at specific versions from the original conda environment to ensure reproducibility.
- The command-line tool `umic-seq-pacbio` is installed as a console script entry point.

