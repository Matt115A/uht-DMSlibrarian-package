# UMIC-seq-PacBio


## Performance Optimizations (2025)

## UMIC-seq-pacbio: Complete Pipeline

For PacBio data analysis, use the complete pipeline entry-point that handles the entire workflow from UMI extraction to final variant analysis:

```bash
python UMIC-seq-pacbio.py all \
  --input reads.fastq.gz \
  --probe probe.fasta \
  --reference reference.fasta \
  --output_dir /path/to/output
```

**Required Arguments:**
- `--input`: Input FASTQ file (can be .gz compressed)
- `--probe`: Probe FASTA file containing an approximately 50 bp sequence adjacent to the UMI
- `--reference`: Reference FASTA file containing the reference gene sequence
- `--output_dir`: Output directory where all results will be written

**Optional Arguments (with defaults):**

**UMI Extraction:**
- `--umi_len` (default: 52): Length of the UMI in base pairs
- `--umi_loc` (default: 'up'): Location of UMI relative to probe. Options: 'up' (upstream) or 'down' (downstream)
- `--min_probe_score` (default: 15): Minimum alignment score required for probe matching. For a 50bp probe, perfect match = 50. Lower values accept more mismatches.

**Clustering:**
- `--fast` (default: True): Use fast CD-HIT clustering (recommended)
- `--slow`: Use slow alignment-based clustering (alternative to --fast, legacy)
- `--identity` (default: 0.90): Sequence identity threshold for fast clustering (0-1). 0.90 = 90% identity = allows up to 10% mismatch. For a 52bp UMI, this allows ~5 mismatches.
- `--aln_thresh` (default: 0.47): Alignment score threshold for slow clustering (only used with --slow). Converts to integer score: 0.47 → 47. For a 52bp UMI with perfect match ≈ 104, threshold 47 ≈ 45% of perfect. **Note**: This is legacy and will be removed in future versions.

**Cluster Filtering:**
- `--size_thresh` (default: 10): Minimum number of PacBio reads required per cluster. Clusters with fewer reads are discarded. Lower values = more sensitive (detects rare variants), higher values = more conservative (only high-confidence variants).

**Consensus Generation:**
- `--max_reads` (default: 20): Maximum number of reads per cluster used for consensus generation. Uses first N reads from each cluster. More reads = better consensus quality but slower. Fewer reads = faster.

**Performance:**
- `--max_workers` (default: 4): Number of parallel workers for consensus generation and variant calling. Increase for faster processing if you have more CPU cores available.

**Example with custom parameters:**
```bash
python UMIC-seq-pacbio.py all \
  --input reads.fastq.gz \
  --probe probe.fasta \
  --reference reference.fasta \
  --output_dir /path/to/output \
  --umi_len 52 \
  --umi_loc up \
  --min_probe_score 30 \
  --identity 0.95 \
  --size_thresh 15 \
  --max_reads 30 \
  --max_workers 8
```


### Pipeline Steps

The `all` command runs the complete pipeline:

1. **UMI Extraction**: Extract UMIs from PacBio reads
2. **Clustering**: Cluster similar UMIs using ultra-fast hash-based algorithm
3. **Consensus Generation**: Generate consensus sequences using abpoa
4. **Variant Calling**: Call variants using minimap2 and bcftools
5. **Analysis**: Generate detailed CSV with mutation analysis

### Individual Commands

You can also run individual steps:

```bash
# Extract UMIs
python UMIC-seq-pacbio.py extract \
  --input reads.fastq.gz \
  --probe probe.fasta \
  --umi_len 52 \
  --output ExtractedUMIs.fasta \
  --umi_loc up \
  --min_probe_score 15

# Cluster UMIs
python UMIC-seq-pacbio.py cluster \
  --input_umi ExtractedUMIs.fasta \
  --input_reads reads.fastq.gz \
  --output_dir UMIclusterfull_fast \
  --aln_thresh 0.47 \
  --size_thresh 10

# Generate consensus sequences
python UMIC-seq-pacbio.py consensus \
  --input_dir UMIclusterfull_fast \
  --output_dir consensus_results \
  --max_reads 20 \
  --max_workers 4

# Call variants
python UMIC-seq-pacbio.py variants \
  --input_dir consensus_results \
  --reference reference.fasta \
  --output_dir variant_results \
  --combined_vcf combined_variants.vcf \
  --max_workers 4

# Analyze variants
python UMIC-seq-pacbio.py analyze \
  --input_vcf combined_variants.vcf \
  --reference reference.fasta \
  --output final_results.csv

### NGS Pool Counting (Illumina) with UMI matching and haplotypes

This repository includes an NGS pooling/counting module to match Illumina paired-end reads back to consensus haplotypes and count per-variant and per-haplotype occurrences per pool.

Key features:
- PEAR-based merging of R1/R2 per pool (fallback to on-the-fly merge)
- UMI extraction from assembled reads by taking the internal window (ignores first 22 and last 24 bases by default, these numbers should be configured for your reads)
- Circular, strand-aware UMI-to-consensus matching
- Per-variant counts (rows = VCF entries; columns = pools)
- Per-haplotype counts that preserve multi-mutations with amino acid mutations (non-synonymous only)
- Deduplicated by non-synonamous amino acid mutational identity

Requirements:
- PEAR installed and available in PATH (e.g., `conda install -c bioconda pear`)

Usage:
```bash
python UMIC-seq-pacbio.py ngs_count \
  --pools_dir /path/to/NGS_data \
  --consensus_dir /path/to/consensus \
  --variants_dir /path/to/variants \
  --probe /path/to/probe.fasta \
  --reference /path/to/reference.fasta \
  --umi_len 52 \
  --umi_loc up \
  --left_ignore 22 \
  --right_ignore 24 \
  --output /path/to/pool_variant_counts.csv
```

Inputs:
- `pools_dir`: directory containing one subfolder per pool; each subfolder has paired fastqs (`*_R1*.fastq.gz` and `*_R2*.fastq.gz`)
- `consensus_dir`: the consensus sequences directory (one FASTA per cluster)
- `variants_dir`: per-consensus VCFs generated by the variant calling step
- `probe`: probe FASTA (used only for logging; UMI extraction for Illumina uses your defined trimming rules)
- `reference`: reference FASTA for amino acid mapping

Outputs:
- `pool_variant_counts.csv`: wide table, rows = VCF entries (CHROM, POS, REF, ALT), columns = pools
- `pool_haplotype_counts.csv`: rows = consensus haplotypes (cluster), columns = pools
  - Columns: `CONSENSUS`, `MUTATIONS` (nucleotide, position-sorted), `AA_MUTATIONS` (non-synonymous only, grouped by codon)
  - Example AA format: `S45F+Y76P`; wild type is `WT`
- `merged_on_nonsyn_counts.csv`: haplotype counts merged by identical non-synonymous amino acid patterns; includes the contributing consensus IDs, distinct nucleotide mutation strings, and per-pool totals

Notes:
- For Illumina reads, UMIs are taken from the internal region of merged reads (default first 22 and last 24 bases ignored); the probe is not searched in Illumina.
- Amino acid numbering is 1-indexed; multiple nucleotide changes within a codon are combined into a single AA mutation.
```

### Threshold Selection Guide

**Quick reference:**
- **High-quality data**: Use `--min_probe_score 30-40`, `--identity 0.90-0.95`, `--size_thresh 10-20`
- **Lower-quality data**: Use `--min_probe_score 15-20`, `--identity 0.85-0.90`, `--size_thresh 5-10`
- **Rare variant detection**: Lower `--size_thresh` (e.g., 3)
- **High-confidence only**: Higher `--size_thresh` (e.g., 20)

### Output Files

The pipeline generates:
- `ExtractedUMIs.fasta`: Extracted UMI sequences
- `UMIclusterfull_fast/`: Cluster files (cluster_1.fasta, cluster_2.fasta, ...)
- `consensus_results/`: Consensus sequences per cluster
- `variant_results/`: Individual VCF files per cluster
- `combined_variants.vcf`: Combined variant calls
- `final_results.csv`: Detailed analysis with amino acid mutations, Hamming distance, stop codons, and indels
- `pool_variant_counts.csv`: wide table, rows = VCF entries (CHROM, POS, REF, ALT), columns = pools
- `pool_haplotype_counts.csv`: rows = consensus haplotypes (cluster), columns = pools
  - Columns: `CONSENSUS`, `MUTATIONS` (nucleotide, position-sorted), `AA_MUTATIONS` (non-synonymous only, grouped by codon)
  - Example AA format: `S45F+Y76P`; wild type is `WT`
- `merged_on_nonsyn_counts.csv`: haplotype counts merged by identical non-synonymous amino acid patterns; includes the contributing consensus IDs, distinct nucleotide mutation strings, and per-pool totals

Note that this pipeline has been used for both PacBio and ONT data.

OS requirements: Unix (MacOS or Linux)

Estimated wallclock runtime benchmarks:
- Generates a dictionary and UMI-gene counts in ~2h on an Apple M2 for library size 200k unique variants