# uht-DMSlibrarian

## Interactive GUI
The package includes a  web browser-based GUI for running the complete pipeline workflow interactively. Launch it with:

```bash
umic-seq-pacbio gui
```

This opens a web interface in your browser with three tabs:
- **Dictionary**: Run the complete UMIC-seq PacBio (or ONT) pipeline from raw reads to variant analysis
- **NGS Count**: Count Illumina reads per variant via UMI matching
- **Fitness Analysis**: Calculate fitness from input/output pool comparisons

**Options:**
```bash
umic-seq-pacbio gui --host 0.0.0.0 --port 7860  # Accessible from other devices
umic-seq-pacbio gui --share  # Create a public share link (temporary)
```

## Complete Pipeline

For dictionary generation from long reads, use the complete pipeline entry-point that handles the entire workflow from UMI extraction to final variant analysis:

**Example with custom parameters:**
```bash
umic-seq-pacbio all \
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
  --max_workers 8 \
  --report pipeline_report.txt
```

**External dependencies:**
- `CD-HIT` - UMI clustering
- `Abpoa` - Consensus generation
- `minimap2` - Sequence alignment for variant calling
- `bcftools` - Variant calling (used internally)

Please install all of these and ensure they are in the PATH of your environment. Use e.g. conda to install these:
```bash
conda install -c bioconda cd-hit abpoa minimap2 bcftools
``` 

**Required Arguments:**
- `--input`: Input FASTQ file (can be .gz compressed)
- `--probe`: Probe FASTA file containing an approximately 50 bp sequence adjacent to the UMI
- `--reference`: Reference FASTA file containing 1 or more reference gene sequences. For multi-gene experiments, include multiple sequences in a single FASTA; the pipeline will automatically match each consensus to its best-matching reference with a tolerance of 5% sequence identity.
- `--output_dir`: Output directory where all results will be written

**Optional Arguments (with defaults):**

**UMI Extraction:**
- `--umi_len` (default: 52): Length of the UMI in base pairs
- `--umi_loc` (default: 'up'): Location of UMI relative to probe. Options: 'up' (upstream) or 'down' (downstream)
- `--min_probe_score` (default: 15): Minimum alignment score required for probe matching. For a 50bp probe, perfect match = 50. Lower values accept more mismatches.

**Clustering:**
- `--fast` (default: True): Use fast CD-HIT clustering (recommended)
- `--slow`: Use slow alignment-based clustering (alternative to --fast, legacy, no need to use)
- `--identity` (default: 0.90): Sequence identity threshold for fast clustering (0-1). 0.90 = 90% identity = allows up to 10% mismatch. For a 52bp UMI, this allows ~5 mismatches. 0.9 is sensible to start with for ONT data.
- `--aln_thresh` (default: 0.47, only used with --slow i.e. legacy): Alignment score threshold for slow clustering. Converts to integer score: 0.47 → 47. For a 52bp UMI with perfect match ≈ 104, threshold 47 ≈ 45% of perfect. **Note**: This is legacy and will be removed in future versions.
- **Orientation normalization**: When `--probe` is provided during clustering, sequences are automatically normalized to forward orientation based on probe alignment before being written to cluster files. This ensures all sequences in a cluster have the same orientation, improving consensus quality. If `--probe` is not provided, sequences are written as-is (backward compatible).

**Cluster Filtering:**
- `--size_thresh` (default: 10): Minimum number of long reads required per cluster. Clusters with fewer reads are discarded. Lower values = more sensitive (detects rare variants), higher values = more conservative (only high-confidence variants).

**Consensus Generation:**
- `--max_reads` (default: 20): Maximum number of reads per cluster used for consensus generation. Uses first N reads from each cluster. More reads = better consensus quality (this return of course diminishes) but slower. Fewer reads = faster.
- `--max_seq_len` (default: 15000): Maximum sequence length (bp) allowed for consensus input. Sequences longer than this are skipped to avoid memory spikes.

**Performance:**
- `--max_workers` (default: 4): Number of parallel workers for consensus generation and variant calling. Increase for faster processing if you have more CPU cores available.
- `--memory_monitor`: Enable memory monitoring during consensus generation and variant calling. When enabled, the pipeline will dynamically adjust worker count and batch sizes based on system memory pressure, pause when memory is critically high, and display memory usage in progress output. Requires `psutil` package (`pip install psutil`). Disabled by default.

**Reporting:**
- `--report` (optional): Path to output report file. If provided, generates a comprehensive summary report including execution time, input parameters, pipeline statistics (UMIs extracted, clusters generated, consensus sequences, variants called), and output file locations. Should be a .txt file, not a directory.


### Pipeline Steps

The `all` command runs the complete pipeline:

1. **UMI Extraction**: Extract UMIs from long reads using probe alignment
2. **Clustering**: Cluster similar UMIs using ultra-fast hash-based algorithm. Sequences are normalized to forward orientation (based on probe alignment) to ensure consistent orientation within clusters.
3. **Consensus Generation**: Generate consensus sequences using abpoa from consistently oriented sequences
4. **Variant Calling**: Call variants using minimap2 and bcftools
5. **Analysis**: Generate detailed CSV with mutation analysis

### Multi-Reference Support

The pipeline supports experiments with multiple gene variants (1-6 reference sequences) in a single analysis run. This is useful for:
- Multi-gene DMS experiments targeting several homologs
- Experiments with multiple protein domains
- Libraries containing distinct sequence variants

**How it works:**
1. Include multiple reference sequences in a single FASTA file (maximum 6 sequences)
2. During variant calling, each consensus sequence is aligned against all references using minimap2
3. The best-matching reference is automatically assigned to each consensus
4. Reference IDs are propagated through all downstream outputs

**Reference FASTA format:**
```
>gene_A
ATGCGATCGATCG...
>gene_B
ATGCGATCGATCG...
```

**Output integration:**
- `detailed_mutations.csv`: Contains `reference_id` column indicating which reference was used for each consensus
- `pool_haplotype_counts.csv`: Contains `REFERENCE_ID` column for each haplotype
- `merged_on_nonsyn_counts.csv`: Groups by (REFERENCE_ID, AA_MUTATIONS) to keep genes separate

For single-reference experiments, the pipeline works as before with no changes required.

### Individual Commands

You can also run individual steps. The `variants` and `analyze` commands fully support multi-reference FASTA files - each consensus will be automatically matched to its best-matching reference:

```bash
# Extract UMIs
umic-seq-pacbio extract \
  --input reads.fastq.gz \
  --probe probe.fasta \
  --umi_len 52 \
  --output ExtractedUMIs.fasta \
  --umi_loc up \
  --min_probe_score 15

# Cluster UMIs
umic-seq-pacbio cluster \
  --input_umi ExtractedUMIs.fasta \
  --input_reads reads.fastq.gz \
  --output_dir UMIclusterfull_fast \
  --probe probe.fasta \
  --identity 0.90 \
  --size_thresh 10

# Generate consensus sequences
umic-seq-pacbio consensus \
  --input_dir UMIclusterfull_fast \
  --output_dir consensus_results \
  --max_reads 20 \
  --max_seq_len 15000 \
  --max_workers 4

# Call variants
umic-seq-pacbio variants \
  --input_dir consensus_results \
  --reference reference.fasta \
  --output_dir variant_results \
  --combined_vcf combined_variants.vcf \
  --max_workers 4

# Analyze variants
umic-seq-pacbio analyze \
  --input_vcf combined_variants.vcf \
  --reference reference.fasta \
  --output final_results.csv
```

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
umic-seq-pacbio ngs_count \
  --pools_dir /path/to/NGS_data \
  --consensus_dir /path/to/consensus \
  --variants_dir /path/to/variants \
  --probe /path/to/probe.fasta \
  --reference /path/to/reference.fasta \
  --umi_len 52 \
  --umi_loc up \
  --left_ignore 22 \
  --right_ignore 24 \
  --pear_min_overlap 20 \
  --output /path/to/pool_variant_counts.csv
```

Inputs:
- `pools_dir`: directory containing one subfolder per pool; each subfolder has paired fastqs (`*_R1*.fastq.gz` and `*_R2*.fastq.gz`). You can have more than one pair in each subfolder. '*' indicates any valid character - the R1/R2 pairs are detected so e.g. L001_R1_001.fastq.gz and L001_R2_001.fastq.gz and L002_R1_001.fastq.gz and L002_R2_001.fastq.gz are a compatible 'pair of pairs' that will be easily detected. 
- `consensus_dir`: the consensus sequences directory (one FASTA per cluster)
- `variants_dir`: per-consensus VCFs generated by the variant calling step
- `probe`: probe FASTA (used only for logging; UMI extraction for Illumina uses your defined trimming rules)
- `reference`: reference FASTA for amino acid mapping (fasta should be bases)
- `--pear_min_overlap` (default: 20): Minimum overlap length for PEAR read merging. Lower values allow merging of reads with shorter overlaps.
- `--pear_yolo`: Maximally permissive PEAR settings. Disables the p-value statistical test (`-p 1.0`) and sets minimum overlap to 1 unless `--pear_min_overlap` is explicitly specified. Useful when reads have very short or variable overlaps.

Outputs:
- `pool_variant_counts.csv`: wide table, rows = VCF entries (CHROM, POS, REF, ALT), columns = pools
- `pool_haplotype_counts.csv`: rows = consensus haplotypes (cluster), columns = pools
  - Columns: `CONSENSUS`, `REFERENCE_ID`, `MUTATIONS` (nucleotide, position-sorted), `AA_MUTATIONS` (non-synonymous only, grouped by codon), plus per-pool count columns
  - Example AA format: `S45F+Y76P`; wild type is `WT`
- `merged_on_nonsyn_counts.csv`: haplotype counts merged by identical (REFERENCE_ID, AA_MUTATIONS) patterns
  - Columns: `REFERENCE_ID`, `AA_MUTATIONS`, `CONSENSUS_IDS`, `NUC_MUTATIONS`, plus per-pool count columns
  - Keeps genes separate when using multi-reference mode

Notes:
- For Illumina reads, UMIs are taken from the internal region of merged reads (default first 22 and last 24 bases ignored, change this for your UMIs).
- Amino acid numbering is 1-indexed; multiple nucleotide changes within a codon are combined into a single AA mutation.

### Fitness Analysis

The fitness analysis module processes `merged_on_nonsyn_counts.csv` to calculate fitness from input/output pool comparisons and generate comprehensive visualizations.

**Key Features:**
- Filters variants by minimum input count threshold
- Calculates relative frequencies (column normalization)
- Computes log fitness ratios: `log(rel_output / rel_input)` for each input/output pair
- Calculates average fitness across all pairs
- Bootstrap confidence intervals for fitness estimates (when multiple replicates available)
- Generates mutability, epistasis, fitness distribution, reproducibility, and substitution matrix plots

**Usage:**
```bash
umic-seq-pacbio fitness \
  --input merged_on_nonsyn_counts.csv \
  --output_dir fitness_results/ \
  --input_pools pool1 pool2 \
  --output_pools pool3 pool4 \
  --min_input 10 \
  --aa_filter 'S'
```

**Arguments:**
- `--input`: Path to `merged_on_nonsyn_counts.csv` (from ngs_count step)
- `--output_dir`: Directory to save plots and processed data
- `--input_pools`: Space-separated list of input pool names (e.g., `pool1 pool2`)
- `--output_pools`: Space-separated list of output pool names, paired with inputs (e.g., `pool3 pool4`)
- `--min_input` (default: 10): Minimum count threshold in input pools; variants below this in any input are filtered out
- `--aa_filter` (optional): Filter mutability plot to specific mutant amino acid (e.g., `S` for serine, `P` for proline, `*` for stop codons)
- `--group_by_reference` (optional): Generate separate plots and analysis per reference template. Requires `REFERENCE_ID` column in input (created automatically when using multi-reference mode)

**Outputs:**
- `fitness_analysis_results.csv`: Processed data with fitness calculations, relative frequencies, mutation annotations, and bootstrap confidence intervals (if multiple replicates)
- `mutability_plot.png`: Average fitness at each position for Hamming 1 (single) mutants, relative to WT
- `mutability_plot_{AA}.png`: Mutability plot filtered to specific amino acid (if `--aa_filter` used)
- `epistasis_plot.png`: Scatter plot of sum of single mutant fitnesses (x-axis) vs double mutant fitness (y-axis)
  - Only includes double mutants where both constituent singles are present in the dataset
  - Includes additivity line and correlation coefficient
- `fitness_distributions.png`: Overlaid KDE plots for single mutants:
  - Stop codons
  - Proline mutations
  - All other mutations
- `hamming_distributions.png`: Overlaid KDE plots for fitness distributions at Hamming distances 1-5
- `reproducibility_plot.png`: Pairwise comparison heatmaps of fitness across replicate pairs (only generated if ≥2 replicates)
  - Shows correlation between replicates with density heatmaps (blue→red color scheme)
- `substitution_matrix.png`: Heatmap showing average fitness for each amino acid substitution type (21×21 matrix including stop codons)
  - Amino acids arranged by similarity (hydrophobic, polar, charged, etc.)
  - Red = beneficial, Blue = deleterious
- `substitution_matrix.csv`: Full substitution matrix data for further analysis
- `ref_*/` (with `--group_by_reference`): Per-reference subdirectories containing reference-specific plots and `fitness_analysis_{ref_id}.csv`

**Example with multiple input/output pairs:**
```bash
umic-seq-pacbio fitness \
  --input merged_on_nonsyn_counts.csv \
  --output_dir fitness_results/ \
  --input_pools input_pool1 input_pool2 \
  --output_pools output_pool1 output_pool2 \
  --min_input 15
```

**Notes:**
- Fitness is calculated as `log(rel_output / rel_input)` where relative frequencies are column-normalized
- Average fitness is the mean across all input/output pairs
- Bootstrap confidence intervals (95% CI) are calculated using replicate-level resampling (1000 iterations) when multiple replicates are available
- Epistasis analysis requires both single and double mutants to be present in the dataset
- Mutability plots show average fitness aggregated across all single mutants at each position, relative to WT
- Reproducibility plot requires at least 2 replicate pairs
- When using `--group_by_reference`, all plots are generated twice: once for the combined dataset and once per reference template in subdirectories

### Threshold Selection Guide

**Quick reference:**
- **High-quality data**: For a 50 bp probe, use `--min_probe_score 30-40`
- **Q30+ data**: Use `--identity 0.85-0.90`
- **Rare variant detection**: Lower `--size_thresh` (e.g., 3)
- **High-confidence only**: Higher `--size_thresh` (e.g., 20)

### Output Files

The pipeline generates:
- `ExtractedUMIs.fasta`: Extracted UMI sequences
- `pipeline_report.txt`: Execution report (if `--report` specified) containing summary statistics, execution time, and parameters used
- `UMIclusterfull_fast/`: Cluster files (cluster_1.fasta, cluster_2.fasta, ...)
- `consensus_results/`: Consensus sequences per cluster
- `variant_results/`: Individual VCF files per cluster
- `combined_variants.vcf`: Combined variant calls
- `detailed_mutations.csv`: Detailed analysis with columns: `name`, `reference_id`, `consensus_sequence`, `mutations`, `hamming_distance`, `premature_stop`, `indelsyn`
- `pool_variant_counts.csv`: wide table, rows = VCF entries (CHROM, POS, REF, ALT), columns = pools
- `pool_haplotype_counts.csv`: rows = consensus haplotypes (cluster), columns = pools
  - Columns: `CONSENSUS`, `REFERENCE_ID`, `MUTATIONS` (nucleotide, position-sorted), `AA_MUTATIONS` (non-synonymous only, grouped by codon), plus per-pool count columns
  - Example AA format: `S45F+Y76P`; wild type is `WT`
- `merged_on_nonsyn_counts.csv`: haplotype counts merged by identical (REFERENCE_ID, AA_MUTATIONS) patterns
  - Columns: `REFERENCE_ID`, `AA_MUTATIONS`, `CONSENSUS_IDS`, `NUC_MUTATIONS`, plus per-pool count columns
- `fitness_analysis_results.csv`: Processed fitness data with log ratios, annotations, and bootstrap CIs (from fitness analysis step)
- `mutability_plot.png`: Average fitness by position for single mutants (from fitness analysis step)
- `epistasis_plot.png`: Epistasis analysis plot (from fitness analysis step)
- `fitness_distributions.png`: Fitness distributions by mutation type (from fitness analysis step)
- `hamming_distributions.png`: Fitness distributions by Hamming distance (from fitness analysis step)
- `reproducibility_plot.png`: Replicate comparison heatmaps (from fitness analysis step, if ≥2 replicates)
- `substitution_matrix.png`: Amino acid substitution fitness heatmap (from fitness analysis step)
- `substitution_matrix.csv`: Substitution matrix data (from fitness analysis step)

Note that this pipeline has been used for both PacBio and ONT data.

OS requirements: Unix (MacOS or Linux)

Estimated wallclock runtime benchmarks:
- Generates a dictionary and UMI-gene counts in ~2h on an Apple M2 for library size 200k unique variants
- Integrates approx 25 M illumina reads to the same dictionary in around 1 h
- Fitness analysis runs in seconds
