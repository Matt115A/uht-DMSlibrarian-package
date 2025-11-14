"""
uht-DMSlibrarian: UMIC-seq PacBio Pipeline
A complete pipeline for processing PacBio data from raw FASTQ to detailed mutation analysis.

Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Paul Jannis Zurek, pjz26@cam.ac.uk"

# Note: We don't import modules here that have __main__ blocks
# to avoid execution when the package is imported
__all__ = [
    'UMIC_seq',
    'UMIC_seq_helper',
    'simple_consensus_pipeline',
    'sensitive_variant_pipeline',
    'vcf2csv_detailed',
    'ngs_count',
    'fitness_analysis',
]

