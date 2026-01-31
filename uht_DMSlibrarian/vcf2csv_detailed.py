#!/usr/bin/env python3
"""
Convert combined VCF to detailed CSV with mutations, consensus sequences, and analysis.
Uses streaming approach with progress bar for big datasets.

Supports multi-reference mode: when REFERENCE_ID is present in VCF, uses the correct
reference sequence for each cluster's AA translation.

Generates two output files:
1. Cluster-level CSV (one row per cluster):
   - name, reference_id, reference_length, consensus_protein, mutations, mutation_count,
   - dna_mutations, dna_variant_count, synonymous_count, hamming_distance,
   - premature_stop, stop_position, indelsyn, consensus_dna

2. Mutation-level CSV (one row per AA mutation, suffix: _mutations.csv):
   - cluster_name, reference_id, aa_position, ref_aa, alt_aa, mutation,
   - codon_position, ref_codon, alt_codon, mutation_type
"""

import os
import sys
import argparse
import csv
import time
from typing import Dict, List, Optional, Tuple, Set
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import defaultdict

# Import ReferenceManager for multi-reference support
try:
    from .reference_manager import ReferenceManager
except ImportError:
    ReferenceManager = None


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


def validate_vcf_file(vcf_file: str) -> Tuple[int, Set[str]]:
    """
    Validate VCF file format and extract reference IDs.

    Returns:
        Tuple of (variant_count, set of reference_ids found)

    Raises:
        ValidationError if file is invalid
    """
    if not os.path.exists(vcf_file):
        raise ValidationError(f"VCF file not found: {vcf_file}")

    if os.path.getsize(vcf_file) == 0:
        raise ValidationError(f"VCF file is empty: {vcf_file}")

    variant_count = 0
    reference_ids = set()
    has_header = False

    with open(vcf_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.startswith('##'):
                continue
            elif line.startswith('#CHROM'):
                has_header = True
                continue
            elif line.startswith('#'):
                continue
            else:
                # Data line
                parts = line.strip().split('\t')
                if len(parts) < 8:
                    raise ValidationError(
                        f"VCF line {line_num}: Expected at least 8 columns, got {len(parts)}"
                    )

                # Extract reference ID from INFO field
                info = parts[7]
                for item in info.split(';'):
                    if item.startswith('REFERENCE_ID='):
                        reference_ids.add(item.split('=')[1])

                variant_count += 1

    if not has_header:
        raise ValidationError("VCF file missing #CHROM header line")

    return variant_count, reference_ids


def validate_references(ref_sequences: Dict[str, str], vcf_reference_ids: Set[str]) -> List[str]:
    """
    Validate that VCF reference IDs exist in the reference file.

    Returns:
        List of warning messages (empty if all valid)
    """
    warnings = []
    available_refs = set(ref_sequences.keys())

    missing_refs = vcf_reference_ids - available_refs
    if missing_refs:
        warnings.append(
            f"WARNING: VCF contains {len(missing_refs)} reference ID(s) not in reference file: "
            f"{', '.join(sorted(missing_refs)[:5])}{'...' if len(missing_refs) > 5 else ''}"
        )

    unused_refs = available_refs - vcf_reference_ids
    if unused_refs and len(vcf_reference_ids) > 0:
        warnings.append(
            f"INFO: {len(unused_refs)} reference(s) in file not used by any cluster: "
            f"{', '.join(sorted(unused_refs)[:5])}{'...' if len(unused_refs) > 5 else ''}"
        )

    return warnings


def get_codon_at_position(dna_seq: str, aa_pos: int) -> str:
    """Get the codon at a given amino acid position (1-based)."""
    start = (aa_pos - 1) * 3
    if start + 3 <= len(dna_seq):
        return dna_seq[start:start + 3]
    return "XXX"

def translate_dna_to_protein(dna_seq):
    """Translate DNA sequence to protein sequence."""
    try:
        # Remove any non-standard characters and ensure length is multiple of 3
        clean_seq = ''.join(c for c in dna_seq.upper() if c in 'ATCG')
        if len(clean_seq) % 3 != 0:
            # Pad with N's to make it divisible by 3
            clean_seq += 'N' * (3 - len(clean_seq) % 3)
        
        # Translate
        protein_seq = str(Seq(clean_seq).translate())
        return protein_seq
    except Exception as e:
        print(f"Translation error: {e}")
        return ""

def get_amino_acid_position(dna_pos):
    """Convert DNA position to amino acid position (1-based)."""
    return (dna_pos - 1) // 3 + 1

def format_mutation_amino_acid(ref_aa, pos_aa, alt_aa):
    """Format mutation in amino acid notation (e.g., T241A)."""
    return f"{ref_aa}{pos_aa}{alt_aa}"

def has_premature_stop(protein_seq):
    """Check if protein sequence has premature stop codons."""
    # Look for stop codons (*) before the end
    stop_pos = protein_seq.find('*')
    if stop_pos != -1 and stop_pos < len(protein_seq) - 1:
        return True
    return False

def has_indels(dna_seq, reference_seq):
    """Check if sequence has insertions or deletions compared to reference."""
    return len(dna_seq) != len(reference_seq)

def calculate_hamming_distance(seq1, seq2):
    """Calculate Hamming distance between two sequences."""
    if len(seq1) != len(seq2):
        return -1  # Different lengths
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))

def parse_vcf_combined_streaming(vcf_file):
    """
    Parse the combined VCF file and group variants by cluster using streaming approach.

    Returns:
        Tuple of (cluster_variants dict, cluster_reference_ids dict)
        - cluster_variants: {cluster_name: [{'pos': int, 'ref': str, 'alt': str}, ...]}
        - cluster_reference_ids: {cluster_name: reference_id or None}
    """
    cluster_variants = defaultdict(list)
    cluster_reference_ids = {}  # cluster_name -> reference_id

    # First pass: count total lines for progress bar
    print("Counting variants for progress bar...")
    total_lines = 0
    with open(vcf_file, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                total_lines += 1

    print(f"Processing {total_lines:,} variants...")

    # Second pass: process variants with progress bar
    processed = 0
    start_time = time.time()

    with open(vcf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue

            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue

            chrom = parts[0]
            pos = int(parts[1])
            ref = parts[3]
            alt = parts[4]
            info = parts[7]

            # Extract cluster name, reference ID, and WT flag from INFO field
            cluster_name = None
            reference_id = None
            is_wt = False
            for info_item in info.split(';'):
                if info_item.startswith('CLUSTER='):
                    cluster_name = info_item.split('=')[1]
                elif info_item.startswith('REFERENCE_ID='):
                    reference_id = info_item.split('=')[1]
                elif info_item == 'WT=true':
                    is_wt = True

            if cluster_name:
                # Handle WT marker entries (position 0, no actual variants)
                if is_wt and pos == 0:
                    # Register WT cluster with empty variant list
                    if cluster_name not in cluster_variants:
                        cluster_variants[cluster_name] = []
                    if reference_id:
                        cluster_reference_ids[cluster_name] = reference_id
                else:
                    # Normal variant entry
                    cluster_variants[cluster_name].append({
                        'pos': pos,
                        'ref': ref,
                        'alt': alt
                    })
                    # Store reference ID for this cluster (should be same for all variants in cluster)
                    if reference_id and cluster_name not in cluster_reference_ids:
                        cluster_reference_ids[cluster_name] = reference_id

            # Progress reporting
            processed += 1
            if processed % 10000 == 0 or processed == total_lines:
                elapsed_time = time.time() - start_time
                rate = processed / elapsed_time if elapsed_time > 0 else 0
                progress_percent = (processed / total_lines) * 100

                if rate > 0:
                    remaining = total_lines - processed
                    eta_seconds = remaining / rate
                    eta_minutes = eta_seconds / 60
                    eta_hours = eta_minutes / 60

                    if eta_hours >= 1:
                        eta_str = f"{eta_hours:.1f}h"
                    elif eta_minutes >= 1:
                        eta_str = f"{eta_minutes:.1f}m"
                    else:
                        eta_str = f"{eta_seconds:.0f}s"
                else:
                    eta_str = "unknown"

                # Format elapsed time
                if elapsed_time >= 3600:
                    elapsed_str = f"{elapsed_time/3600:.1f}h"
                elif elapsed_time >= 60:
                    elapsed_str = f"{elapsed_time/60:.1f}m"
                else:
                    elapsed_str = f"{elapsed_time:.0f}s"

                # Progress bar
                bar_length = 50
                filled_length = int(bar_length * processed // total_lines)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)

                progress_line = (f"\rProgress: |{bar}| {progress_percent:.1f}% "
                               f"({processed:,}/{total_lines:,}) | "
                               f"Rate: {rate:.0f} variants/s | ETA: {eta_str} | "
                               f"Elapsed: {elapsed_str}")

                print(progress_line, end='', flush=True)

    # Count WT clusters (those with empty variant list)
    wt_count = sum(1 for variants in cluster_variants.values() if len(variants) == 0)
    mutant_count = len(cluster_variants) - wt_count

    print(f"\nParsed {len(cluster_variants):,} clusters ({mutant_count:,} with variants, {wt_count:,} WT)")

    # Report reference distribution if multi-reference
    unique_refs = set(cluster_reference_ids.values())
    if len(unique_refs) > 1:
        print(f"Found {len(unique_refs)} reference templates: {', '.join(sorted(unique_refs))}")
        ref_counts = defaultdict(int)
        for ref_id in cluster_reference_ids.values():
            ref_counts[ref_id] += 1
        for ref_id, count in sorted(ref_counts.items()):
            print(f"  {ref_id}: {count:,} clusters")

    return cluster_variants, cluster_reference_ids

def apply_variants_to_reference(reference_seq, variants):
    """Apply variants to reference sequence to create consensus."""
    # Convert reference to list for easy modification
    consensus = list(str(reference_seq))
    
    # Sort variants by position (descending) to avoid position shifts
    sorted_variants = sorted(variants, key=lambda x: x['pos'], reverse=True)
    
    for variant in sorted_variants:
        pos = variant['pos'] - 1  # Convert to 0-based indexing
        ref = variant['ref']
        alt = variant['alt']
        
        # Check if position is valid
        if pos < 0 or pos >= len(consensus):
            continue
        
        # Check if reference matches
        if consensus[pos] == ref:
            consensus[pos] = alt
        else:
            print(f"Warning: Reference mismatch at position {pos+1}: expected {ref}, found {consensus[pos]}")
    
    return ''.join(consensus)

def vcf_to_csv_detailed(vcf_file, reference_fasta_or_manager, output_csv,
                        generate_demultiplexed: bool = True):
    """
    Convert combined VCF to detailed CSV with mutations and analysis using streaming approach.

    Args:
        vcf_file: Path to combined VCF file
        reference_fasta_or_manager: Either path to reference FASTA (single ref mode)
                                    or ReferenceManager instance (multi-ref mode)
        output_csv: Path to output CSV file
        generate_demultiplexed: If True, also generate a demultiplexed mutation-level CSV

    Returns:
        True on success, False on failure
    """
    # === Input Validation ===
    try:
        print("Validating input files...")
        variant_count, vcf_ref_ids = validate_vcf_file(vcf_file)
        print(f"  VCF valid: {variant_count:,} variant entries")
        if vcf_ref_ids:
            print(f"  VCF references: {len(vcf_ref_ids)} unique ({', '.join(sorted(vcf_ref_ids)[:3])}{'...' if len(vcf_ref_ids) > 3 else ''})")
    except ValidationError as e:
        print(f"ERROR: {e}")
        return False

    # Determine reference mode
    is_multi_ref = (ReferenceManager is not None and
                   isinstance(reference_fasta_or_manager, ReferenceManager))

    if is_multi_ref:
        ref_manager = reference_fasta_or_manager
        print(ref_manager.get_reference_info())

        # Build lookup dicts for sequences and proteins
        ref_sequences: Dict[str, str] = {}
        ref_proteins: Dict[str, str] = {}
        for ref_id in ref_manager.get_reference_ids():
            ref_sequences[ref_id] = ref_manager.get_reference_sequence(ref_id)
            ref_proteins[ref_id] = ref_manager.get_reference_protein(ref_id)

        default_ref_id = ref_manager.get_default_reference_id()
    else:
        # Single reference mode (backward compatible)
        reference_fasta = reference_fasta_or_manager
        reference_records = list(SeqIO.parse(reference_fasta, "fasta"))
        if not reference_records:
            print(f"Error: No sequences found in reference file {reference_fasta}")
            return False

        reference_seq = str(reference_records[0].seq)
        default_ref_id = reference_records[0].id

        ref_sequences = {default_ref_id: reference_seq}
        ref_proteins = {default_ref_id: translate_dna_to_protein(reference_seq)}

        print(f"Reference sequence: {default_ref_id} (length: {len(reference_seq)})")
        print(f"Reference protein: {ref_proteins[default_ref_id][:50]}... (length: {len(ref_proteins[default_ref_id])})")

    # Validate reference consistency
    if vcf_ref_ids:
        warnings = validate_references(ref_sequences, vcf_ref_ids)
        for warning in warnings:
            print(f"  {warning}")

    # Parse VCF and group variants by cluster using streaming
    cluster_variants, cluster_reference_ids = parse_vcf_combined_streaming(vcf_file)

    # Process clusters with progress bar
    total_clusters = len(cluster_variants)
    print(f"\nProcessing {total_clusters:,} clusters...")

    processed = 0
    start_time = time.time()

    # Prepare CSV output with streaming
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['name', 'reference_id', 'reference_length', 'consensus_protein',
                      'mutations', 'mutation_count', 'dna_mutations', 'dna_variant_count',
                      'synonymous_count', 'hamming_distance', 'premature_stop', 'stop_position',
                      'indelsyn', 'consensus_dna']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Process each cluster
        for cluster_name, variants in cluster_variants.items():
            # Determine which reference to use for this cluster
            cluster_ref_id = cluster_reference_ids.get(cluster_name, default_ref_id)
            reference_seq = ref_sequences.get(cluster_ref_id, ref_sequences[default_ref_id])
            reference_protein = ref_proteins.get(cluster_ref_id, ref_proteins[default_ref_id])

            # Apply variants to reference
            consensus_dna = apply_variants_to_reference(reference_seq, variants)

            # Translate consensus to protein
            consensus_protein = translate_dna_to_protein(consensus_dna)

            # Format mutations in amino acid notation and track DNA mutations
            mutations_aa = []
            mutations_dna = []
            synonymous_count = 0

            for variant in variants:
                pos_dna = variant['pos']
                ref_nt = variant['ref']
                alt_nt = variant['alt']
                pos_aa = get_amino_acid_position(pos_dna)

                # Record DNA mutation
                mutations_dna.append(f"{ref_nt}{pos_dna}{alt_nt}")

                # Get the amino acid at this position in reference
                ref_aa = reference_protein[pos_aa - 1] if pos_aa <= len(reference_protein) else 'X'

                # Get the amino acid at this position in consensus
                alt_aa = consensus_protein[pos_aa - 1] if pos_aa <= len(consensus_protein) else 'X'

                if ref_aa != alt_aa:
                    mutation_str = format_mutation_amino_acid(ref_aa, pos_aa, alt_aa)
                    mutations_aa.append(mutation_str)
                else:
                    # DNA change but same AA = synonymous
                    synonymous_count += 1

            # Calculate metrics
            hamming_dist = calculate_hamming_distance(reference_protein, consensus_protein)
            premature_stop = "yes" if has_premature_stop(consensus_protein) else "no"
            indelsyn = "yes" if has_indels(consensus_dna, str(reference_seq)) else "no"

            # Detect premature stop position
            stop_pos = consensus_protein.find('*')
            if stop_pos != -1 and stop_pos < len(consensus_protein) - 1:
                stop_position = stop_pos + 1  # 1-based
            else:
                stop_position = ""

            # Write to CSV immediately (streaming)
            writer.writerow({
                'name': cluster_name,
                'reference_id': cluster_ref_id,
                'reference_length': len(reference_protein),
                'consensus_protein': consensus_protein,
                'mutations': '+'.join(mutations_aa) if mutations_aa else 'WT',
                'mutation_count': len(mutations_aa),
                'dna_mutations': '+'.join(mutations_dna) if mutations_dna else '',
                'dna_variant_count': len(mutations_dna),
                'synonymous_count': synonymous_count,
                'hamming_distance': hamming_dist,
                'premature_stop': premature_stop,
                'stop_position': stop_position,
                'indelsyn': indelsyn,
                'consensus_dna': consensus_dna
            })

            # Progress reporting
            processed += 1
            if processed % 1000 == 0 or processed == total_clusters:
                elapsed_time = time.time() - start_time
                rate = processed / elapsed_time if elapsed_time > 0 else 0
                progress_percent = (processed / total_clusters) * 100

                if rate > 0:
                    remaining = total_clusters - processed
                    eta_seconds = remaining / rate
                    eta_minutes = eta_seconds / 60
                    eta_hours = eta_minutes / 60

                    if eta_hours >= 1:
                        eta_str = f"{eta_hours:.1f}h"
                    elif eta_minutes >= 1:
                        eta_str = f"{eta_minutes:.1f}m"
                    else:
                        eta_str = f"{eta_seconds:.0f}s"
                else:
                    eta_str = "unknown"

                # Format elapsed time
                if elapsed_time >= 3600:
                    elapsed_str = f"{elapsed_time/3600:.1f}h"
                elif elapsed_time >= 60:
                    elapsed_str = f"{elapsed_time/60:.1f}m"
                else:
                    elapsed_str = f"{elapsed_time:.0f}s"

                # Progress bar
                bar_length = 50
                filled_length = int(bar_length * processed // total_clusters)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)

                progress_line = (f"\rProgress: |{bar}| {progress_percent:.1f}% "
                               f"({processed:,}/{total_clusters:,}) | "
                               f"Rate: {rate:.0f} clusters/s | ETA: {eta_str} | "
                               f"Elapsed: {elapsed_str}")

                print(progress_line, end='', flush=True)

    print(f"\nSuccessfully generated cluster-level CSV: {output_csv}")

    # Generate demultiplexed mutation-level CSV
    if generate_demultiplexed:
        # Create demultiplexed filename by inserting _mutations before .csv
        base, ext = os.path.splitext(output_csv)
        demux_output = f"{base}_mutations{ext}"
        print(f"\nGenerating mutation-level demultiplexed CSV...")
        vcf_to_csv_demultiplexed(vcf_file, reference_fasta_or_manager, demux_output)

    return True


def vcf_to_csv_demultiplexed(vcf_file, reference_fasta_or_manager, output_csv):
    """
    Convert combined VCF to demultiplexed CSV with one row per amino acid mutation.

    This format is useful for:
    - Counting mutation frequency at each position
    - Comparing mutation patterns across references
    - Statistical analysis of mutation distributions

    Args:
        vcf_file: Path to combined VCF file
        reference_fasta_or_manager: Either path to reference FASTA (single ref mode)
                                    or ReferenceManager instance (multi-ref mode)
        output_csv: Path to output CSV file

    Returns:
        True on success, False on failure
    """
    # Determine reference mode
    is_multi_ref = (ReferenceManager is not None and
                   isinstance(reference_fasta_or_manager, ReferenceManager))

    if is_multi_ref:
        ref_manager = reference_fasta_or_manager
        ref_sequences: Dict[str, str] = {}
        ref_proteins: Dict[str, str] = {}
        for ref_id in ref_manager.get_reference_ids():
            ref_sequences[ref_id] = ref_manager.get_reference_sequence(ref_id)
            ref_proteins[ref_id] = ref_manager.get_reference_protein(ref_id)
        default_ref_id = ref_manager.get_default_reference_id()
    else:
        reference_fasta = reference_fasta_or_manager
        reference_records = list(SeqIO.parse(reference_fasta, "fasta"))
        if not reference_records:
            print(f"Error: No sequences found in reference file {reference_fasta}")
            return False

        reference_seq = str(reference_records[0].seq)
        default_ref_id = reference_records[0].id
        ref_sequences = {default_ref_id: reference_seq}
        ref_proteins = {default_ref_id: translate_dna_to_protein(reference_seq)}

    # Parse VCF - reuse the streaming parser
    cluster_variants, cluster_reference_ids = parse_vcf_combined_streaming(vcf_file)

    total_clusters = len(cluster_variants)
    print(f"\nGenerating demultiplexed mutation table...")

    # Prepare CSV output
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = [
            'cluster_name', 'reference_id', 'reference_length',
            'aa_position', 'ref_aa', 'alt_aa', 'mutation',
            'dna_position', 'ref_nt', 'alt_nt', 'dna_mutation',
            'codon_position', 'ref_codon', 'alt_codon',
            'mutation_type', 'is_synonymous'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total_mutations = 0
        wt_clusters = 0

        for cluster_name, variants in cluster_variants.items():
            cluster_ref_id = cluster_reference_ids.get(cluster_name, default_ref_id)
            reference_seq = ref_sequences.get(cluster_ref_id, ref_sequences[default_ref_id])
            reference_protein = ref_proteins.get(cluster_ref_id, ref_proteins[default_ref_id])

            if not variants:
                # WT cluster - write a single row indicating no mutations
                wt_clusters += 1
                writer.writerow({
                    'cluster_name': cluster_name,
                    'reference_id': cluster_ref_id,
                    'reference_length': len(reference_protein),
                    'aa_position': '',
                    'ref_aa': '',
                    'alt_aa': '',
                    'mutation': 'WT',
                    'dna_position': '',
                    'ref_nt': '',
                    'alt_nt': '',
                    'dna_mutation': '',
                    'codon_position': '',
                    'ref_codon': '',
                    'alt_codon': '',
                    'mutation_type': 'WT',
                    'is_synonymous': ''
                })
                continue

            # Apply variants to get consensus
            consensus_dna = apply_variants_to_reference(reference_seq, variants)
            consensus_protein = translate_dna_to_protein(consensus_dna)

            # Track which AA positions we've already written (to handle multiple DNA changes in same codon)
            aa_positions_written = set()

            for variant in variants:
                pos_dna = variant['pos']
                ref_nt = variant['ref']
                alt_nt = variant['alt']
                pos_aa = get_amino_acid_position(pos_dna)

                # Get amino acids
                ref_aa = reference_protein[pos_aa - 1] if pos_aa <= len(reference_protein) else 'X'
                alt_aa = consensus_protein[pos_aa - 1] if pos_aa <= len(consensus_protein) else 'X'

                # Get codons
                ref_codon = get_codon_at_position(reference_seq, pos_aa)
                alt_codon = get_codon_at_position(consensus_dna, pos_aa)

                # Determine mutation type
                is_synonymous = (ref_aa == alt_aa)
                if ref_aa == alt_aa:
                    mutation_type = 'synonymous'
                elif alt_aa == '*':
                    mutation_type = 'nonsense'
                elif ref_aa == '*':
                    mutation_type = 'readthrough'
                else:
                    mutation_type = 'missense'

                # For non-synonymous mutations, only write once per AA position
                if not is_synonymous:
                    if pos_aa in aa_positions_written:
                        continue
                    aa_positions_written.add(pos_aa)

                mutation_str = f"{ref_aa}{pos_aa}{alt_aa}" if not is_synonymous else 'synonymous'
                dna_mutation_str = f"{ref_nt}{pos_dna}{alt_nt}"

                writer.writerow({
                    'cluster_name': cluster_name,
                    'reference_id': cluster_ref_id,
                    'reference_length': len(reference_protein),
                    'aa_position': pos_aa,
                    'ref_aa': ref_aa,
                    'alt_aa': alt_aa,
                    'mutation': mutation_str,
                    'dna_position': pos_dna,
                    'ref_nt': ref_nt,
                    'alt_nt': alt_nt,
                    'dna_mutation': dna_mutation_str,
                    'codon_position': ((pos_dna - 1) % 3) + 1,
                    'ref_codon': ref_codon,
                    'alt_codon': alt_codon,
                    'mutation_type': mutation_type,
                    'is_synonymous': 'yes' if is_synonymous else 'no'
                })
                total_mutations += 1

    print(f"Generated demultiplexed CSV: {output_csv}")
    print(f"  Total mutation rows: {total_mutations:,}")
    print(f"  WT clusters: {wt_clusters:,}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert combined VCF to detailed CSV with mutations and analysis')
    parser.add_argument('--input', required=True, help='Combined VCF file')
    parser.add_argument('--reference', required=True, help='Reference FASTA file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    
    args = parser.parse_args()
    
    # Check input files exist
    if not os.path.exists(args.input):
        print(f"Error: VCF file {args.input} not found")
        sys.exit(1)
    
    if not os.path.exists(args.reference):
        print(f"Error: Reference file {args.reference} not found")
        sys.exit(1)
    
    # Convert VCF to CSV
    success = vcf_to_csv_detailed(args.input, args.reference, args.output)
    
    if success:
        print(f"Successfully generated detailed CSV: {args.output}")
    else:
        print("Failed to generate CSV")
        sys.exit(1)

if __name__ == "__main__":
    main()
