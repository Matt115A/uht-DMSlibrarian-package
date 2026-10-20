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

# Same amino acid annotation as the fitness table's AA_MUTATIONS
from .ngs_count import annotate_variants, _indel_events


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


def validate_vcf_file(vcf_file: str) -> Tuple[int, Set[str], Set[str], int]:
    """
    Validate VCF file format and extract metadata.

    Returns:
        Tuple of (variant_count, reference_ids set, cluster_names set, entries_without_cluster)

    Raises:
        ValidationError if file is invalid
    """
    if not os.path.exists(vcf_file):
        raise ValidationError(f"VCF file not found: {vcf_file}")

    if os.path.getsize(vcf_file) == 0:
        raise ValidationError(f"VCF file is empty: {vcf_file}")

    variant_count = 0
    reference_ids = set()
    cluster_names = set()
    entries_without_cluster = 0
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

                # Extract reference ID and cluster name from INFO field
                info = parts[7]
                has_cluster = False
                for item in info.split(';'):
                    if item.startswith('REFERENCE_ID='):
                        reference_ids.add(item.split('=')[1])
                    elif item.startswith('CLUSTER='):
                        cluster_names.add(item.split('=')[1])
                        has_cluster = True

                if not has_cluster:
                    entries_without_cluster += 1

                variant_count += 1

    if not has_header:
        raise ValidationError("VCF file missing #CHROM header line")

    return variant_count, reference_ids, cluster_names, entries_without_cluster


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
    absent_clusters = set()     # consensuses that do not contain the gene

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
            is_absent = False
            for info_item in info.split(';'):
                if info_item.startswith('CLUSTER='):
                    cluster_name = info_item.split('=')[1]
                elif info_item.startswith('REFERENCE_ID='):
                    reference_id = info_item.split('=')[1]
                elif info_item == 'WT=true':
                    is_wt = True
                elif info_item == 'GENE_ABSENT=true':
                    is_absent = True

            if cluster_name:
                # Handle WT / gene-absent marker entries (position 0, no actual variants)
                if (is_wt or is_absent) and pos == 0:
                    if is_absent:
                        absent_clusters.add(cluster_name)
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
    wt_count = sum(1 for name, variants in cluster_variants.items()
                   if len(variants) == 0 and name not in absent_clusters)
    mutant_count = len(cluster_variants) - wt_count - len(absent_clusters)

    print(f"\nParsed {len(cluster_variants):,} clusters ({mutant_count:,} with variants, {wt_count:,} WT, "
          f"{len(absent_clusters):,} gene-absent)")

    # Report reference distribution if multi-reference
    unique_refs = set(cluster_reference_ids.values())
    if len(unique_refs) > 1:
        print(f"Found {len(unique_refs)} reference templates: {', '.join(sorted(unique_refs))}")
        ref_counts = defaultdict(int)
        for ref_id in cluster_reference_ids.values():
            ref_counts[ref_id] += 1
        for ref_id, count in sorted(ref_counts.items()):
            print(f"  {ref_id}: {count:,} clusters")

    return cluster_variants, cluster_reference_ids, absent_clusters

def apply_variants_to_reference(reference_seq, variants):
    """Apply SNVs and anchored VCF indels to the reference to rebuild the gene.

    Each record is converted to an edit of the reference: a substitution, a
    deletion of the bases after the anchor (or before it, for a deletion at
    base 1), or an insertion after the anchor base. Edits are applied left to
    right in reference coordinates.
    """
    reference_seq = str(reference_seq)
    edits = []  # (start0, end0, replacement) in reference coordinates
    for variant in variants:
        pos, ref, alt = variant['pos'], variant['ref'], variant['alt']
        start = pos - 1
        if start < 0 or reference_seq[start:start + len(ref)] != ref:
            print(f"Warning: Reference mismatch at position {pos}: expected {ref}, "
                  f"found {reference_seq[start:start + len(ref)]}")
            continue
        if len(ref) == len(alt):
            edits.append((start, start + len(ref), alt))
        elif len(ref) > len(alt):
            n = len(ref) - len(alt)
            if ref[0] == alt[0]:  # anchor precedes the deleted bases
                edits.append((pos, pos + n, ''))
            else:                 # deletion at base 1, anchored on the next base
                edits.append((0, n, ''))
        else:
            # Insertion after the anchor base (an empty edit at index pos)
            edits.append((pos, pos, alt[len(ref):]))

    out, cursor = [], 0
    # Insertions (empty span) sort before a substitution at the same index
    for start, end, replacement in sorted(edits, key=lambda e: (e[0], e[1] - e[0])):
        if start < cursor:
            continue  # overlaps an edit already applied (e.g. SNV inside a deletion)
        out.append(reference_seq[cursor:start])
        out.append(replacement)
        cursor = end
    out.append(reference_seq[cursor:])
    return ''.join(out)


def _variant_tuples(variants, ref_id):
    return [(ref_id, v['pos'], v['ref'], v['alt']) for v in variants]


def _premature_stop(consensus_protein, reference_protein, net_length_change):
    """Position (1-based) of a stop codon before the expected one, else None.

    The expected stop is the reference stop, moved by in-frame indels; for
    frameshifts any stop before the reference stop position counts.
    """
    expected = reference_protein.find('*')
    if expected == -1:
        expected = len(reference_protein)
    if net_length_change % 3 == 0:
        expected += net_length_change // 3
    first = consensus_protein.find('*')
    if first != -1 and first < expected:
        return first + 1
    return None


def _load_references(reference_fasta_or_manager):
    """Return (ref_sequences, ref_proteins, default_ref_id) for single or multi-reference input."""
    is_multi_ref = (ReferenceManager is not None and
                    isinstance(reference_fasta_or_manager, ReferenceManager))
    if is_multi_ref:
        ref_manager = reference_fasta_or_manager
        ref_sequences = {r: ref_manager.get_reference_sequence(r) for r in ref_manager.get_reference_ids()}
        ref_proteins = {r: ref_manager.get_reference_protein(r) for r in ref_manager.get_reference_ids()}
        return ref_sequences, ref_proteins, ref_manager.get_default_reference_id()
    reference_records = list(SeqIO.parse(reference_fasta_or_manager, "fasta"))
    if not reference_records:
        raise ValidationError(f"No sequences found in reference file {reference_fasta_or_manager}")
    reference_seq = str(reference_records[0].seq).upper()
    default_ref_id = reference_records[0].id
    return ({default_ref_id: reference_seq},
            {default_ref_id: translate_dna_to_protein(reference_seq)},
            default_ref_id)


def vcf_to_csv_detailed(vcf_file, reference_fasta_or_manager, output_csv,
                        generate_demultiplexed: bool = True):
    """
    Convert combined VCF to detailed CSV with mutations and analysis using streaming approach.

    Amino acid mutations use the same annotation as ngs_count's AA_MUTATIONS
    (including frameshift 'fs' and in-frame 'delNaa'/'insNaa' tokens), so this
    table and the fitness table always agree. Gene-absent consensuses are
    reported with mutations 'NO_GENE'.

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
        variant_count, vcf_ref_ids, vcf_clusters, missing_cluster_count = validate_vcf_file(vcf_file)
        print(f"  VCF valid: {variant_count:,} variant entries, {len(vcf_clusters):,} clusters")
        if vcf_ref_ids:
            print(f"  VCF references: {len(vcf_ref_ids)} unique ({', '.join(sorted(vcf_ref_ids)[:3])}{'...' if len(vcf_ref_ids) > 3 else ''})")
        if missing_cluster_count > 0:
            print(f"  WARNING: {missing_cluster_count} entries missing CLUSTER tag - these will be skipped!")
            print(f"           Ensure VCF was generated by 'umic-seq-pacbio variants' command")
        ref_sequences, ref_proteins, default_ref_id = _load_references(reference_fasta_or_manager)
    except ValidationError as e:
        print(f"ERROR: {e}")
        return False

    if ReferenceManager is not None and isinstance(reference_fasta_or_manager, ReferenceManager):
        print(reference_fasta_or_manager.get_reference_info())
    else:
        print(f"Reference sequence: {default_ref_id} (length: {len(ref_sequences[default_ref_id])})")

    # Validate reference consistency
    if vcf_ref_ids:
        for warning in validate_references(ref_sequences, vcf_ref_ids):
            print(f"  {warning}")

    # Parse VCF and group variants by cluster using streaming
    cluster_variants, cluster_reference_ids, absent_clusters = parse_vcf_combined_streaming(vcf_file)

    total_clusters = len(cluster_variants)
    print(f"\nProcessing {total_clusters:,} clusters...")
    start_time = time.time()
    clusters_with_unknown_ref = []

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['name', 'reference_id', 'reference_length', 'consensus_protein',
                      'mutations', 'mutation_count', 'dna_mutations', 'dna_variant_count',
                      'synonymous_count', 'hamming_distance', 'premature_stop', 'stop_position',
                      'indelsyn', 'consensus_dna']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for processed, (cluster_name, variants) in enumerate(cluster_variants.items(), 1):
            cluster_ref_id = cluster_reference_ids.get(cluster_name, default_ref_id)
            if cluster_ref_id not in ref_sequences:
                if cluster_name not in absent_clusters:
                    clusters_with_unknown_ref.append((cluster_name, cluster_ref_id))
                cluster_ref_id = default_ref_id
            reference_seq = ref_sequences[cluster_ref_id]
            reference_protein = ref_proteins[cluster_ref_id]

            if cluster_name in absent_clusters:
                # The consensus does not contain the gene (e.g. insert-less plasmid)
                writer.writerow({
                    'name': cluster_name, 'reference_id': cluster_ref_id,
                    'reference_length': len(reference_protein), 'consensus_protein': '',
                    'mutations': 'NO_GENE', 'mutation_count': 0, 'dna_mutations': '',
                    'dna_variant_count': 0, 'synonymous_count': 0, 'hamming_distance': '',
                    'premature_stop': '', 'stop_position': '', 'indelsyn': '', 'consensus_dna': '',
                })
                continue

            consensus_dna = apply_variants_to_reference(reference_seq, variants)
            consensus_protein = translate_dna_to_protein(consensus_dna)
            annotation = annotate_variants(_variant_tuples(variants, cluster_ref_id), reference_seq)
            tokens = annotation['tokens']
            stop_position = _premature_stop(consensus_protein, reference_protein,
                                            len(consensus_dna) - len(reference_seq))

            writer.writerow({
                'name': cluster_name,
                'reference_id': cluster_ref_id,
                'reference_length': len(reference_protein),
                'consensus_protein': consensus_protein,
                'mutations': '+'.join(tokens) if tokens else 'WT',
                'mutation_count': len(tokens),
                'dna_mutations': '+'.join(f"{v['ref']}{v['pos']}{v['alt']}" for v in variants),
                'dna_variant_count': len(variants),
                'synonymous_count': annotation['n_synonymous'],
                'hamming_distance': calculate_hamming_distance(reference_protein, consensus_protein),
                'premature_stop': "yes" if stop_position else "no",
                'stop_position': stop_position or "",
                'indelsyn': "yes" if annotation['indels'] else "no",
                'consensus_dna': consensus_dna
            })

            if processed % 1000 == 0 or processed == total_clusters:
                elapsed_time = time.time() - start_time
                rate = processed / elapsed_time if elapsed_time > 0 else 0
                print(f"\rProgress: {processed / total_clusters * 100:.1f}% "
                      f"({processed:,}/{total_clusters:,}) | Rate: {rate:.0f} clusters/s",
                      end='', flush=True)

    if clusters_with_unknown_ref:
        print(f"\n\nWARNING: {len(clusters_with_unknown_ref)} cluster(s) had unknown REFERENCE_ID and used default reference:")
        for cluster, ref_id in clusters_with_unknown_ref[:5]:
            print(f"  - {cluster}: REFERENCE_ID='{ref_id}' not in reference file")
        if len(clusters_with_unknown_ref) > 5:
            print(f"  ... and {len(clusters_with_unknown_ref) - 5} more")
        print(f"  This may cause incorrect mutation calls! Check that reference file matches variant calling.")

    print(f"\nSuccessfully generated cluster-level CSV: {output_csv}")

    if generate_demultiplexed:
        base, ext = os.path.splitext(output_csv)
        demux_output = f"{base}_mutations{ext}"
        print(f"\nGenerating mutation-level demultiplexed CSV...")
        vcf_to_csv_demultiplexed(vcf_file, reference_fasta_or_manager, demux_output)

    return True


def vcf_to_csv_demultiplexed(vcf_file, reference_fasta_or_manager, output_csv):
    """
    Convert combined VCF to demultiplexed CSV with one row per amino acid mutation.

    Substitutions are evaluated per codon against the reference (all SNVs in a
    codon together); indels get one row each, typed frameshift or in-frame.

    Args:
        vcf_file: Path to combined VCF file
        reference_fasta_or_manager: Either path to reference FASTA (single ref mode)
                                    or ReferenceManager instance (multi-ref mode)
        output_csv: Path to output CSV file

    Returns:
        True on success, False on failure
    """
    try:
        ref_sequences, ref_proteins, default_ref_id = _load_references(reference_fasta_or_manager)
    except ValidationError as e:
        print(f"Error: {e}")
        return False

    cluster_variants, cluster_reference_ids, absent_clusters = parse_vcf_combined_streaming(vcf_file)
    print(f"\nGenerating demultiplexed mutation table...")

    fieldnames = [
        'cluster_name', 'reference_id', 'reference_length',
        'aa_position', 'ref_aa', 'alt_aa', 'mutation',
        'dna_position', 'ref_nt', 'alt_nt', 'dna_mutation',
        'codon_position', 'ref_codon', 'alt_codon',
        'mutation_type', 'is_synonymous'
    ]
    blank = {k: '' for k in fieldnames}
    total_mutations = 0
    wt_clusters = 0
    absent_count = 0

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for cluster_name, variants in cluster_variants.items():
            cluster_ref_id = cluster_reference_ids.get(cluster_name, default_ref_id)
            if cluster_ref_id not in ref_sequences:
                cluster_ref_id = default_ref_id
            reference_seq = ref_sequences[cluster_ref_id]
            reference_protein = ref_proteins[cluster_ref_id]
            common = {'cluster_name': cluster_name, 'reference_id': cluster_ref_id,
                      'reference_length': len(reference_protein)}

            if cluster_name in absent_clusters:
                absent_count += 1
                writer.writerow({**blank, **common, 'mutation': 'NO_GENE', 'mutation_type': 'NO_GENE'})
                continue
            if not variants:
                wt_clusters += 1
                writer.writerow({**blank, **common, 'mutation': 'WT', 'mutation_type': 'WT'})
                continue

            annotation = annotate_variants(_variant_tuples(variants, cluster_ref_id), reference_seq)

            for codon_no, wt_codon, mut_codon, wt_aa, mut_aa, snvs in annotation['codons']:
                is_synonymous = wt_aa == mut_aa
                if is_synonymous:
                    mutation_type = 'synonymous'
                elif mut_aa == '*':
                    mutation_type = 'nonsense'
                elif wt_aa == '*':
                    mutation_type = 'readthrough'
                else:
                    mutation_type = 'missense'
                # One row per DNA change for synonymous codons, one per codon otherwise
                rows = snvs if is_synonymous else snvs[:1]
                for pos_0, ref_nt, alt_nt in rows:
                    dna_mutation = (f"{ref_nt}{pos_0 + 1}{alt_nt}" if is_synonymous else
                                    '+'.join(f"{r}{p + 1}{a}" for p, r, a in snvs))
                    writer.writerow({
                        **common,
                        'aa_position': codon_no, 'ref_aa': wt_aa, 'alt_aa': mut_aa,
                        'mutation': 'synonymous' if is_synonymous else f"{wt_aa}{codon_no}{mut_aa}",
                        'dna_position': pos_0 + 1, 'ref_nt': ref_nt, 'alt_nt': alt_nt,
                        'dna_mutation': dna_mutation,
                        'codon_position': (pos_0 % 3) + 1,
                        'ref_codon': wt_codon, 'alt_codon': mut_codon,
                        'mutation_type': mutation_type,
                        'is_synonymous': 'yes' if is_synonymous else 'no'
                    })
                    total_mutations += 1

            indel_tokens = {(start0, kind, n): (codon_no, token)
                            for codon_no, token, kind, n, start0 in annotation['indels']}
            for v in variants:
                event = _indel_events([(cluster_ref_id, v['pos'], v['ref'], v['alt'])])
                if not event:
                    continue
                start0, kind, n = event[0]
                codon_no, token = indel_tokens[(start0, kind, n)]
                frameshift = n % 3 != 0
                writer.writerow({
                    **blank, **common,
                    'aa_position': codon_no, 'ref_aa': reference_protein[codon_no - 1],
                    # Only the first frameshift gets a token; later ones are out of frame anyway
                    'mutation': token or f"{reference_protein[codon_no - 1]}{codon_no}fs",
                    'dna_position': v['pos'], 'ref_nt': v['ref'], 'alt_nt': v['alt'],
                    'dna_mutation': f"{v['ref']}{v['pos']}{v['alt']}",
                    'mutation_type': ('frameshift' if frameshift else f"inframe_{'deletion' if kind == 'del' else 'insertion'}"),
                    'is_synonymous': 'no'
                })
                total_mutations += 1

    print(f"Generated demultiplexed CSV: {output_csv}")
    print(f"  Total mutation rows: {total_mutations:,}")
    print(f"  WT clusters: {wt_clusters:,}")
    if absent_count:
        print(f"  Gene-absent clusters: {absent_count:,}")
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
