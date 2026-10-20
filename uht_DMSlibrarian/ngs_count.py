#!/usr/bin/env python3
"""
NGS pool counting via UMI-to-consensus exact matching (circular, both strands).

- Streams paired-end reads per pool (R1/R2), extracts UMIs using probe and umi_loc
- Builds an index of all possible UMI-length substrings from circular consensus
  sequences on both strands for fast lookup
- On perfect UMI match, increments counts for that consensus' VCF variant rows
- Outputs a wide CSV: rows = unique VCF entries (CHROM, POS, REF, ALT),
  columns = pools

Designed for low memory and speed: consensus index built once; reads streamed.

Supports multi-reference mode: when REFERENCE_ID is present in VCF, uses the correct
reference sequence for each cluster's AA translation.
"""

import os
import sys
import gzip
import json
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional
import traceback
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq

# Import ReferenceManager for multi-reference support
try:
    from .reference_manager import ReferenceManager
except ImportError:
    ReferenceManager = None

# Force immediate output - write directly to stderr if needed
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)


def open_maybe_gzip(path: str):
    if path.endswith('.gz'):
        return gzip.open(path, 'rt')
    return open(path, 'r')


def read_fastq_sequences(path: str) -> Iterable[str]:
    """Stream sequences from FASTQ (text or gz)."""
    with open_maybe_gzip(path) as fh:
        i = 0
        seq = None
        for line in fh:
            i += 1
            m = i % 4
            if m == 2:
                seq = line.strip()
                yield seq


def reverse_complement(seq: str) -> str:
    comp = str.maketrans('ACGTNacgtn', 'TGCANtgcan')
    return seq.translate(comp)[::-1]


def get_block_ranges(umi_len: int, max_mismatches: int) -> List[Tuple[int, int]]:
    """Split UMI into (max_mismatches+1) blocks for candidate narrowing."""
    if max_mismatches < 1:
        return []
    blocks = max_mismatches + 1
    base = umi_len // blocks
    rem = umi_len % blocks
    ranges: List[Tuple[int, int]] = []
    start = 0
    for i in range(blocks):
        size = base + (1 if i < rem else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def build_block_index(umi_list: List[str], block_ranges: List[Tuple[int, int]]) -> Dict[Tuple[int, str], List[int]]:
    """Build block index for mismatch-tolerant matching."""
    index: Dict[Tuple[int, str], List[int]] = {}
    if not block_ranges:
        return index
    for idx, umi in enumerate(umi_list):
        for block_id, (start, end) in enumerate(block_ranges):
            key = (block_id, umi[start:end])
            index.setdefault(key, []).append(idx)
    return index


def hamming_distance(a: str, b: str, max_mismatches: int) -> int:
    """Hamming distance with early exit once max_mismatches exceeded."""
    mismatches = 0
    for x, y in zip(a, b):
        if x != y:
            mismatches += 1
            if mismatches > max_mismatches:
                return mismatches
    return mismatches


def find_umi_with_mismatches(query: str, umi_list: List[str],
                             block_index: Dict[Tuple[int, str], List[int]],
                             block_ranges: List[Tuple[int, int]],
                             max_mismatches: int,
                             cons_list: Optional[List[str]] = None,
                             genotypes: Optional[Dict[str, object]] = None) -> Optional[int]:
    """Return index of best matching UMI within max_mismatches or None.

    With cons_list and genotypes, returns -1 if equally good matches belong to
    consensuses with different genotypes (an ambiguous read).
    """
    if not block_ranges:
        return None
    candidates = set()
    for block_id, (start, end) in enumerate(block_ranges):
        key = (block_id, query[start:end])
        for idx in block_index.get(key, []):
            candidates.add(idx)
    if not candidates:
        return None
    best_idx = None
    best_mismatches = max_mismatches + 1
    best_genotypes = set()
    for idx in sorted(candidates):
        cand = umi_list[idx]
        mismatches = hamming_distance(query, cand, max_mismatches)
        if mismatches > max_mismatches:
            continue
        genotype = None
        if cons_list is not None:
            name = cons_list[idx]
            genotype = AMBIGUOUS if name is AMBIGUOUS else (genotypes or {}).get(name, name)
        if mismatches < best_mismatches:
            best_mismatches = mismatches
            best_idx = idx
            best_genotypes = {genotype}
        elif mismatches == best_mismatches:
            best_genotypes.add(genotype)
    if best_idx is not None and cons_list is not None and (
            len(best_genotypes) > 1 or AMBIGUOUS in best_genotypes):
        return -1
    return best_idx


def find_approx(query: str, target: str, max_mismatch: int) -> int:
    """Return start index of first occurrence of query in target allowing up to max_mismatch mismatches; -1 if not found."""
    q = query.upper()
    t = target.upper()
    qlen = len(q)
    tlen = len(t)
    if qlen == 0 or qlen > tlen:
        return -1
    # Fast exact pass first
    pos = t.find(q)
    if pos != -1:
        return pos
    # Sliding Hamming window
    for i in range(0, tlen - qlen + 1):
        window = t[i:i+qlen]
        mism = 0
        for a, b in zip(q, window):
            if a != b:
                mism += 1
                if mism > max_mismatch:
                    break
        if mism <= max_mismatch:
            return i
    return -1


def load_probe_sequences(probe_fasta: str) -> Tuple[str, str]:
    """Load probe FASTA (single record)."""
    with open(probe_fasta, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    seq_lines = [l for l in lines if not l.startswith('>')]
    probe = ''.join(seq_lines)
    return probe.upper(), reverse_complement(probe.upper())


def merge_trimmed_internal(s1: str, s2: str, left_ignore: int, right_ignore: int, min_overlap: int = 12) -> str:
    """Create merged internal fragment from paired-end reads.
    - Trim first 22 and last 24 bases from each
    - Reverse-complement R2 internal
    - Merge by maximal suffix/prefix overlap; fallback to concatenation
    """
    s1f = s1.upper()
    s2f = s2.upper()
    if len(s1f) <= left_ignore + right_ignore or len(s2f) <= left_ignore + right_ignore:
        return ''
    r1 = s1f[left_ignore:len(s1f) - right_ignore]
    r2 = s2f[left_ignore:len(s2f) - right_ignore]
    r2rc = reverse_complement(r2)
    # find maximal overlap of r1 suffix with r2rc prefix
    max_ol = 0
    max_possible = min(len(r1), len(r2rc))
    for ol in range(max_possible, min_overlap - 1, -1):
        if r1.endswith(r2rc[:ol]):
            max_ol = ol
            break
    if max_ol >= min_overlap:
        return r1 + r2rc[max_ol:]
    # also try the other orientation (if library orientation unexpected)
    # overlap of r2rc suffix with r1 prefix
    for ol in range(max_possible, min_overlap - 1, -1):
        if r2rc.endswith(r1[:ol]):
            max_ol = ol
            return r2rc + r1[max_ol:]
    # fallback concatenate
    return r1 + r2rc


def extract_umi_from_assembled(seq: str, umi_len: int, left_ignore: int, right_ignore: int) -> str:
    """Extract UMI by removing first 22 and last 24 bases and returning the internal UMI-length substring.
    For Illumina, UMI lives in the internal window. If trimmed internal is too short, return empty.
    """
    s = seq.upper()
    if len(s) < left_ignore + right_ignore + umi_len:
        return ''
    internal = s[left_ignore:len(s)-right_ignore]
    if len(internal) < umi_len:
        return ''
    # Return first umi_len bases from internal window
    return internal[:umi_len]


def load_consensus_fasta(consensus_dir: str) -> Dict[str, str]:
    cons: Dict[str, str] = {}
    # Skip hidden files such as macOS AppleDouble '._' files
    files = [f for f in os.listdir(consensus_dir)
             if (f.endswith('.fasta') or f.endswith('.fa')) and not f.startswith('.')]
    total = len(files)
    for idx, fn in enumerate(files):
        if idx % 1000 == 0 and idx > 0:
            print(f"  Loaded {idx}/{total} consensus files...", flush=True)
        path = os.path.join(consensus_dir, fn)
        name = Path(fn).stem
        # read first record only
        with open(path, 'r') as f:
            seq_lines = []
            for line in f:
                if line.startswith('>'):
                    if seq_lines:
                        break
                    continue
                seq_lines.append(line.strip())
        cons[name] = ''.join(seq_lines).upper()
    return cons


# Index value for a UMI-length sequence found in consensuses with different
# genotypes: reads carrying it cannot be assigned and are counted as ambiguous.
AMBIGUOUS = ''


def build_umi_index(consensus_map: Dict[str, str], umi_len: int,
                    genotypes: Optional[Dict[str, object]] = None) -> Dict[str, str]:
    """Build mapping from UMI (exact) to consensus name.
    Index is built from circular sequences on both strands.

    If several consensuses contain the same sequence (e.g. one molecule split
    into two clusters), the first one seen keeps it when their genotypes agree,
    since the reads then count towards the same variant either way. If the
    genotypes differ (given `genotypes`: consensus name -> genotype key), the
    sequence is marked AMBIGUOUS instead of going to whichever came first.
    """
    idx: Dict[str, str] = {}

    def add(umi: str, name: str) -> None:
        other = idx.get(umi)
        if other is None:
            idx[umi] = name
        elif (genotypes is not None and other is not AMBIGUOUS and other != name
              and genotypes.get(other) != genotypes.get(name)):
            idx[umi] = AMBIGUOUS

    total = len(consensus_map)
    for count, (name, seq) in enumerate(consensus_map.items()):
        if count > 0 and count % 1000 == 0:
            print(f"  Indexed {count}/{total} consensus sequences...", flush=True)
        if not seq:
            continue
        seq_circ = (seq + seq)
        seq_rc = reverse_complement(seq)
        seq_rc_circ = (seq_rc + seq_rc)
        L = len(seq)
        if L < umi_len:
            continue
        # forward
        for i in range(L):
            umi = seq_circ[i:i+umi_len]
            if len(umi) == umi_len:
                add(umi, name)
        # reverse
        for i in range(L):
            umi = seq_rc_circ[i:i+umi_len]
            if len(umi) == umi_len:
                add(umi, name)
    return idx


def read_vcf_variants(vcf_path: str) -> List[Tuple[str, int, str, str]]:
    rows: List[Tuple[str, int, str, str]] = []
    if not os.path.exists(vcf_path):
        return rows
    with open(vcf_path, 'r') as f:
        for line in f:
            if not line or line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            chrom = parts[2] if parts[0] == '' else parts[0]
            pos = int(parts[1])
            ref = parts[3]
            alt = parts[4].split(',')[0]
            rows.append((chrom, pos, ref, alt))
    return rows


def read_vcf_haplotype(vcf_path: str) -> Tuple[str, List[Tuple[str, int, str, str]], Optional[str]]:
    """
    Return a semicolon-joined, position-sorted mutation list, list of variant tuples,
    and reference ID (if present in VCF).

    Format per mutation: CHROM:POS:REF>ALT. Returns empty string if no variants.

    Returns:
        Tuple of (mutations_string, variants_list, reference_id)
    """
    muts = []
    variants = []
    reference_id = None

    if not os.path.exists(vcf_path):
        return '', variants, None

    with open(vcf_path, 'r') as f:
        for line in f:
            if not line:
                continue
            if line.startswith('##reference='):
                reference_id = line.strip().split('=', 1)[1]
                continue
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue

            # Skip WT marker entries (position 0)
            if parts[1] == '0' and len(parts) > 6 and parts[6] == 'WT':
                # Extract REFERENCE_ID from WT marker
                if len(parts) > 7:
                    for info_item in parts[7].split(';'):
                        if info_item.startswith('REFERENCE_ID='):
                            reference_id = info_item.split('=')[1]
                continue

            chrom = parts[0] if parts[0] else parts[2]
            pos = int(parts[1])
            ref = parts[3]
            alt = parts[4].split(',')[0]

            # Extract REFERENCE_ID from INFO field if present
            if len(parts) > 7:
                for info_item in parts[7].split(';'):
                    if info_item.startswith('REFERENCE_ID='):
                        reference_id = info_item.split('=')[1]

            variants.append((chrom, pos, ref, alt))
            muts.append((pos, f"{chrom}:{pos}:{ref}>{alt}"))

    muts.sort(key=lambda x: x[0])
    return ';'.join(m for _, m in muts), variants, reference_id


def read_vcf_gene_status(vcf_path: str) -> Optional[str]:
    """Return ##gene_status (FULL/PARTIAL/ABSENT) from a VCF, or None for VCFs
    written by older versions that did not record it."""
    if not os.path.exists(vcf_path):
        return None
    with open(vcf_path, 'r') as f:
        for line in f:
            if not line.startswith('##'):
                break
            if line.startswith('##gene_status='):
                return line.strip().split('=', 1)[1]
    return None


def _indel_events(variants: List[Tuple[str, int, str, str]]) -> List[Tuple[int, str, int]]:
    """Convert VCF indel records to (start0, kind, n_bases) in reference coordinates.

    start0 is the 0-based index of the first deleted base (deletions) or the
    0-based index before which bases are inserted (insertions).
    """
    events = []
    for _chrom, pos, ref_allele, alt_allele in variants:
        if len(ref_allele) == len(alt_allele):
            continue
        if len(ref_allele) > len(alt_allele):
            n = len(ref_allele) - len(alt_allele)
            # Anchor base precedes the deletion, except for a deletion at base 1
            start0 = pos if ref_allele[0] == alt_allele[0] else pos - 1
            events.append((start0, 'del', n))
        else:
            n = len(alt_allele) - len(ref_allele)
            events.append((pos, 'ins', n))
    return sorted(events)


def annotate_variants(variants: List[Tuple[str, int, str, str]], ref) -> Dict[str, object]:
    """Amino acid annotation of one consensus' variants against a reference.

    Shared by ngs_count (AA_MUTATIONS) and analyze (detailed_mutations.csv) so
    both always agree. Returns a dict with:
        tokens: AA mutation tokens sorted by codon, e.g. ['A28V', 'L81fs', 'Q51del1aa']
        codons: [(codon_no, wt_codon, mut_codon, wt_aa, mut_aa, snvs)] for codons with SNVs
        indels: [(codon_no, token_or_None, kind, n_bases, start0)] per indel event
        n_synonymous: number of codons whose SNVs do not change the amino acid
    """
    ref = Seq(str(ref))
    ref_protein = str(ref.translate())

    # FIX: indels were never present in the VCFs. Now they are; keep them
    # out of the codon-substitution logic and annotate them separately.
    snvs = [v for v in variants if len(v[2]) == 1 and len(v[3]) == 1]
    indels = _indel_events(variants)

    # Convert to amino acid mutations - prevariant neess fall within codon
    codons: Dict[int, List[Tuple[int, str, str]]] = {}
    for chrom, pos, ref_allele, alt_allele in snvs:
        pos_0 = pos - 1  # 0-based
        codon_pos = pos_0 // 3
        if codon_pos not in codons:
            codons[codon_pos] = []
        codons[codon_pos].append((pos_0, ref_allele, alt_allele))

    # (codon number, token) so substitutions and indels sort together
    aa_tokens: List[Tuple[int, str]] = []
    codon_info = []
    indel_info = []
    n_synonymous = 0
    for codon_pos in sorted(codons.keys()):
        codon_start = codon_pos * 3
        if codon_start + 3 > len(ref):
            # Codon extends beyond reference, skip this codon
            continue

        wt_codon = str(ref[codon_start:codon_start + 3])
        mut_codon = list(wt_codon)

        # Apply all mutations in this codon
        valid = True
        for pos_0, ref_allele, alt_allele in codons[codon_pos]:
            rel_pos = pos_0 - codon_start
            if rel_pos < 3 and pos_0 < len(ref) and ref[pos_0] == ref_allele:
                mut_codon[rel_pos] = alt_allele
            else:
                valid = False
                break

        if not valid:
            # Fallback: record as codon number (amino acid position)
            wt_aa = str(Seq(wt_codon).translate())
            aa_tokens.append((codon_pos + 1, f"{wt_aa}{codon_pos + 1}X"))
            continue

        mut_codon_str = ''.join(mut_codon)
        wt_aa = str(Seq(wt_codon).translate())
        mut_aa = str(Seq(mut_codon_str).translate())

        codon_info.append((codon_pos + 1, wt_codon, mut_codon_str, wt_aa, mut_aa, codons[codon_pos]))
        if wt_aa != mut_aa:
            # Non-synonymous - record as amino acid
            aa_tokens.append((codon_pos + 1, f"{wt_aa}{codon_pos + 1}{mut_aa}"))
        else:
            # Synonymous (silent) change: not an AA mutation
            n_synonymous += 1

    # Indel tokens use lowercase suffixes so fitness_analysis's substitution
    # regex ([A-Z])(\d+)([A-Z*]) does not mistake them for substitutions:
    #   frameshift   L81fs    (first frameshifting indel only; frame is lost after it)
    #   in-frame     Q51del1aa / A50ins2aa
    fs_seen = False
    for start0, kind, n in indels:
        codon = min(start0 // 3 + 1, len(ref_protein))
        wt_aa = ref_protein[codon - 1]
        token = None
        if n % 3:
            if not fs_seen:
                token = f"{wt_aa}{codon}fs"
                fs_seen = True
        else:
            token = f"{wt_aa}{codon}{kind}{n // 3}aa"
        if token:
            aa_tokens.append((codon, token))
        indel_info.append((codon, token, kind, n, start0))

    return {
        'tokens': [t for _, t in sorted(aa_tokens, key=lambda x: x[0])],
        'codons': codon_info,
        'indels': indel_info,
        'n_synonymous': n_synonymous,
    }


def collect_all_variant_rows(variants_dir: str) -> Tuple[List[Tuple[str,int,str,str]], Dict[str, List[int]]]:
    """Return list of unique variant keys and mapping from consensus name to indices in that list."""
    unique: Dict[Tuple[str,int,str,str], int] = {}
    per_consensus: Dict[str, List[int]] = {}
    for fn in os.listdir(variants_dir):
        if not fn.endswith('.vcf') or fn.startswith('.'):
            continue
        name = Path(fn).stem  # matches consensus stem
        vpath = os.path.join(variants_dir, fn)
        idxs: List[int] = []
        for key in read_vcf_variants(vpath):
            if key not in unique:
                unique[key] = len(unique)
            idxs.append(unique[key])
        per_consensus[name] = idxs
    # invert to list
    rows: List[Tuple[str,int,str,str]] = [None] * len(unique)
    for key, i in unique.items():
        rows[i] = key
    return rows, per_consensus


def collect_haplotype_rows(variants_dir: str, ref_seq_or_manager) -> Tuple[List[Tuple[str, str, str, str]], Dict[str, int]]:
    """
    Return list of (consensus, mutations_str, aa_mutations_str, reference_id) and mapping from consensus name to row index.

    Args:
        variants_dir: Directory containing per-consensus VCF files
        ref_seq_or_manager: Either a reference sequence string (single ref mode)
                           or ReferenceManager instance (multi-ref mode)

    Returns:
        Tuple of (rows list, index dict)
        - rows: [(name, mutations_str, aa_mutations_str, reference_id), ...]
        - index: {consensus_name: row_index}
    """
    rows: List[Tuple[str, str, str, str]] = []
    index: Dict[str, int] = {}

    # Determine reference mode
    is_multi_ref = (ReferenceManager is not None and
                   isinstance(ref_seq_or_manager, ReferenceManager))

    if is_multi_ref:
        ref_manager = ref_seq_or_manager
        # Build lookup dict for reference sequences
        ref_sequences: Dict[str, Seq] = {}
        for ref_id in ref_manager.get_reference_ids():
            ref_sequences[ref_id] = Seq(ref_manager.get_reference_sequence(ref_id))
        default_ref_id = ref_manager.get_default_reference_id()
    else:
        # Single reference mode
        ref_seq = ref_seq_or_manager
        default_ref_id = "reference"
        ref_sequences = {default_ref_id: Seq(ref_seq)}

    for fn in os.listdir(variants_dir):
        if not fn.endswith('.vcf') or fn.startswith('.'):
            continue
        name = Path(fn).stem
        vpath = os.path.join(variants_dir, fn)
        muts_str, variants, vcf_ref_id = read_vcf_haplotype(vpath)

        # Determine which reference to use for this consensus
        ref_id = vcf_ref_id if vcf_ref_id and vcf_ref_id in ref_sequences else default_ref_id
        ref = ref_sequences[ref_id]

        # FIX: a consensus that does not contain the gene is not WT
        if read_vcf_gene_status(vpath) == 'ABSENT':
            rows.append((name, muts_str, 'NO_GENE', ref_id))
            index[name] = len(rows) - 1
            continue

        aa_muts = annotate_variants(variants, ref)['tokens']
        aa_muts_str = '+'.join(aa_muts) if aa_muts else 'WT'
        rows.append((name, muts_str, aa_muts_str, ref_id))
        index[name] = len(rows) - 1
    return rows, index


def find_pool_folders(pools_dir: str) -> List[str]:
    folders = []
    for entry in os.scandir(pools_dir):
        if entry.is_dir() and not entry.name.startswith('.'):
            folders.append(entry.path)
    folders.sort()
    return folders


def find_r1_r2_pairs(folder: str) -> List[Tuple[str, str]]:
    """Find all R1/R2 pairs in a folder. Pairs are identified by matching basenames
    that differ only by R1/R2 designation.
    Returns list of (r1_path, r2_path) tuples.
    """
    r1_files = {}
    r2_files = {}
    
    for fn in os.listdir(folder):
        if not (fn.endswith('.fastq') or fn.endswith('.fastq.gz')) or fn.startswith('.'):
            continue
        
        full_path = os.path.join(folder, fn)
        
        # Normalize filename to extract base key (remove R1/R2 and extension)
        # Handle patterns: sample_R1.fastq.gz, sample_R1_001.fastq.gz, sampleR1.fastq.gz
        key = None
        
        if '_R1' in fn:
            # Replace _R1 with placeholder, then remove extension
            key = fn.replace('_R1', '_R*')
            # Remove extension
            if key.endswith('.fastq.gz'):
                key = key[:-9]  # Remove .fastq.gz
            elif key.endswith('.fastq'):
                key = key[:-6]  # Remove .fastq
            r1_files[key] = full_path
        elif 'R1.' in fn:
            # Handle R1.fastq.gz pattern
            key = fn.replace('R1.', 'R*.')
            if key.endswith('.fastq.gz'):
                key = key[:-9]
            elif key.endswith('.fastq'):
                key = key[:-6]
            r1_files[key] = full_path
        
        if '_R2' in fn:
            key = fn.replace('_R2', '_R*')
            if key.endswith('.fastq.gz'):
                key = key[:-9]
            elif key.endswith('.fastq'):
                key = key[:-6]
            r2_files[key] = full_path
        elif 'R2.' in fn:
            key = fn.replace('R2.', 'R*.')
            if key.endswith('.fastq.gz'):
                key = key[:-9]
            elif key.endswith('.fastq'):
                key = key[:-6]
            r2_files[key] = full_path
    
    # Match pairs by key
    pairs = []
    for key in r1_files:
        if key in r2_files:
            pairs.append((r1_files[key], r2_files[key]))
    
    return pairs


def run_ngs_count(pools_dir: str, consensus_dir: str, variants_dir: str, probe_fasta: str,
                  umi_len: int, umi_loc: str, output_csv: str, reference_fasta_or_manager,
                  left_ignore: int = 22, right_ignore: int = 24,
                  pear_min_overlap: int = 20, pear_yolo: bool = False,
                  umi_mismatches: int = 0) -> bool:
    """
    Run NGS pool counting.

    Args:
        pools_dir: Directory containing per-pool folders with R1/R2 fastqs
        consensus_dir: Consensus directory (from consensus step)
        variants_dir: Variants directory with per-consensus VCFs
        probe_fasta: Probe FASTA file
        umi_len: UMI length
        umi_loc: UMI location ('up' or 'down')
        output_csv: Output counts CSV file
        reference_fasta_or_manager: Either path to reference FASTA (single ref mode)
                                    or ReferenceManager instance (multi-ref mode)
        left_ignore: Bases to ignore from start of assembled read
        right_ignore: Bases to ignore from end of assembled read
        pear_min_overlap: Minimum overlap length for PEAR read merging (default: 20)
        pear_yolo: If True, use maximally permissive PEAR settings (disables p-value test)
        umi_mismatches: Maximum mismatches allowed in UMI matching (default: 0)

    Returns:
        True on success, False on failure
    """
    try:
        print("Starting...", flush=True)
        # Create the output folder up front: otherwise a missing folder only
        # fails when writing results, after all pools have been counted
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        print("Loading probe sequences...", flush=True)
        probe_fwd, probe_rev = load_probe_sequences(probe_fasta)
        print(f"Probe length: {len(probe_fwd)} bp", flush=True)

        print("Loading consensus sequences...", flush=True)
        consensus = load_consensus_fasta(consensus_dir)
        print(f"Loaded {len(consensus)} consensus sequences", flush=True)

        if umi_mismatches < 0:
            raise ValueError("umi_mismatches must be >= 0")
        if umi_mismatches > 5:
            raise ValueError("umi_mismatches must be <= 5")
        if umi_mismatches >= umi_len:
            raise ValueError("umi_mismatches must be less than umi_len")

        # Determine reference mode
        is_multi_ref = (ReferenceManager is not None and
                       isinstance(reference_fasta_or_manager, ReferenceManager))
        if is_multi_ref:
            ref_seq_or_manager = reference_fasta_or_manager
        else:
            ref_seq_or_manager = str(SeqIO.read(reference_fasta_or_manager, "fasta").seq).upper()

        # Genotype of each consensus, used to tell harmless duplicate clusters
        # (same genotype) from conflicting ones when they share a UMI
        print("Annotating consensus genotypes...", flush=True)
        hap_rows, hap_index = collect_haplotype_rows(variants_dir, ref_seq_or_manager)
        genotypes = {name: (ref_id, aa_muts) for name, _muts, aa_muts, ref_id in hap_rows}

        print("Building UMI index (circular, both strands)...", flush=True)
        umi_index = build_umi_index(consensus, umi_len, genotypes)
        n_ambiguous_keys = sum(1 for v in umi_index.values() if v is AMBIGUOUS)
        print(f"Index contains {len(umi_index)} unique UMIs "
              f"({n_ambiguous_keys:,} shared by consensuses with different genotypes)", flush=True)
        if umi_mismatches > 0:
            print(f"UMI mismatch tolerance: {umi_mismatches}", flush=True)
            umi_list = list(umi_index.keys())
            cons_list = [umi_index[umi] for umi in umi_list]
            block_ranges = get_block_ranges(umi_len, umi_mismatches)
            block_index = build_block_index(umi_list, block_ranges)
        else:
            umi_list = []
            cons_list = []
            block_ranges = []
            block_index = {}

        print("Collecting variant rows...", flush=True)
        var_rows, per_consensus = collect_all_variant_rows(variants_dir)
        print(f"Found {len(var_rows)} unique variant rows across {len(per_consensus)} consensus", flush=True)

        if is_multi_ref:
            print(reference_fasta_or_manager.get_reference_info(), flush=True)
        else:
            print(f"Reference length: {len(ref_seq_or_manager)} bp", flush=True)

        print(f"Finding pool folders in: {pools_dir}", flush=True)
        pool_folders = find_pool_folders(pools_dir)
        pool_names = [Path(p).name for p in pool_folders]
        print(f"Found {len(pool_folders)} pools: {pool_names}", flush=True)
    except Exception as e:
        print(f"ERROR in initialization: {e}", flush=True)
        traceback.print_exc()
        return False

    counts = [[0 for _ in pool_folders] for _ in range(len(var_rows))]
    # Haplotype rows (consensus-level, preserve multi-mutations)
    hap_counts = [[0 for _ in pool_folders] for _ in range(len(hap_rows))]
    metrics = {"pools": [], "global_read_count": 0, "global_umi_extracted": 0, "global_umi_matched": 0,
               "global_umi_ambiguous": 0}

    def match_umi(umi: str) -> Optional[str]:
        """Consensus name for a read's UMI, AMBIGUOUS, or None if unmatched."""
        cons_name = umi_index.get(umi)
        if cons_name is None and umi_mismatches > 0:
            idx = find_umi_with_mismatches(umi, umi_list, block_index, block_ranges, umi_mismatches,
                                           cons_list, genotypes)
            if idx == -1:
                return AMBIGUOUS
            if idx is not None:
                cons_name = cons_list[idx]
        return cons_name

    for j, pool in enumerate(pool_folders):
        pool_name = pool_names[j]
        print(f"\nProcessing pool {j+1}/{len(pool_folders)}: {pool_name}")
        
        # Find all R1/R2 pairs in this pool
        pairs = find_r1_r2_pairs(pool)
        if not pairs:
            print(f"  WARNING: No R1/R2 pairs found, skipping")
            continue
        
        print(f"  Found {len(pairs)} R1/R2 pair(s)")
        
        # Accumulate counts across all pairs in this pool
        pool_read_count = 0
        pool_umi_extracted = 0
        pool_umi_matched = 0
        pool_umi_ambiguous = 0
        
        for pair_idx, (r1, r2) in enumerate(pairs):
            pair_name = Path(r1).stem.replace('_R1', '').replace('R1', '')
            if len(pairs) > 1:
                print(f"  Processing pair {pair_idx+1}/{len(pairs)}: {Path(r1).name} / {Path(r2).name}")
            else:
                print(f"  R1: {Path(r1).name}")
                print(f"  R2: {Path(r2).name}")
            
            # Prefer PEAR merging for robust overlap handling
            # Use unique prefix for each pair to avoid conflicts
            prefix = os.path.join(pool, f"{pool_name}_pair{pair_idx+1}")
            assembled_fastq = prefix + '.assembled.fastq'
            
            # Reuse PEAR output only if a finished run with the same inputs and
            # settings wrote it (the parameter record is written after PEAR
            # completes, so interrupted runs and changed settings are redone)
            pear_params = {
                'r1': os.path.basename(r1), 'r1_size': os.path.getsize(r1),
                'r2': os.path.basename(r2), 'r2_size': os.path.getsize(r2),
                'q': 20, 'v': pear_min_overlap, 'p': 1.0 if pear_yolo else None,
            }
            params_file = prefix + '.pear_params.json'
            previous_params = None
            if os.path.exists(params_file):
                try:
                    with open(params_file) as pf:
                        previous_params = json.load(pf)
                except (OSError, ValueError):
                    previous_params = None
            if os.path.exists(assembled_fastq) and previous_params == pear_params:
                print("    Using existing PEAR output...", flush=True)
            else:
                if os.path.exists(params_file):
                    os.remove(params_file)
                if os.path.exists(assembled_fastq):
                    print("    Existing PEAR output is incomplete or used other settings; re-running PEAR", flush=True)
                try:
                    print("    Merging with PEAR...", flush=True)
                    cmd = [
                        'pear',
                        '-f', r1,
                        '-r', r2,
                        '-o', prefix,
                        '-q', '20',
                        '-v', str(pear_min_overlap)
                    ]
                    if pear_yolo:
                        cmd.extend(['-p', '1.0'])
                        print("    (YOLO mode: p-value test disabled)", flush=True)
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if os.path.exists(assembled_fastq):
                        with open(params_file, 'w') as pf:
                            json.dump(pear_params, pf)
                    else:
                        print("    WARNING: PEAR did not produce assembled reads; falling back to on-the-fly merge")
                        assembled_fastq = ''
                except Exception as e:
                    print(f"    WARNING: PEAR merge failed ({e}); falling back to on-the-fly merge")
                    assembled_fastq = ''

            read_count = 0
            umi_extracted = 0
            umi_matched = 0
            umi_ambiguous = 0

            if assembled_fastq:
                print("    Scanning assembled reads for UMI matches...", flush=True)
                for seq in read_fastq_sequences(assembled_fastq):
                    read_count += 1
                    if read_count % 100000 == 0:
                        print(f"    Processed {read_count:,} merged reads, extracted {umi_extracted}, matched {umi_matched}", end='\r')
                    umi = extract_umi_from_assembled(seq, umi_len, left_ignore, right_ignore)
                    if not umi:
                        continue
                    umi_extracted += 1
                    cons_name = match_umi(umi)
                    if cons_name is None:
                        continue
                    if cons_name is AMBIGUOUS:
                        umi_ambiguous += 1
                        continue
                    umi_matched += 1
                    idxs = per_consensus.get(cons_name, [])
                    for i in idxs:
                        counts[i][j] += 1
                    if cons_name in hap_index:
                        hap_counts[hap_index[cons_name]][j] += 1
            else:
                # Fallback: stream paired reads and merge on the fly
                print("    Merging reads on-the-fly...", flush=True)
                for s1, s2 in zip(read_fastq_sequences(r1), read_fastq_sequences(r2)):
                    read_count += 1
                    if read_count % 100000 == 0:
                        print(f"    Processed {read_count:,} reads, extracted {umi_extracted}, matched {umi_matched}", end='\r')
                    # Merge, trim and extract from merged
                    merged = merge_trimmed_internal(s1, s2, left_ignore, right_ignore)
                    if not merged:
                        continue
                    umi = extract_umi_from_assembled(merged, umi_len, left_ignore, right_ignore)
                    if not umi:
                        continue
                    umi_extracted += 1
                    cons_name = match_umi(umi)
                    if cons_name is None:
                        continue
                    if cons_name is AMBIGUOUS:
                        umi_ambiguous += 1
                        continue
                    umi_matched += 1
                    idxs = per_consensus.get(cons_name, [])
                    for i in idxs:
                        counts[i][j] += 1
                    if cons_name in hap_index:
                        hap_counts[hap_index[cons_name]][j] += 1
            
            pool_read_count += read_count
            pool_umi_extracted += umi_extracted
            pool_umi_matched += umi_matched
            pool_umi_ambiguous += umi_ambiguous
            
            if len(pairs) > 1:
                print(f"\n    Pair {pair_idx+1} summary: {read_count:,} reads, {umi_extracted:,} UMIs extracted, {umi_matched:,} matched")

        print(f"\n  Pool {pool_name} total: {pool_read_count:,} reads, {pool_umi_extracted:,} UMIs extracted, {pool_umi_matched:,} matched to consensus, "
              f"{pool_umi_ambiguous:,} ambiguous (UMI shared by consensuses with different genotypes; not counted)")
        metrics["pools"].append(
            {
                "pool_name": pool_name,
                "read_count": int(pool_read_count),
                "umi_extracted": int(pool_umi_extracted),
                "umi_matched": int(pool_umi_matched),
                "umi_ambiguous": int(pool_umi_ambiguous),
                "match_rate": float(pool_umi_matched / pool_umi_extracted) if pool_umi_extracted > 0 else 0.0,
            }
        )
        metrics["global_read_count"] += int(pool_read_count)
        metrics["global_umi_extracted"] += int(pool_umi_extracted)
        metrics["global_umi_matched"] += int(pool_umi_matched)
        metrics["global_umi_ambiguous"] += int(pool_umi_ambiguous)

    print(f"\nWriting output to: {output_csv}")
    with open(output_csv, 'w') as out:
        header = ['CHROM', 'POS', 'REF', 'ALT'] + pool_names
        out.write(','.join(header) + '\n')
        for i, key in enumerate(var_rows):
            chrom, pos, ref, alt = key
            row = [str(chrom), str(pos), ref, alt] + [str(counts[i][j]) for j in range(len(pool_folders))]
            out.write(','.join(row) + '\n')
    # Write haplotype counts (now includes REFERENCE_ID)
    hap_csv = str(Path(output_csv).parent / 'pool_haplotype_counts.csv')
    print(f"Writing haplotype counts to: {hap_csv}")
    with open(hap_csv, 'w') as out:
        header = ['CONSENSUS', 'REFERENCE_ID', 'MUTATIONS', 'AA_MUTATIONS'] + pool_names
        out.write(','.join(header) + '\n')
        for i, (cons_name, muts, aa_muts, ref_id) in enumerate(hap_rows):
            row = [cons_name, ref_id, muts, aa_muts] + [str(hap_counts[i][j]) for j in range(len(pool_folders))]
            out.write(','.join(row) + '\n')

    # Write counts merged on non-synonymous amino acid mutations (now includes REFERENCE_ID)
    merged_csv = str(Path(output_csv).parent / 'merged_on_nonsyn_counts.csv')
    print(f"Writing merged non-synonymous counts to: {merged_csv}")
    # Merge by (reference_id, aa_mutations) tuple to keep genes separate
    merged: Dict[Tuple[str, str], Dict[str, object]] = {}
    for i, (cons_name, muts, aa_muts, ref_id) in enumerate(hap_rows):
        key = (ref_id, aa_muts)
        entry = merged.setdefault(key, {
            'consensus': [],
            'mutations': set(),
            'reference_id': ref_id,
            'counts': [0] * len(pool_folders)
        })
        entry['consensus'].append(cons_name)
        if muts:
            entry['mutations'].add(muts)
        counts_row = entry['counts']
        for j in range(len(pool_folders)):
            counts_row[j] += hap_counts[i][j]

    with open(merged_csv, 'w') as out:
        header = ['REFERENCE_ID', 'AA_MUTATIONS', 'CONSENSUS_IDS', 'NUC_MUTATIONS'] + pool_names
        out.write(','.join(header) + '\n')
        for (ref_id, aa_muts) in sorted(merged.keys()):
            entry = merged[(ref_id, aa_muts)]
            consensus_ids = ';'.join(sorted(entry['consensus']))
            nuc_muts = ';'.join(sorted(entry['mutations'])) if entry['mutations'] else ''
            row = [ref_id, aa_muts, consensus_ids, nuc_muts] + [str(c) for c in entry['counts']]
            out.write(','.join(row) + '\n')

    metrics["global_match_rate"] = (
        float(metrics["global_umi_matched"] / metrics["global_umi_extracted"])
        if metrics["global_umi_extracted"] > 0
        else 0.0
    )
    metrics_path = str(Path(output_csv).parent / "ngs_count_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Writing ngs count metrics to: {metrics_path}")
    
    print("Done!")
    return True


if __name__ == '__main__':
    # Minimal CLI for ad-hoc execution
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--pools_dir', required=True)
    ap.add_argument('--consensus_dir', required=True)
    ap.add_argument('--variants_dir', required=True)
    ap.add_argument('--probe', required=True)
    ap.add_argument('--umi_len', type=int, required=True)
    ap.add_argument('--umi_loc', choices=['up','down'], required=True)
    ap.add_argument('--reference', required=True, help='Reference FASTA file for amino acid mapping')
    ap.add_argument('--umi_mismatches', type=int, default=0, help='Max UMI mismatches allowed (default: 0)')
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    ok = run_ngs_count(a.pools_dir, a.consensus_dir, a.variants_dir, a.probe, a.umi_len, a.umi_loc, a.output, a.reference,
                       umi_mismatches=a.umi_mismatches)
    sys.exit(0 if ok else 1)
