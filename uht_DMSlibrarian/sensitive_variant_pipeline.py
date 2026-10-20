#!/usr/bin/env python3
"""
Sensitive variant calling pipeline that calls ALL variants, including single mismatches.
Uses custom SAM parsing to bypass bcftools filtering.

Supports multi-reference mode: when multiple references are provided, the pipeline
identifies the best-matching reference for each consensus before variant calling.
"""

import os
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import shutil
import re
from typing import Optional, Tuple, Union

# Import ReferenceManager for multi-reference support
try:
    from .reference_manager import ReferenceManager
except ImportError:
    ReferenceManager = None


def extract_variants_from_sam(sam_file, reference_file_or_seq, output_vcf,
                              reference_id: Optional[str] = None):
    """
    Extract all variants from SAM alignment and create VCF.

    Args:
        sam_file: Path to SAM alignment file
        reference_file_or_seq: Either a path to reference FASTA file or the reference
                               sequence string directly
        output_vcf: Path to output VCF file
        reference_id: Optional reference ID to include in VCF (for multi-ref mode)

    Returns:
        Number of variants found
    """
    # Determine if we have a file path or direct sequence
    if os.path.isfile(reference_file_or_seq):
        # Read reference sequence from file
        with open(reference_file_or_seq, 'r') as f:
            ref_lines = [line.strip() for line in f if not line.startswith('>')]
            ref_seq = ''.join(ref_lines)
    else:
        # Use directly as sequence
        ref_seq = reference_file_or_seq
    
    # Read SAM alignment
    # Keyed by (chr, pos, ref, alt) so an SNV and an indel anchored on the same
    # base are both kept. Only the primary alignment is used.
    variants_dict = {}
    # Reference span covered by the primary alignment (1-based, inclusive);
    # None means the gene was not found in the consensus.
    gene_span = None

    with open(sam_file, 'r') as f:
        for line in f:
            if line.startswith('@'):
                continue

            parts = line.strip().split('\t')
            if len(parts) < 11:
                continue

            # FIX: skip unmapped (0x4), secondary (0x100) and supplementary (0x800)
            # alignments. Supplementary records previously added spurious calls.
            try:
                flag = int(parts[1])
                if flag & (0x4 | 0x100 | 0x800):
                    continue
            except (ValueError, IndexError):
                continue

            cigar = parts[5]
            seq = parts[9].upper()

            # Skip if CIGAR is * (unmapped)
            if cigar == '*':
                continue

            pos = int(parts[3]) - 1  # Convert to 0-based
            seq_pos = 0
            ref_pos = pos
            chrom = parts[2]

            cigar_parts = re.findall(r'(\d+)([MIDNSHP=X])', cigar)

            for length, op in cigar_parts:
                length = int(length)

                if op in 'M=X':  # Match or mismatch
                    for i in range(length):
                        if ref_pos < len(ref_seq) and seq_pos < len(seq):
                            if ref_seq[ref_pos] != seq[seq_pos]:
                                key = (chrom, ref_pos + 1, ref_seq[ref_pos], seq[seq_pos])
                                variants_dict.setdefault(key, {
                                    'chr': chrom,
                                    'pos': ref_pos + 1,
                                    'ref': ref_seq[ref_pos],
                                    'alt': seq[seq_pos],
                                    'qual': 30
                                })
                        ref_pos += 1
                        seq_pos += 1

                elif op in 'DN':  # Deletion from the reference
                    # FIX: previously skipped silently. Emit a VCF deletion
                    # anchored on the preceding reference base (VCF 4.2).
                    if ref_pos > 0:
                        vpos = ref_pos  # 1-based position of anchor base
                        vref = ref_seq[ref_pos - 1:ref_pos + length]
                        valt = ref_seq[ref_pos - 1]
                    else:
                        # Deletion at the first base: anchor on the following base
                        vpos = 1
                        vref = ref_seq[0:length + 1]
                        valt = ref_seq[length]
                    key = (chrom, vpos, vref, valt)
                    variants_dict.setdefault(key, {
                        'chr': chrom, 'pos': vpos, 'ref': vref, 'alt': valt, 'qual': 30
                    })
                    ref_pos += length

                elif op == 'I':  # Insertion relative to the reference
                    # FIX: previously skipped silently. Emit a VCF insertion
                    # anchored on the preceding reference base. Insertions
                    # before the first or after the last gene base are flank.
                    if 0 < ref_pos < len(ref_seq):
                        inserted = seq[seq_pos:seq_pos + length]
                        vref = ref_seq[ref_pos - 1]
                        key = (chrom, ref_pos, vref, vref + inserted)
                        variants_dict.setdefault(key, {
                            'chr': chrom, 'pos': ref_pos, 'ref': vref,
                            'alt': vref + inserted, 'qual': 30
                        })
                    seq_pos += length

                elif op == 'S':  # Soft clip
                    seq_pos += length
                # H and P consume neither sequence

            gene_span = (pos + 1, ref_pos)

    # Convert dictionary to sorted list
    variants = sorted(variants_dict.values(), key=lambda x: (x['chr'], x['pos'], len(x['ref']), len(x['alt'])))
    
    # Write VCF
    with open(output_vcf, 'w') as f:
        # VCF header
        f.write("##fileformat=VCFv4.2\n")
        # Include reference info - use ID if available, otherwise file path if it's a file
        if reference_id:
            f.write(f"##reference={reference_id}\n")
        elif os.path.isfile(reference_file_or_seq):
            f.write(f"##reference={reference_file_or_seq}\n")
        # FIX: record whether/where the gene was found, so downstream steps can
        # tell a WT consensus from one that does not contain the gene.
        if gene_span is None:
            f.write("##gene_status=ABSENT\n")
        else:
            status = "FULL" if gene_span == (1, len(ref_seq)) else "PARTIAL"
            f.write(f"##gene_status={status}\n")
            f.write(f"##gene_span={gene_span[0]}-{gene_span[1]}\n")
        f.write("##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Total Depth\">\n")
        f.write("##INFO=<ID=AD,Number=R,Type=Integer,Description=\"Allelic Depth\">\n")
        f.write("##INFO=<ID=REFERENCE_ID,Number=1,Type=String,Description=\"Reference template ID\">\n")
        f.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        f.write("##FORMAT=<ID=AD,Number=R,Type=Integer,Description=\"Allelic Depth\">\n")
        f.write("##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Total Depth\">\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")

        # Write variants with REFERENCE_ID in INFO field
        for var in variants:
            info_field = "DP=1"
            if reference_id:
                info_field += f";REFERENCE_ID={reference_id}"
            f.write(f"{var['chr']}\t{var['pos']}\t.\t{var['ref']}\t{var['alt']}\t"
                   f"{var['qual']}\tPASS\t{info_field}\tGT:AD:DP\t1/1:0,1:1\n")
    
    return len(variants)


def _read_gene_status(vcf_file):
    with open(vcf_file) as f:
        for line in f:
            if not line.startswith('##'):
                break
            if line.startswith('##gene_status='):
                return line.strip().split('=', 1)[1]
    return None


def realign_partial_globally(consensus_file, ref_seq, vcf_file, reference_id=None):
    """FIX: rescue consensuses where minimap2's primary alignment does not span
    the whole gene (status PARTIAL).

    minimap2 clips or splits alignments around large deletions and sometimes
    around clustered mismatches, so variants outside the aligned span were
    silently treated as WT. Here the full reference is aligned end-to-end
    against the consensus (consensus flanks free, both strands) and the VCF is
    rewritten. If even this alignment is poor, the gene is marked ABSENT.
    """
    from Bio import Align
    from Bio.Seq import Seq

    with open(consensus_file) as f:
        cons = ''.join(l.strip() for l in f if not l.startswith('>')).upper()
    ref_seq = ref_seq.upper()

    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -3
    aligner.open_gap_score = -6
    aligner.extend_gap_score = -1
    aligner.query_end_gap_score = 0    # consensus flanks around the gene are free
    aligner.target_end_gap_score = -6  # unaligned gene ends are penalised

    best = None
    for seq in (cons, str(Seq(cons).reverse_complement())):
        aln = aligner.align(seq, ref_seq)[0]
        if best is None or aln.score > best[0].score:
            best = (aln, seq)
    aln, seq = best

    variants = []
    # Same threshold as the independent check: ~>=85% identity over the gene
    if aln.score >= len(ref_seq):
        t_blocks, q_blocks = aln.aligned  # target = consensus, query = reference
        if q_blocks[0][0] > 0:  # gene start missing: deletion anchored on next base
            n = q_blocks[0][0]
            variants.append((1, ref_seq[:n + 1], ref_seq[n]))
        for i in range(len(t_blocks)):
            ts, te = t_blocks[i]
            qs, qe = q_blocks[i]
            for k in range(qe - qs):
                if seq[ts + k] != ref_seq[qs + k]:
                    variants.append((qs + k + 1, ref_seq[qs + k], seq[ts + k]))
            if i + 1 < len(t_blocks):
                dq = q_blocks[i + 1][0] - qe
                dt = t_blocks[i + 1][0] - te
                if dq > 0:
                    variants.append((qe, ref_seq[qe - 1:qe + dq], ref_seq[qe - 1]))
                if dt > 0:
                    anchor = qe + dq
                    variants.append((anchor, ref_seq[anchor - 1], ref_seq[anchor - 1] + seq[te:te + dt]))
        if q_blocks[-1][1] < len(ref_seq):  # gene end missing
            qe = q_blocks[-1][1]
            variants.append((qe, ref_seq[qe - 1:], ref_seq[qe - 1]))
        status = 'FULL_REALIGNED'
    else:
        status = 'ABSENT'

    chrom = reference_id or 'reference'
    with open(vcf_file) as f:
        header = [l for l in f if l.startswith('#')]
    with open(vcf_file, 'w') as f:
        for l in header:
            if l.startswith('##gene_status='):
                l = f"##gene_status={status}\n##minimap2_gene_status=PARTIAL\n"
            elif l.startswith('##gene_span=') and status == 'ABSENT':
                continue
            f.write(l)
        for pos, r, a in sorted(variants):
            info = "DP=1" + (f";REFERENCE_ID={reference_id}" if reference_id else "")
            f.write(f"{chrom}\t{pos}\t.\t{r}\t{a}\t30\tPASS\t{info}\tGT:AD:DP\t1/1:0,1:1\n")
    return len(variants)

def align_and_call_variants_sensitive(consensus_file, reference_file_or_manager, output_dir):
    """
    Align consensus to reference and call ALL variants using sensitive method.

    Args:
        consensus_file: Path to consensus FASTA file
        reference_file_or_manager: Either a path to reference FASTA file (single ref mode)
                                   or a ReferenceManager instance (multi-ref mode)
        output_dir: Output directory for VCF files

    Returns:
        VCF file path on success, or error message string on failure.
        In multi-ref mode, also stores reference_id in the VCF INFO field.
    """
    try:
        consensus_name = Path(consensus_file).stem
        vcf_file = os.path.join(output_dir, f"{consensus_name}.vcf")

        # Skip if VCF already exists
        if os.path.exists(vcf_file):
            return vcf_file

        # Determine reference mode
        is_multi_ref = (ReferenceManager is not None and
                       isinstance(reference_file_or_manager, ReferenceManager))

        # Multi-reference mode: identify best reference first
        if is_multi_ref:
            ref_manager = reference_file_or_manager
            # Read consensus sequence for reference identification
            with open(consensus_file, 'r') as f:
                lines = [line.strip() for line in f if not line.startswith('>')]
                consensus_seq = ''.join(lines)

            # Identify best matching reference
            best_ref_id, score, is_ambiguous = ref_manager.identify_best_reference(consensus_seq)

            if best_ref_id == "UNASSIGNED":
                return f"No good reference match for {consensus_name} (score: {score})"

            # Get the reference sequence for this consensus
            ref_seq = ref_manager.get_reference_sequence(best_ref_id)

            # Create temp reference file for minimap2
            ref_temp_file = ref_manager.get_reference_file(best_ref_id)
            try:
                # Step 1: Align consensus to identified reference using minimap2
                sam_file = os.path.join(output_dir, f"{consensus_name}.sam")

                minimap2_cmd = [
                    "minimap2",
                    "-ax", "map-ont",
                    "-t", "1",
                    ref_temp_file,
                    consensus_file
                ]

                with open(sam_file, 'w') as sam_out:
                    subprocess.run(minimap2_cmd, check=True, stdout=sam_out,
                                 stderr=subprocess.PIPE, text=True)

                # Step 2: Extract variants - pass reference sequence and ID directly
                variant_count = extract_variants_from_sam(
                    sam_file, ref_seq, vcf_file, reference_id=best_ref_id
                )
                if _read_gene_status(vcf_file) == 'PARTIAL':
                    variant_count = realign_partial_globally(
                        consensus_file, ref_seq, vcf_file, reference_id=best_ref_id
                    )

                # Clean up intermediate files
                os.remove(sam_file)

            finally:
                # Clean up temp reference file
                if ref_temp_file and os.path.exists(ref_temp_file):
                    os.remove(ref_temp_file)

            # FIX: the VCF already carries ##reference and ##gene_status. The old
            # WT placeholder overwrote it, turning gene-absent consensuses into WT.

            return vcf_file

        else:
            # Single reference mode (backward compatible)
            reference_file = reference_file_or_manager

            # Step 1: Align consensus to reference using minimap2
            sam_file = os.path.join(output_dir, f"{consensus_name}.sam")

            # Run minimap2 alignment
            minimap2_cmd = [
                "minimap2",
                "-ax", "map-ont",
                "-t", "1",
                reference_file,
                consensus_file
            ]

            with open(sam_file, 'w') as sam_out:
                subprocess.run(minimap2_cmd, check=True, stdout=sam_out,
                             stderr=subprocess.PIPE, text=True)

            # Step 2: Extract variants directly from SAM
            variant_count = extract_variants_from_sam(sam_file, reference_file, vcf_file)
            if _read_gene_status(vcf_file) == 'PARTIAL':
                with open(reference_file) as rf:
                    ref_str = ''.join(l.strip() for l in rf if not l.startswith('>'))
                variant_count = realign_partial_globally(consensus_file, ref_str, vcf_file)

            # Clean up intermediate files
            os.remove(sam_file)

            if variant_count == 0:
                return f"No variants found for {consensus_name}"

            return vcf_file

    except Exception as e:
        return f"Error processing {consensus_name}: {str(e)}"


def _write_empty_vcf_with_ref(vcf_file: str, reference_id: str) -> None:
    """Write a minimal VCF file with reference info for WT consensus (no variants)."""
    with open(vcf_file, 'w') as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write(f"##reference={reference_id}\n")
        f.write("##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Total Depth\">\n")
        f.write("##INFO=<ID=REFERENCE_ID,Number=1,Type=String,Description=\"Reference template ID\">\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
        # Write a single placeholder entry marking this as WT for this reference
        # Using position 0 as a marker (will be filtered out in downstream processing)
        f.write(f"{reference_id}\t0\t.\t.\t.\t.\tWT\tREFERENCE_ID={reference_id}\t.\t.\n")

def process_consensus_file_sensitive(consensus_file, reference_file, output_dir):
    """Process a single consensus file to generate VCF using sensitive method."""
    consensus_name = Path(consensus_file).stem
    
    try:
        # Align and call variants
        vcf_result = align_and_call_variants_sensitive(consensus_file, reference_file, output_dir)
        
        if isinstance(vcf_result, str) and vcf_result.endswith('.vcf'):
            # Count variants in the VCF
            with open(vcf_result, 'r') as f:
                variant_count = sum(1 for line in f if not line.startswith('#'))
            return f"SUCCESS: {consensus_name} ({variant_count} variants)"
        else:
            return f"FAILED: {consensus_name} - {vcf_result}"
            
    except Exception as e:
        return f"ERROR: {consensus_name} - {str(e)}"

def combine_vcf_files(vcf_dir, output_vcf):
    """
    Combine all VCF files into a single VCF with one entry per consensus.

    Preserves REFERENCE_ID from individual VCFs (for multi-reference mode).
    """
    try:
        # Get list of VCF files
        vcf_files = []
        for file in os.listdir(vcf_dir):
            # Skip hidden files such as macOS AppleDouble '._' files
            if file.endswith('.vcf') and not file.startswith('.'):
                vcf_files.append(os.path.join(vcf_dir, file))

        if not vcf_files:
            print("No VCF files found to combine")
            return False

        print(f"Combining {len(vcf_files)} VCF files...")

        # Create combined VCF header
        with open(output_vcf, 'w') as outfile:
            # Write VCF header
            outfile.write("##fileformat=VCFv4.2\n")
            outfile.write("##INFO=<ID=CLUSTER,Number=1,Type=String,Description=\"Cluster ID\">\n")
            outfile.write("##INFO=<ID=VARIANT_COUNT,Number=1,Type=Integer,Description=\"Number of variants in this consensus\">\n")
            outfile.write("##INFO=<ID=REFERENCE_ID,Number=1,Type=String,Description=\"Reference template ID\">\n")
            outfile.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

            # Process each VCF file
            for vcf_file in vcf_files:
                cluster_name = Path(vcf_file).stem

                try:
                    # Read VCF file and extract variants and reference ID
                    reference_id = None
                    variants = []
                    is_wt = False

                    with open(vcf_file, 'r') as infile:
                        for line in infile:
                            if line.startswith('##reference='):
                                # Extract reference ID from header
                                reference_id = line.strip().split('=', 1)[1]
                            elif line.startswith('#') or line.strip() == '':
                                continue
                            else:
                                parts = line.strip().split('\t')
                                if len(parts) >= 8:
                                    # Check for WT marker entries (position 0)
                                    if parts[1] == '0' and parts[6] == 'WT':
                                        # Extract REFERENCE_ID from WT marker
                                        for info_item in parts[7].split(';'):
                                            if info_item.startswith('REFERENCE_ID='):
                                                reference_id = info_item.split('=')[1]
                                        is_wt = True
                                        continue

                                    # Add cluster information to INFO field
                                    info_parts = parts[7].split(';')
                                    info_parts.append(f"CLUSTER={cluster_name}")

                                    # Ensure REFERENCE_ID is present if we know it
                                    has_ref_id = any(p.startswith('REFERENCE_ID=') for p in info_parts)
                                    if reference_id and not has_ref_id:
                                        info_parts.append(f"REFERENCE_ID={reference_id}")

                                    parts[7] = ';'.join(info_parts)
                                    variants.append('\t'.join(parts[:8]))  # Only first 8 columns

                    # Write variants to combined VCF
                    for variant in variants:
                        outfile.write(variant + '\n')

                    # For WT clusters (no variants), write a WT marker entry
                    # so the analysis stage knows about this cluster and its reference
                    if not variants and is_wt and reference_id:
                        wt_info = f"CLUSTER={cluster_name};REFERENCE_ID={reference_id};WT=true"
                        outfile.write(f"{reference_id}\t0\t.\t.\t.\t.\tWT\t{wt_info}\n")

                except Exception as e:
                    print(f"Error processing {vcf_file}: {e}")
                    continue

        print(f"Combined VCF written to: {output_vcf}")
        return True

    except Exception as e:
        print(f"Error combining VCF files: {e}")
        return False

def main():
    # Configuration
    consensus_dir = '/Volumes/Matt115A1TB_1/temp-dict-epgh11/simple_consensus_results'
    reference_file = 'GH11_epPCR/reference.fasta'
    output_dir = '/Volumes/Matt115A1TB_1/temp-dict-epgh11/sensitive_variant_results'
    combined_vcf = '/Volumes/Matt115A1TB_1/temp-dict-epgh11/combined_variants_sensitive.vcf'
    
    # Use 4 threads for parallel processing
    max_workers = 4
    
    print(f"Starting SENSITIVE variant calling pipeline...")
    print(f"This will call ALL variants, including single mismatches!")
    print(f"Consensus directory: {consensus_dir}")
    print(f"Reference file: {reference_file}")
    print(f"Output directory: {output_dir}")
    print(f"Combined VCF: {combined_vcf}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of consensus files
    consensus_files = []
    for file in os.listdir(consensus_dir):
        # Skip hidden files such as macOS AppleDouble '._' files
        if file.endswith('_consensus.fasta') and not file.startswith('.'):
            consensus_files.append(os.path.join(consensus_dir, file))
    
    total_consensus = len(consensus_files)
    print(f"Found {total_consensus:,} consensus files to process")
    
    # Track progress
    success_count = 0
    failed_count = 0
    start_time = time.time()
    
    # Thread-safe progress tracking
    progress_lock = threading.Lock()
    
    def update_progress():
        nonlocal success_count, failed_count
        with progress_lock:
            processed = success_count + failed_count
            elapsed_time = time.time() - start_time
            rate = processed / elapsed_time if elapsed_time > 0 else 0
            
            # Calculate ETA
            if rate > 0:
                remaining = total_consensus - processed
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
            progress_percent = (processed / total_consensus) * 100
            bar_length = 50
            filled_length = int(bar_length * processed // total_consensus)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            # Print progress line
            progress_line = (f"\rProgress: |{bar}| {progress_percent:.1f}% "
                           f"({processed:,}/{total_consensus:,}) | "
                           f"Success: {success_count:,} | Failed: {failed_count:,} | "
                           f"Rate: {rate:.1f} consensus/s | ETA: {eta_str} | "
                           f"Elapsed: {elapsed_str}")
            
            print(progress_line, end='', flush=True)
    
    # Process consensus files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_consensus_file_sensitive, consensus_file, reference_file, output_dir) for consensus_file in consensus_files]
        
        # Process completed jobs
        for future in futures:
            result = future.result()
            
            with progress_lock:
                if "SUCCESS" in result:
                    success_count += 1
                else:
                    failed_count += 1
                    if failed_count <= 10:  # Only show first 10 failures
                        print(f"\nFailed: {result}")
            
            update_progress()
    
    # Combine VCF files
    print(f"\n\nCombining VCF files...")
    combine_success = combine_vcf_files(output_dir, combined_vcf)
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\nSensitive variant calling pipeline completed!")
    print(f"Total consensus processed: {total_consensus:,}")
    print(f"Successful: {success_count:,}")
    print(f"Failed: {failed_count:,}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average rate: {total_consensus/total_time:.1f} consensus/second")
    print(f"Individual VCF files: {output_dir}")
    if combine_success:
        print(f"Combined VCF: {combined_vcf}")
    else:
        print("Failed to create combined VCF")

if __name__ == "__main__":
    main()
