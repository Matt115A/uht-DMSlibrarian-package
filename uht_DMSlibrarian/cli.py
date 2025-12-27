#!/usr/bin/env python3
"""
UMIC-seq PacBio Pipeline - Main Entry Point
Complete pipeline for processing PacBio data from raw FASTQ to detailed mutation analysis.

This script orchestrates the entire UMIC-seq pipeline:
1. UMI extraction from raw PacBio reads
2. Clustering of similar UMIs
3. Consensus generation using abpoa
4. Variant calling with sensitive parameters
5. Detailed mutation analysis and CSV output

Usage:
    umic-seq-pacbio --help
    umic-seq-pacbio all --input raw_reads.fastq.gz --probe probe.fasta --reference reference.fasta --output_dir /path/to/output
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
from Bio import SeqIO

# Import from package
from . import simple_consensus_pipeline
from . import sensitive_variant_pipeline
from . import vcf2csv_detailed
from . import ngs_count
from . import fitness_analysis


def run_command(cmd, description="", check=True):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=check, capture_output=False, text=True)
        elapsed = time.time() - start_time
        print(f"\n✓ COMPLETED: {description} ({elapsed:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ FAILED: {description} ({elapsed:.1f}s)")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def run_umi_extraction(args):
    """Run UMI extraction step."""
    cmd = [
        sys.executable, "-m", "uht_DMSlibrarian.UMIC_seq", "UMIextract",
        "-i", args.input,
        "-o", args.output,
        "--probe", args.probe,
        "--umi_loc", args.umi_loc,
        "--umi_len", str(args.umi_len),
        "--min_probe_score", str(getattr(args, 'min_probe_score', 15))
    ]
    
    return run_command(cmd, "UMI Extraction")


def run_clustering(args):
    """Run clustering step."""
    # Determine which clustering method to use
    use_fast = args.fast and not args.slow
    
    # Get probe file if available (for orientation normalization)
    probe_file = getattr(args, 'probe', None)
    
    if use_fast:
        # Use fast CD-HIT clustering
        cmd = [
            sys.executable, "-m", "uht_DMSlibrarian.UMIC_seq", "clusterfull_fast",
            "-i", args.input_umi,
            "-o", args.output_dir,
            "--reads", args.input_reads,
            "--identity", str(args.identity),
            "--size_thresh", str(args.size_thresh)
        ]
        # Add probe if provided
        if probe_file:
            cmd.extend(["--probe", probe_file])
        description = "Fast UMI Clustering (CD-HIT)"
    else:
        # Use slow alignment-based clustering
        aln_thresh_int = int(args.aln_thresh * 100)
        cmd = [
            sys.executable, "-m", "uht_DMSlibrarian.UMIC_seq", "clusterfull",
            "-i", args.input_umi,
            "-o", args.output_dir,
            "--reads", args.input_reads,
            "--aln_thresh", str(aln_thresh_int),
            "--size_thresh", str(args.size_thresh)
        ]
        # Add probe if provided
        if probe_file:
            cmd.extend(["--probe", probe_file])
        description = "Slow UMI Clustering (alignment-based)"
    
    return run_command(cmd, description)


def run_consensus_generation(args):
    """Run consensus generation step."""
    cluster_files_dir = args.input_dir
    output_dir = args.output_dir
    max_reads = args.max_reads
    max_workers = args.max_workers
    
    print(f"Starting simple consensus pipeline...")
    print(f"Cluster files directory: {cluster_files_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max reads per consensus: {max_reads}")
    print(f"Max workers: {max_workers}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of cluster files
    cluster_files = []
    for file in os.listdir(cluster_files_dir):
        if file.endswith('.fasta') and file.startswith('cluster_'):
            cluster_files.append(os.path.join(cluster_files_dir, file))
    
    total_clusters = len(cluster_files)
    print(f"Found {total_clusters:,} cluster files to process")
    
    if total_clusters == 0:
        print("No cluster files found!")
        return False
    
    # Track progress
    success_count = 0
    failed_count = 0
    start_time = time.time()
    
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import gc
    
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
                remaining = total_clusters - processed
                eta_seconds = remaining / rate
                eta_minutes = eta_seconds / 60
                
                if eta_minutes >= 1:
                    eta_str = f"{eta_minutes:.1f}m"
                else:
                    eta_str = f"{eta_seconds:.0f}s"
            else:
                eta_str = "unknown"
            
            # Progress percentage
            progress_percent = (processed / total_clusters) * 100
            
            # Print progress
            progress_line = (f"\rProgress: {progress_percent:.1f}% "
                           f"({processed:,}/{total_clusters:,}) | "
                           f"Success: {success_count:,} | Failed: {failed_count:,} | "
                           f"Rate: {rate:.1f} clusters/s | ETA: {eta_str}")
            
            print(progress_line, end='', flush=True)
            
            # Periodic garbage collection every 5000 clusters
            if processed > 0 and processed % 5000 == 0:
                gc.collect()
                print(f"\n[GC at {processed:,} clusters]", flush=True)
    
    # Process clusters in parallel with batched submission
    BATCH_SIZE = 1000  # Submit 1000 at a time to avoid memory buildup
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch_start in range(0, total_clusters, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_clusters)
            batch_files = cluster_files[batch_start:batch_end]
            
            # Submit batch
            futures = {executor.submit(simple_consensus_pipeline.process_cluster_simple, cf, output_dir, max_reads): cf 
                      for cf in batch_files}
            
            # Process as they complete
            for future in as_completed(futures):
                result = future.result()
                
                with progress_lock:
                    if "SUCCESS" in result:
                        success_count += 1
                    else:
                        failed_count += 1
                        if failed_count <= 10:  # Only show first 10 failures
                            print(f"\nFailed: {result}")
                
                update_progress()
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n\nConsensus generation completed!")
    print(f"Total clusters processed: {total_clusters:,}")
    print(f"Successful: {success_count:,}")
    print(f"Failed: {failed_count:,}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Output directory: {output_dir}")
    
    return success_count > 0


def run_variant_calling(args):
    """Run variant calling step."""
    consensus_dir = args.input_dir
    reference_file = args.reference
    output_dir = args.output_dir
    combined_vcf = args.combined_vcf
    max_workers = args.max_workers
    
    print(f"\n{'='*60}")
    print(f"RUNNING: Variant Calling")
    print(f"{'='*60}")
    
    start_time = time.time()
    
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
        if file.endswith('_consensus.fasta'):
            consensus_files.append(os.path.join(consensus_dir, file))
    
    total_consensus = len(consensus_files)
    print(f"Found {total_consensus:,} consensus files to process\n")
    
    # Track progress
    success_count = 0
    failed_count = 0
    
    # Thread-safe progress tracking
    import threading
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
    
    # Process consensus files in parallel with batched submission
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(sensitive_variant_pipeline.process_consensus_file_sensitive, 
                                  cf, reference_file, output_dir): cf 
                  for cf in consensus_files}
        
        # Process as they complete
        for future in as_completed(futures):
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
    combine_success = sensitive_variant_pipeline.combine_vcf_files(output_dir, combined_vcf)
    
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
    
    print(f"\n✓ COMPLETED: Variant Calling ({total_time:.1f}s)")
    
    return success_count > 0


def run_analysis(args):
    """Run detailed analysis step."""
    return vcf2csv_detailed.vcf_to_csv_detailed(args.input_vcf, args.reference, args.output)


def count_fasta_sequences(fasta_file):
    """Count sequences in a FASTA file."""
    try:
        count = 0
        with open(fasta_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    count += 1
        return count
    except Exception:
        return None


def count_files_in_directory(directory, pattern_start="", pattern_end=""):
    """Count files in a directory matching patterns."""
    try:
        count = 0
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.startswith(pattern_start) and file.endswith(pattern_end):
                    count += 1
        return count
    except Exception:
        return None


def count_vcf_variants(vcf_file):
    """Count variant lines in a VCF file (excluding header)."""
    try:
        count = 0
        with open(vcf_file, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    count += 1
        return count
    except Exception:
        return None


def get_file_size_mb(file_path):
    """Get file size in MB."""
    try:
        if os.path.exists(file_path):
            return os.path.getsize(file_path) / (1024 * 1024)
        return None
    except Exception:
        return None


def format_duration(seconds):
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        if minutes < 1:
            return f"{hours:.1f} hours"
        else:
            return f"{int(hours)} hours {int(minutes)} minutes"


def write_pipeline_report(args, pipeline_stats, start_time, end_time, report_file):
    """Write a comprehensive pipeline execution report."""
    duration = end_time - start_time
    duration_str = format_duration(duration)
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("UMIC-seq PACBIO PIPELINE EXECUTION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Execution information
        f.write("EXECUTION INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total duration: {duration_str}\n")
        f.write(f"Status: {'SUCCESS' if pipeline_stats.get('success', False) else 'FAILED'}\n")
        f.write("\n")
        
        # Input parameters
        f.write("INPUT PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Input FASTQ: {args.input}\n")
        f.write(f"Probe file: {args.probe}\n")
        f.write(f"Reference file: {args.reference}\n")
        f.write(f"Output directory: {args.output_dir}\n")
        f.write(f"UMI length: {args.umi_len}\n")
        f.write(f"UMI location: {args.umi_loc}\n")
        f.write(f"Min probe score: {getattr(args, 'min_probe_score', 15)}\n")
        f.write(f"Clustering method: {'Fast (CD-HIT)' if getattr(args, 'fast', True) else 'Slow (alignment-based)'}\n")
        if getattr(args, 'fast', True):
            f.write(f"Identity threshold: {getattr(args, 'identity', 0.90)}\n")
        else:
            f.write(f"Alignment threshold: {getattr(args, 'aln_thresh', 0.47)}\n")
        f.write(f"Size threshold: {args.size_thresh}\n")
        f.write(f"Max reads per consensus: {args.max_reads}\n")
        f.write(f"Max workers: {args.max_workers}\n")
        f.write("\n")
        
        # Pipeline statistics
        f.write("PIPELINE STATISTICS\n")
        f.write("-" * 80 + "\n")
        
        # Step 1: UMI Extraction
        f.write("\n1. UMI EXTRACTION\n")
        umi_count = pipeline_stats.get('umis_extracted')
        if umi_count is not None:
            f.write(f"   UMIs extracted: {umi_count:,}\n")
        else:
            f.write("   UMIs extracted: N/A\n")
        umi_file_size = pipeline_stats.get('umi_file_size_mb')
        if umi_file_size is not None:
            f.write(f"   UMI file size: {umi_file_size:.2f} MB\n")
        
        # Step 2: Clustering
        f.write("\n2. CLUSTERING\n")
        cluster_count = pipeline_stats.get('clusters_generated')
        if cluster_count is not None:
            f.write(f"   Clusters generated: {cluster_count:,}\n")
        else:
            f.write("   Clusters generated: N/A\n")
        clustering_time = pipeline_stats.get('clustering_time')
        if clustering_time is not None:
            f.write(f"   Clustering duration: {format_duration(clustering_time)}\n")
        
        # Step 3: Consensus Generation
        f.write("\n3. CONSENSUS GENERATION\n")
        consensus_count = pipeline_stats.get('consensus_sequences')
        if consensus_count is not None:
            f.write(f"   Consensus sequences: {consensus_count:,}\n")
        else:
            f.write("   Consensus sequences: N/A\n")
        consensus_success = pipeline_stats.get('consensus_success')
        consensus_failed = pipeline_stats.get('consensus_failed')
        if consensus_success is not None:
            f.write(f"   Successful: {consensus_success:,}\n")
        if consensus_failed is not None:
            f.write(f"   Failed: {consensus_failed:,}\n")
        consensus_time = pipeline_stats.get('consensus_time')
        if consensus_time is not None:
            f.write(f"   Consensus generation duration: {format_duration(consensus_time)}\n")
        
        # Step 4: Variant Calling
        f.write("\n4. VARIANT CALLING\n")
        variant_count = pipeline_stats.get('variants_called')
        if variant_count is not None:
            f.write(f"   Total variants: {variant_count:,}\n")
        else:
            f.write("   Total variants: N/A\n")
        variant_success = pipeline_stats.get('variant_success')
        variant_failed = pipeline_stats.get('variant_failed')
        if variant_success is not None:
            f.write(f"   Successful: {variant_success:,}\n")
        if variant_failed is not None:
            f.write(f"   Failed: {variant_failed:,}\n")
        variant_time = pipeline_stats.get('variant_time')
        if variant_time is not None:
            f.write(f"   Variant calling duration: {format_duration(variant_time)}\n")
        
        # Step 5: Analysis
        f.write("\n5. ANALYSIS\n")
        analysis_file_size = pipeline_stats.get('analysis_file_size_mb')
        if analysis_file_size is not None:
            f.write(f"   Analysis file size: {analysis_file_size:.2f} MB\n")
        else:
            f.write("   Analysis file: Generated\n")
        
        # Output files
        f.write("\nOUTPUT FILES\n")
        f.write("-" * 80 + "\n")
        output_files = pipeline_stats.get('output_files', {})
        for file_desc, file_path in output_files.items():
            if file_path and os.path.exists(file_path):
                size_mb = get_file_size_mb(file_path)
                if size_mb is not None:
                    f.write(f"{file_desc}: {file_path} ({size_mb:.2f} MB)\n")
                else:
                    f.write(f"{file_desc}: {file_path}\n")
            elif file_path:
                f.write(f"{file_desc}: {file_path} (not found)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Report generated by uht-DMSlibrarian\n")
        f.write("=" * 80 + "\n")


def run_full_pipeline(args):
    """Run the complete pipeline."""
    start_time = time.time()
    pipeline_stats = {'success': False}
    
    print(f"\n{'='*80}")
    print("STARTING COMPLETE UMIC-seq PACBIO PIPELINE")
    print(f"{'='*80}")
    print(f"Input FASTQ: {args.input}")
    print(f"Probe file: {args.probe}")
    print(f"Reference file: {args.reference}")
    print(f"Output directory: {args.output_dir}")
    print(f"UMI length: {args.umi_len}")
    print(f"Alignment threshold: {getattr(args, 'aln_thresh', 0.47)}")
    print(f"Size threshold: {args.size_thresh}")
    print(f"Max reads per consensus: {args.max_reads}")
    print(f"Max workers: {args.max_workers}")
    print(f"{'='*80}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: UMI Extraction
    step_start = time.time()
    umi_file = os.path.join(args.output_dir, "ExtractedUMIs.fasta")
    extract_args = argparse.Namespace(
        input=args.input,
        probe=args.probe,
        umi_len=args.umi_len,
        umi_loc=args.umi_loc,
        output=umi_file,
        min_probe_score=getattr(args, 'min_probe_score', 15)
    )
    
    if not run_umi_extraction(extract_args):
        print("Pipeline failed at UMI extraction step")
        end_time = time.time()
        pipeline_stats['success'] = False
        if getattr(args, 'report', None):
            write_pipeline_report(args, pipeline_stats, start_time, end_time, args.report)
        return False
    
    # Collect UMI extraction statistics
    pipeline_stats['umis_extracted'] = count_fasta_sequences(umi_file)
    pipeline_stats['umi_file_size_mb'] = get_file_size_mb(umi_file)
    
    # Step 2: Clustering
    clustering_start = time.time()
    cluster_dir = os.path.join(args.output_dir, "clusters")
    cluster_args = argparse.Namespace(
        input_umi=umi_file,
        input_reads=args.input,
        aln_thresh=getattr(args, 'aln_thresh', 0.47),
        identity=getattr(args, 'identity', 0.90),
        size_thresh=args.size_thresh,
        output_dir=cluster_dir,
        probe=args.probe,  # Pass probe file for orientation normalization
        fast=getattr(args, 'fast', True),
        slow=getattr(args, 'slow', False)
    )
    
    if not run_clustering(cluster_args):
        print("Pipeline failed at clustering step")
        end_time = time.time()
        pipeline_stats['success'] = False
        if getattr(args, 'report', None):
            write_pipeline_report(args, pipeline_stats, start_time, end_time, args.report)
        return False
    
    # Collect clustering statistics
    pipeline_stats['clustering_time'] = time.time() - clustering_start
    pipeline_stats['clusters_generated'] = count_files_in_directory(cluster_dir, "cluster_", ".fasta")
    
    # Step 3: Consensus Generation
    consensus_start = time.time()
    consensus_dir = os.path.join(args.output_dir, "consensus")
    consensus_args = argparse.Namespace(
        input_dir=cluster_dir,
        output_dir=consensus_dir,
        max_reads=args.max_reads,
        max_workers=args.max_workers
    )
    
    if not run_consensus_generation(consensus_args):
        print("Pipeline failed at consensus generation step")
        end_time = time.time()
        pipeline_stats['success'] = False
        if getattr(args, 'report', None):
            write_pipeline_report(args, pipeline_stats, start_time, end_time, args.report)
        return False
    
    # Collect consensus statistics
    pipeline_stats['consensus_time'] = time.time() - consensus_start
    pipeline_stats['consensus_sequences'] = count_files_in_directory(consensus_dir, "", "_consensus.fasta")
    
    # Step 4: Variant Calling
    variant_start = time.time()
    variant_dir = os.path.join(args.output_dir, "variants")
    combined_vcf = os.path.join(args.output_dir, "combined_variants.vcf")
    variant_args = argparse.Namespace(
        input_dir=consensus_dir,
        reference=args.reference,
        output_dir=variant_dir,
        combined_vcf=combined_vcf,
        max_workers=args.max_workers
    )
    
    if not run_variant_calling(variant_args):
        print("Pipeline failed at variant calling step")
        end_time = time.time()
        pipeline_stats['success'] = False
        if getattr(args, 'report', None):
            write_pipeline_report(args, pipeline_stats, start_time, end_time, args.report)
        return False
    
    # Collect variant calling statistics
    pipeline_stats['variant_time'] = time.time() - variant_start
    pipeline_stats['variants_called'] = count_vcf_variants(combined_vcf) if os.path.exists(combined_vcf) else None
    
    # Step 5: Detailed Analysis
    analysis_file = os.path.join(args.output_dir, "detailed_mutations.csv")
    analysis_args = argparse.Namespace(
        input_vcf=combined_vcf,
        reference=args.reference,
        output=analysis_file
    )
    
    if not run_analysis(analysis_args):
        print("Pipeline failed at analysis step")
        end_time = time.time()
        pipeline_stats['success'] = False
        if getattr(args, 'report', None):
            write_pipeline_report(args, pipeline_stats, start_time, end_time, args.report)
        return False
    
    # Collect analysis statistics
    pipeline_stats['analysis_file_size_mb'] = get_file_size_mb(analysis_file)
    
    # Pipeline completed successfully
    end_time = time.time()
    pipeline_stats['success'] = True
    
    # Store output file paths
    pipeline_stats['output_files'] = {
        'UMI file': umi_file,
        'Cluster directory': cluster_dir,
        'Consensus directory': consensus_dir,
        'Variant directory': variant_dir,
        'Combined VCF': combined_vcf,
        'Analysis file': analysis_file
    }
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"Final output: {analysis_file}")
    print(f"Cluster directory: {cluster_dir}")
    print(f"Consensus directory: {consensus_dir}")
    print(f"Variant directory: {variant_dir}")
    print(f"Combined VCF: {combined_vcf}")
    print(f"{'='*80}")
    
    # Generate report if requested
    if getattr(args, 'report', None):
        print(f"\nGenerating pipeline report: {args.report}")
        write_pipeline_report(args, pipeline_stats, start_time, end_time, args.report)
        print(f"Report saved to: {args.report}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="UMIC-seq PacBio Pipeline - Complete pipeline for processing PacBio data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  umic-seq-pacbio all --input raw_reads.fastq.gz --probe probe.fasta --reference reference.fasta --output_dir /path/to/output
  
  # Run individual steps
  umic-seq-pacbio extract --input raw_reads.fastq.gz --probe probe.fasta --output umis.fasta
  umic-seq-pacbio cluster --input_umi umis.fasta --input_reads raw_reads.fastq.gz --output_dir clusters/
  umic-seq-pacbio consensus --input_dir clusters/ --output_dir consensus/
  umic-seq-pacbio variants --input_dir consensus/ --reference reference.fasta --output_dir variants/
  umic-seq-pacbio analyze --input_vcf combined.vcf --reference reference.fasta --output detailed.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Pipeline step to run')
    
    # Full pipeline command
    all_parser = subparsers.add_parser('all', help='Run complete pipeline')
    all_parser.add_argument('--input', required=True, help='Input FASTQ file (can be .gz)')
    all_parser.add_argument('--probe', required=True, help='Probe FASTA file')
    all_parser.add_argument('--reference', required=True, help='Reference FASTA file')
    all_parser.add_argument('--output_dir', required=True, help='Output directory')
    all_parser.add_argument('--umi_len', type=int, default=52, help='UMI length (default: 52)')
    all_parser.add_argument('--aln_thresh', type=float, default=0.47, help='Alignment threshold for slow clustering (default: 0.47)')
    all_parser.add_argument('--identity', type=float, default=0.90, help='Sequence identity for fast clustering (default: 0.90)')
    all_parser.add_argument('--size_thresh', type=int, default=10, help='Size threshold (default: 10)')
    all_parser.add_argument('--max_reads', type=int, default=20, help='Max reads per consensus (default: 20)')
    all_parser.add_argument('--max_workers', type=int, default=4, help='Max parallel workers (default: 4)')
    all_parser.add_argument('--fast', action='store_true', default=True, help='Use fast CD-HIT clustering (default: True)')
    all_parser.add_argument('--slow', action='store_true', help='Use slow alignment-based clustering')
    all_parser.add_argument('--umi_loc', type=str, default='up', choices=['up', 'down'], help='UMI location relative to probe (up or down, default: up)')
    all_parser.add_argument('--min_probe_score', type=int, default=15, help='Minimal alignment score of probe for processing (default: 15)')
    all_parser.add_argument('--report', help='Path to output report file. Generates a summary report with execution time, parameters, and statistics.')
    
    # Individual step commands
    extract_parser = subparsers.add_parser('extract', help='Extract UMIs from raw reads')
    extract_parser.add_argument('--input', required=True, help='Input FASTQ file (can be .gz)')
    extract_parser.add_argument('--probe', required=True, help='Probe FASTA file')
    extract_parser.add_argument('--umi_len', type=int, default=52, help='UMI length (default: 52)')
    extract_parser.add_argument('--output', required=True, help='Output FASTA file')
    extract_parser.add_argument('--umi_loc', type=str, default='up', choices=['up', 'down'], help='UMI location relative to probe (default: up)')
    extract_parser.add_argument('--min_probe_score', type=int, default=15, help='Minimal alignment score of probe for processing (default: 15)')
    
    cluster_parser = subparsers.add_parser('cluster', help='Cluster UMIs')
    cluster_parser.add_argument('--input_umi', required=True, help='Input UMI FASTA file')
    cluster_parser.add_argument('--input_reads', required=True, help='Input reads FASTQ file (can be .gz)')
    cluster_parser.add_argument('--probe', help='Probe sequence file (fasta format). NOTE: Orientation is determined during UMI extraction and stored in UMI headers. This argument is deprecated but kept for backward compatibility.')
    cluster_parser.add_argument('--aln_thresh', type=float, default=0.47, help='Alignment threshold for slow method (default: 0.47)')
    cluster_parser.add_argument('--identity', type=float, default=0.90, help='Sequence identity for fast method (default: 0.90)')
    cluster_parser.add_argument('--size_thresh', type=int, default=10, help='Size threshold (default: 10)')
    cluster_parser.add_argument('--output_dir', required=True, help='Output directory for clusters')
    cluster_parser.add_argument('--fast', action='store_true', default=True, help='Use fast CD-HIT clustering (default: True)')
    cluster_parser.add_argument('--slow', action='store_true', help='Use slow alignment-based clustering')
    
    consensus_parser = subparsers.add_parser('consensus', help='Generate consensus sequences')
    consensus_parser.add_argument('--input_dir', required=True, help='Input cluster directory')
    consensus_parser.add_argument('--output_dir', required=True, help='Output consensus directory')
    consensus_parser.add_argument('--max_reads', type=int, default=20, help='Max reads per consensus (default: 20)')
    consensus_parser.add_argument('--max_workers', type=int, default=4, help='Max parallel workers (default: 4)')
    
    variant_parser = subparsers.add_parser('variants', help='Call variants')
    variant_parser.add_argument('--input_dir', required=True, help='Input consensus directory')
    variant_parser.add_argument('--reference', required=True, help='Reference FASTA file')
    variant_parser.add_argument('--output_dir', required=True, help='Output variant directory')
    variant_parser.add_argument('--combined_vcf', required=True, help='Combined VCF output file')
    variant_parser.add_argument('--max_workers', type=int, default=4, help='Max parallel workers (default: 4)')
    
    analyze_parser = subparsers.add_parser('analyze', help='Analyze variants and generate detailed CSV')
    analyze_parser.add_argument('--input_vcf', required=True, help='Input combined VCF file')
    analyze_parser.add_argument('--reference', required=True, help='Reference FASTA file')
    analyze_parser.add_argument('--output', required=True, help='Output CSV file')
    
    # NGS counting command
    ngs_parser = subparsers.add_parser('ngs_count', help='Count pool reads per variant via UMI matching')
    ngs_parser.add_argument('--pools_dir', required=True, help='Directory containing per-pool folders with R1/R2 fastqs')
    ngs_parser.add_argument('--consensus_dir', required=True, help='Consensus directory (from consensus step)')
    ngs_parser.add_argument('--variants_dir', required=True, help='Variants directory with per-consensus VCFs')
    ngs_parser.add_argument('--probe', required=True, help='Probe FASTA file (same used for UMI extraction)')
    ngs_parser.add_argument('--reference', required=True, help='Reference FASTA file for amino acid mapping')
    ngs_parser.add_argument('--umi_len', type=int, default=52, help='UMI length (default: 52)')
    ngs_parser.add_argument('--umi_loc', type=str, default='up', choices=['up','down'], help='UMI location relative to probe (default: up)')
    ngs_parser.add_argument('--output', required=True, help='Output counts CSV file')
    ngs_parser.add_argument('--left_ignore', type=int, default=22, help='Bases to ignore from start of assembled read (default: 22)')
    ngs_parser.add_argument('--right_ignore', type=int, default=24, help='Bases to ignore from end of assembled read (default: 24)')
    
    # Fitness analysis command
    fitness_parser = subparsers.add_parser('fitness', help='Analyze fitness from merged non-synonymous counts')
    fitness_parser.add_argument('--input', required=True, help='Input CSV file (merged_on_nonsyn_counts.csv)')
    fitness_parser.add_argument('--output_dir', required=True, help='Output directory for plots and results')
    fitness_parser.add_argument('--input_pools', required=True, nargs='+', help='Input pool names (space-separated)')
    fitness_parser.add_argument('--output_pools', required=True, nargs='+', help='Output pool names (space-separated, paired with inputs)')
    fitness_parser.add_argument('--min_input', type=int, default=10, help='Minimum count threshold in input pools (default: 10)')
    fitness_parser.add_argument('--aa_filter', type=str, default=None, help='Filter mutability plot to specific mutant amino acid (e.g., S for serine, P for proline, * for stop codons)')
    
    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Launch interactive web-based GUI')
    gui_parser.add_argument('--host', default='127.0.0.1', help='Server host (default: 127.0.0.1)')
    gui_parser.add_argument('--port', type=int, default=7860, help='Server port (default: 7860)')
    gui_parser.add_argument('--share', action='store_true', help='Create public share link')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Run the appropriate command
    if args.command == 'all':
        success = run_full_pipeline(args)
    elif args.command == 'extract':
        success = run_umi_extraction(args)
    elif args.command == 'cluster':
        success = run_clustering(args)
    elif args.command == 'consensus':
        success = run_consensus_generation(args)
    elif args.command == 'variants':
        success = run_variant_calling(args)
    elif args.command == 'analyze':
        success = run_analysis(args)
    elif args.command == 'ngs_count':
        print("=" * 60)
        print("NGS POOL COUNTING")
        print("=" * 60)
        success = ngs_count.run_ngs_count(
            args.pools_dir,
            args.consensus_dir,
            args.variants_dir,
            args.probe,
            args.umi_len,
            args.umi_loc,
            args.output,
            args.reference,
            args.left_ignore,
            args.right_ignore
        )
    elif args.command == 'fitness':
        print("=" * 60)
        print("FITNESS ANALYSIS")
        print("=" * 60)
        success = fitness_analysis.run_fitness_analysis(
            args.input,
            args.output_dir,
            args.input_pools,
            args.output_pools,
            args.min_input,
            args.aa_filter
        )
    elif args.command == 'gui':
        try:
            from .gui import launch_gui
            print("Launching GUI...")
            launch_gui(server_name=args.host, server_port=args.port, share=args.share)
            return 0
        except ImportError:
            print("ERROR: Gradio is not installed. Please install it with: pip install gradio")
            return 1
    else:
        print(f"Unknown command: {args.command}")
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

