#!/usr/bin/env python3
"""
Simple consensus pipeline for PacBio data.
Generates consensus sequences directly from cluster files using abpoa.
No VCF generation - just consensus sequences.
"""

import os
import subprocess
import sys
import time
import threading
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
import gc
import resource
import re

# Try to import psutil for memory monitoring (optional but recommended)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# Valid nucleotide characters (case-insensitive)
VALID_NUCLEOTIDES = set('ACGTNacgtn')


def validate_fasta_file(fasta_path):
    """
    Validate a FASTA file for common issues before passing to abpoa.

    Args:
        fasta_path: Path to the FASTA file to validate

    Returns:
        dict with keys:
            - valid: bool, True if file is valid
            - error_type: str or None, type of error if invalid
            - message: str, human-readable description
            - sequence_count: int, number of sequences found
            - file_size: int, file size in bytes
            - problematic_headers: list, headers with issues (if any)
    """
    result = {
        'valid': False,
        'error_type': None,
        'message': '',
        'sequence_count': 0,
        'file_size': 0,
        'problematic_headers': []
    }

    # Check file exists and is accessible
    if not os.path.exists(fasta_path):
        result['error_type'] = 'FILE_NOT_FOUND'
        result['message'] = f'File does not exist: {fasta_path}'
        return result

    try:
        result['file_size'] = os.path.getsize(fasta_path)
    except OSError as e:
        result['error_type'] = 'FILE_ACCESS_ERROR'
        result['message'] = f'Cannot access file: {e}'
        return result

    # Check file is not empty
    if result['file_size'] == 0:
        result['error_type'] = 'EMPTY_FILE'
        result['message'] = 'File is empty (0 bytes)'
        return result

    # Parse and validate FASTA content
    try:
        with open(fasta_path, 'r') as f:
            current_header = None
            current_sequence = []
            sequence_count = 0
            problematic_headers = []
            line_number = 0

            for line in f:
                line_number += 1
                line = line.strip()

                if not line:
                    continue  # Skip empty lines

                if line.startswith('>'):
                    # Process previous sequence if exists
                    if current_header is not None:
                        seq = ''.join(current_sequence)
                        if len(seq) == 0:
                            problematic_headers.append({
                                'header': current_header,
                                'issue': 'empty_sequence'
                            })
                        else:
                            # Check for invalid nucleotides
                            invalid_chars = set(seq) - VALID_NUCLEOTIDES
                            if invalid_chars:
                                problematic_headers.append({
                                    'header': current_header,
                                    'issue': 'invalid_nucleotides',
                                    'chars': list(invalid_chars)[:5]  # First 5 invalid chars
                                })

                    # Start new sequence
                    current_header = line[1:].split()[0] if len(line) > 1 else f'seq_{line_number}'
                    current_sequence = []
                    sequence_count += 1

                else:
                    # Sequence line
                    if current_header is None:
                        result['error_type'] = 'INVALID_FORMAT'
                        result['message'] = f'Sequence data before header at line {line_number}'
                        return result
                    current_sequence.append(line)

            # Process last sequence
            if current_header is not None:
                seq = ''.join(current_sequence)
                if len(seq) == 0:
                    problematic_headers.append({
                        'header': current_header,
                        'issue': 'empty_sequence'
                    })
                else:
                    invalid_chars = set(seq) - VALID_NUCLEOTIDES
                    if invalid_chars:
                        problematic_headers.append({
                            'header': current_header,
                            'issue': 'invalid_nucleotides',
                            'chars': list(invalid_chars)[:5]
                        })

            result['sequence_count'] = sequence_count
            result['problematic_headers'] = problematic_headers

            # Check if we found any sequences
            if sequence_count == 0:
                result['error_type'] = 'NO_SEQUENCES'
                result['message'] = 'No valid FASTA sequences found in file'
                return result

            # Check for problematic sequences
            empty_seqs = [h for h in problematic_headers if h['issue'] == 'empty_sequence']
            invalid_seqs = [h for h in problematic_headers if h['issue'] == 'invalid_nucleotides']

            if empty_seqs:
                result['error_type'] = 'EMPTY_SEQUENCES'
                result['message'] = f'{len(empty_seqs)} sequence(s) have empty content'
                return result

            if invalid_seqs:
                result['error_type'] = 'INVALID_NUCLEOTIDES'
                chars = set()
                for h in invalid_seqs:
                    chars.update(h.get('chars', []))
                result['message'] = f'{len(invalid_seqs)} sequence(s) contain invalid characters: {list(chars)[:10]}'
                return result

            # All checks passed
            result['valid'] = True
            result['message'] = f'Valid FASTA with {sequence_count} sequence(s)'
            return result

    except UnicodeDecodeError as e:
        result['error_type'] = 'ENCODING_ERROR'
        result['message'] = f'File encoding error: {e}'
        return result
    except Exception as e:
        result['error_type'] = 'PARSE_ERROR'
        result['message'] = f'Error parsing file: {e}'
        return result


class ConsensusErrorLogger:
    """
    Thread-safe error logger for consensus generation.
    Writes errors to a JSON Lines file for later analysis.
    """

    def __init__(self, output_dir, filename='consensus_errors.jsonl'):
        """
        Initialize the error logger.

        Args:
            output_dir: Directory to write the log file
            filename: Name of the log file (default: consensus_errors.jsonl)
        """
        self.log_path = os.path.join(output_dir, filename)
        self.lock = threading.Lock()
        self.error_counts = {}  # Track counts by error type
        self.start_time = datetime.now()

        # Write header/start marker
        with open(self.log_path, 'w') as f:
            start_record = {
                'type': 'SESSION_START',
                'timestamp': self.start_time.isoformat(),
                'output_dir': output_dir
            }
            f.write(json.dumps(start_record) + '\n')

    def log_error(self, cluster_name, error_type, message, details=None):
        """
        Log an error for a cluster.

        Args:
            cluster_name: Name of the cluster that failed
            error_type: Category of error (VALIDATION, TIMEOUT, MEMORY, ABPOA, OTHER)
            message: Human-readable error message
            details: Optional dict with additional details
        """
        record = {
            'type': 'ERROR',
            'timestamp': datetime.now().isoformat(),
            'cluster_name': cluster_name,
            'error_type': error_type,
            'message': message
        }
        if details:
            record['details'] = details

        with self.lock:
            # Update error counts
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

            # Write to file
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(record) + '\n')

    def write_summary(self, total_processed, success_count, failed_count):
        """
        Write a summary record at the end of processing.

        Args:
            total_processed: Total number of clusters processed
            success_count: Number of successful consensus generations
            failed_count: Number of failed consensus generations
        """
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        summary = {
            'type': 'SESSION_SUMMARY',
            'timestamp': end_time.isoformat(),
            'duration_seconds': duration,
            'total_processed': total_processed,
            'success_count': success_count,
            'failed_count': failed_count,
            'error_breakdown': self.error_counts
        }

        with self.lock:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(summary) + '\n')

        return self.log_path

    def get_error_counts(self):
        """Return current error counts by type."""
        with self.lock:
            return dict(self.error_counts)


def get_memory_percent():
    """Get current memory usage percentage. Returns None if unavailable."""
    if PSUTIL_AVAILABLE:
        return psutil.virtual_memory().percent
    return None


def get_available_memory_gb():
    """Get available memory in GB. Returns None if unavailable."""
    if PSUTIL_AVAILABLE:
        return psutil.virtual_memory().available / (1024 ** 3)
    return None


def set_process_memory_limit(max_mb=2048):
    """
    Set soft memory limit for subprocess.
    This helps prevent individual abpoa processes from consuming too much memory.
    """
    try:
        # Set soft limit to max_mb, hard limit slightly higher
        soft_limit = max_mb * 1024 * 1024  # Convert to bytes
        hard_limit = (max_mb + 512) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (soft_limit, hard_limit))
    except (ValueError, resource.error):
        # If we can't set limits, continue anyway
        pass


def check_memory_pressure(threshold_percent=85):
    """
    Check if system is under memory pressure.
    Returns True if memory usage exceeds threshold.
    """
    mem_percent = get_memory_percent()
    if mem_percent is not None:
        return mem_percent > threshold_percent
    # If we can't check, assume we're okay
    return False


def wait_for_memory_relief(threshold_percent=75, timeout=60, check_interval=2):
    """
    Wait for memory usage to drop below threshold.
    Returns True if memory dropped, False if timeout.
    """
    if not PSUTIL_AVAILABLE:
        # Can't monitor, just do a brief pause
        time.sleep(5)
        gc.collect()
        return True

    start_time = time.time()
    while time.time() - start_time < timeout:
        gc.collect()
        mem_percent = get_memory_percent()
        if mem_percent is not None and mem_percent < threshold_percent:
            return True
        time.sleep(check_interval)
    return False

def run_abpoa_consensus(cluster_file, output_dir, max_reads=20, memory_limit_mb=1024, max_seq_len=15000, error_logger=None):
    """
    Run abpoa to generate consensus sequence from first N reads.

    Args:
        cluster_file: Path to cluster FASTA file
        output_dir: Output directory for consensus
        max_reads: Maximum reads to use for consensus (default: 20)
        memory_limit_mb: Memory limit for abpoa subprocess in MB (default: 1024)
        max_seq_len: Maximum sequence length to include (default: 15000)
        error_logger: Optional ConsensusErrorLogger instance for logging errors
    """
    temp_file = None
    cluster_name = Path(cluster_file).stem

    try:
        consensus_file = os.path.join(output_dir, f"{cluster_name}_consensus.fasta")

        # Skip if already exists
        if os.path.exists(consensus_file):
            return consensus_file

        # Validate input FASTA file before processing
        validation = validate_fasta_file(cluster_file)
        if not validation['valid']:
            error_msg = f"Validation failed for {cluster_name}: {validation['message']}"
            if error_logger:
                error_logger.log_error(
                    cluster_name,
                    'VALIDATION',
                    validation['message'],
                    details={
                        'error_type': validation['error_type'],
                        'sequence_count': validation['sequence_count'],
                        'file_size': validation['file_size'],
                        'problematic_headers': validation['problematic_headers'][:5]  # Limit to 5
                    }
                )
            return error_msg

        # Create temporary file with first N sequences
        temp_file = os.path.join(output_dir, f"{cluster_name}_temp_{max_reads}.fasta")

        # Extract first N sequences while skipping overly long reads
        seq_count = 0
        skipped_long = 0
        seq_header = None
        seq_lines = []
        seq_len = 0
        max_len = max_seq_len if max_seq_len and max_seq_len > 0 else None

        def flush_sequence():
            nonlocal seq_count, skipped_long, seq_header, seq_lines, seq_len
            if seq_header is None:
                return
            if max_len is not None and seq_len > max_len:
                skipped_long += 1
            else:
                outfile.write(seq_header)
                outfile.writelines(seq_lines)
                seq_count += 1
            seq_header = None
            seq_lines = []
            seq_len = 0

        with open(cluster_file, 'r') as infile, open(temp_file, 'w') as outfile:
            for line in infile:
                if line.startswith('>'):
                    if seq_header is not None:
                        flush_sequence()
                        if seq_count >= max_reads:
                            break
                    seq_header = line
                    seq_lines = []
                    seq_len = 0
                else:
                    if seq_header is None:
                        continue
                    seq_lines.append(line)
                    seq_len += len(line.strip())

            if seq_header is not None and seq_count < max_reads:
                flush_sequence()

        # Skip if too few sequences (abpoa needs at least 1)
        if seq_count == 0:
            if skipped_long > 0 and max_len is not None:
                error_msg = f"All sequences exceeded max length ({max_len}) in {cluster_name}"
                if error_logger:
                    error_logger.log_error(
                        cluster_name,
                        'LENGTH',
                        error_msg,
                        details={'max_seq_len': max_len, 'skipped_sequences': skipped_long}
                    )
            else:
                error_msg = f"No sequences in {cluster_name}"
            if error_logger:
                error_logger.log_error(
                    cluster_name,
                    'VALIDATION',
                    'No sequences extracted from cluster file',
                    details={'file_size': validation['file_size']}
                )
            return error_msg

        # Run abpoa consensus
        cmd = [
            "abpoa",
            "-r", "0",    # output consensus in FASTA format
            "-a", "0",    # heaviest bundling path algorithm
            "-o", consensus_file,
            temp_file
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            preexec_fn=lambda: set_process_memory_limit(memory_limit_mb)
        )

        if result.returncode != 0:
            # Check for specific error patterns
            stderr = result.stderr.lower() if result.stderr else ""
            if "memory" in stderr or "alloc" in stderr:
                error_msg = f"abpoa memory error for {cluster_name}"
                if error_logger:
                    error_logger.log_error(
                        cluster_name,
                        'MEMORY',
                        'abpoa reported memory/allocation error',
                        details={'stderr': result.stderr[:500] if result.stderr else None}
                    )
                return error_msg
            error_msg = f"abpoa failed for {cluster_name}: {result.stderr}"
            if error_logger:
                error_logger.log_error(
                    cluster_name,
                    'ABPOA',
                    f'abpoa exited with code {result.returncode}',
                    details={
                        'returncode': result.returncode,
                        'stderr': result.stderr[:500] if result.stderr else None,
                        'stdout': result.stdout[:500] if result.stdout else None
                    }
                )
            return error_msg

        # Verify output file was created
        if not os.path.exists(consensus_file):
            error_msg = f"abpoa did not produce output for {cluster_name}"
            if error_logger:
                error_logger.log_error(
                    cluster_name,
                    'ABPOA',
                    'abpoa succeeded but no output file was created',
                    details={'expected_output': consensus_file}
                )
            return error_msg

        return consensus_file

    except subprocess.TimeoutExpired:
        # Clean up any partial output
        if os.path.exists(os.path.join(output_dir, f"{cluster_name}_consensus.fasta")):
            try:
                os.remove(os.path.join(output_dir, f"{cluster_name}_consensus.fasta"))
            except:
                pass
        error_msg = f"abpoa timeout for {cluster_name}"
        if error_logger:
            error_logger.log_error(
                cluster_name,
                'TIMEOUT',
                'abpoa exceeded 30 second timeout',
                details={'timeout_seconds': 30}
            )
        return error_msg
    except MemoryError:
        error_msg = f"Memory error processing {cluster_name}"
        if error_logger:
            error_logger.log_error(
                cluster_name,
                'MEMORY',
                'Python MemoryError during processing'
            )
        return error_msg
    except Exception as e:
        error_msg = f"Error in abpoa for {cluster_name}: {str(e)}"
        if error_logger:
            error_logger.log_error(
                cluster_name,
                'OTHER',
                str(e),
                details={'exception_type': type(e).__name__}
            )
        return error_msg
    finally:
        # Always clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

def process_cluster_simple(cluster_file, output_dir, max_reads=20, memory_limit_mb=1024, max_seq_len=15000, error_logger=None):
    """
    Process a single cluster to generate consensus sequence.

    Args:
        cluster_file: Path to cluster FASTA file
        output_dir: Output directory
        max_reads: Max reads for consensus
        memory_limit_mb: Memory limit per abpoa process
        error_logger: Optional ConsensusErrorLogger instance for logging errors
    """
    cluster_name = Path(cluster_file).stem

    try:
        # Generate consensus using abpoa with memory limits and error logging
        consensus_result = run_abpoa_consensus(
            cluster_file,
            output_dir,
            max_reads,
            memory_limit_mb,
            max_seq_len,
            error_logger
        )

        if isinstance(consensus_result, str) and consensus_result.endswith('.fasta'):
            return f"SUCCESS: {cluster_name}"
        else:
            return f"FAILED: {cluster_name} - {consensus_result}"

    except MemoryError:
        gc.collect()
        if error_logger:
            error_logger.log_error(
                cluster_name,
                'MEMORY',
                'Python MemoryError in process_cluster_simple'
            )
        return f"MEMORY_ERROR: {cluster_name}"
    except Exception as e:
        if error_logger:
            error_logger.log_error(
                cluster_name,
                'OTHER',
                str(e),
                details={'exception_type': type(e).__name__}
            )
        return f"ERROR: {cluster_name} - {str(e)}"

def main():
    # Configuration
    cluster_files_dir = '/Volumes/Matt115A1TB_1/temp-dict-epgh11/UMIclusterfull_fast'
    output_dir = '/Volumes/Matt115A1TB_1/temp-dict-epgh11/simple_consensus_results'
    max_reads = 20  # Use first 20 reads for consensus
    
    # Use 4 threads for parallel processing
    max_workers = 4
    
    print(f"Starting simple consensus pipeline...")
    print(f"Cluster files directory: {cluster_files_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max reads per consensus: {max_reads}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of cluster files
    cluster_files = []
    for file in os.listdir(cluster_files_dir):
        if file.endswith('.fasta') and file.startswith('cluster_'):
            cluster_files.append(os.path.join(cluster_files_dir, file))
    
    total_clusters = len(cluster_files)
    print(f"Found {total_clusters:,} cluster files to process")
    
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
            progress_percent = (processed / total_clusters) * 100
            bar_length = 50
            filled_length = int(bar_length * processed // total_clusters)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            # Print progress line
            progress_line = (f"\rProgress: |{bar}| {progress_percent:.1f}% "
                           f"({processed:,}/{total_clusters:,}) | "
                           f"Success: {success_count:,} | Failed: {failed_count:,} | "
                           f"Rate: {rate:.1f} clusters/s | ETA: {eta_str} | "
                           f"Elapsed: {elapsed_str}")
            
            print(progress_line, end='', flush=True)
    
    # Process clusters in parallel with memory-safe batching
    BATCH_SIZE = 100  # Process in smaller batches to control memory
    processed_total = 0

    while processed_total < total_clusters:
        batch_start = processed_total
        batch_end = min(batch_start + BATCH_SIZE, total_clusters)
        batch_files = cluster_files[batch_start:batch_end]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_cluster_simple, cluster_file, output_dir, max_reads): cluster_file
                for cluster_file in batch_files
            }

            # Process completed jobs - delete futures to prevent memory leak
            for future in as_completed(futures):
                result = future.result()

                with progress_lock:
                    if "SUCCESS" in result:
                        success_count += 1
                    else:
                        failed_count += 1
                        if failed_count <= 10:  # Only show first 10 failures
                            print(f"\nFailed: {result}")

                # MEMORY LEAK FIX: cleanup references
                del result
                del futures[future]

                update_progress()

        processed_total = batch_end
        gc.collect()  # Force cleanup between batches
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\nSimple consensus pipeline completed!")
    print(f"Total clusters processed: {total_clusters:,}")
    print(f"Successful: {success_count:,}")
    print(f"Failed: {failed_count:,}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average rate: {total_clusters/total_time:.1f} clusters/second")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()
