#!/usr/bin/env python3
"""
Reference Manager module for handling multiple reference sequences.

Supports 2-6 wild-type reference sequences for multi-gene DMS experiments.
Provides reference identification via minimap2 alignment and caches protein translations.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from Bio import SeqIO
from Bio.Seq import Seq


class ReferenceManager:
    """
    Manages multiple reference sequences for variant calling.

    Handles loading multi-FASTA reference files, identifying which reference
    a consensus sequence best matches, and providing reference-specific
    sequences and protein translations.
    """

    def __init__(self, reference_fasta: str, ambiguity_threshold: float = 0.05,
                 min_alignment_score: float = 50.0):
        """
        Initialize ReferenceManager from a FASTA file (single or multi-sequence).

        Args:
            reference_fasta: Path to reference FASTA file (can contain 1-6 sequences)
            ambiguity_threshold: Score difference threshold (as fraction) to flag
                                ambiguous assignments (default: 0.05 = 5%)
            min_alignment_score: Minimum alignment score to consider a match valid
        """
        self.reference_fasta = reference_fasta
        self.ambiguity_threshold = ambiguity_threshold
        self.min_alignment_score = min_alignment_score

        # Storage for reference sequences and proteins
        self._references: Dict[str, str] = {}  # ref_id -> DNA sequence
        self._proteins: Dict[str, str] = {}    # ref_id -> protein sequence
        self._ref_order: List[str] = []        # preserve order from file

        # Load references
        self._load_references()

        # Create combined reference file for multi-ref alignment
        self._combined_ref_file: Optional[str] = None
        if len(self._references) > 1:
            self._create_combined_reference()

    def _load_references(self) -> None:
        """Load reference sequences from FASTA file."""
        records = list(SeqIO.parse(self.reference_fasta, "fasta"))

        if not records:
            raise ValueError(f"No sequences found in reference file: {self.reference_fasta}")

        if len(records) > 6:
            raise ValueError(f"Too many reference sequences ({len(records)}). Maximum supported: 6")

        for record in records:
            ref_id = record.id
            seq = str(record.seq).upper()
            self._references[ref_id] = seq
            self._ref_order.append(ref_id)

            # Pre-compute protein translation
            self._proteins[ref_id] = self._translate_dna(seq)

    def _translate_dna(self, dna_seq: str) -> str:
        """Translate DNA sequence to protein sequence."""
        try:
            # Clean sequence and ensure length is multiple of 3
            clean_seq = ''.join(c for c in dna_seq.upper() if c in 'ATCG')
            if len(clean_seq) % 3 != 0:
                clean_seq = clean_seq[:len(clean_seq) - (len(clean_seq) % 3)]

            if len(clean_seq) == 0:
                return ""

            return str(Seq(clean_seq).translate())
        except Exception:
            return ""

    def _create_combined_reference(self) -> None:
        """Create a temporary combined reference file for multi-ref alignment."""
        # Create temp file that persists for the lifetime of this object
        fd, path = tempfile.mkstemp(suffix='.fasta', prefix='combined_ref_')
        os.close(fd)
        self._combined_ref_file = path

        with open(path, 'w') as f:
            for ref_id in self._ref_order:
                f.write(f">{ref_id}\n{self._references[ref_id]}\n")

    def __del__(self):
        """Clean up temporary combined reference file."""
        if self._combined_ref_file and os.path.exists(self._combined_ref_file):
            try:
                os.remove(self._combined_ref_file)
            except Exception:
                pass

    def identify_best_reference(self, consensus_seq: str) -> Tuple[str, float, bool]:
        """
        Identify which reference best matches a consensus sequence.

        Uses minimap2 alignment to find the best-matching reference.

        Args:
            consensus_seq: The consensus DNA sequence to identify

        Returns:
            Tuple of (best_ref_id, alignment_score, is_ambiguous)
            - best_ref_id: ID of best matching reference, or "UNASSIGNED" if no good match
            - alignment_score: Score of the best alignment
            - is_ambiguous: True if top two refs have similar scores (within threshold)
        """
        # Single reference case - no alignment needed
        if len(self._references) == 1:
            return (self._ref_order[0], 100.0, False)

        # Write consensus to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(f">query\n{consensus_seq}\n")
            query_file = f.name

        try:
            # Run minimap2 against combined reference
            result = subprocess.run(
                [
                    "minimap2",
                    "-c",  # Output CIGAR and alignment scores
                    "-x", "map-ont",
                    "-t", "1",
                    self._combined_ref_file,
                    query_file
                ],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                return ("UNASSIGNED", 0.0, False)

            # Parse PAF output to find best matching reference
            scores: Dict[str, float] = {}
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 12:
                    ref_id = parts[5]
                    # Use alignment score (AS tag) if available, else use match count
                    score = float(parts[9])  # Number of matching bases
                    for tag in parts[12:]:
                        if tag.startswith('AS:i:'):
                            score = float(tag[5:])
                            break

                    # Keep highest score per reference
                    if ref_id not in scores or score > scores[ref_id]:
                        scores[ref_id] = score

            if not scores:
                return ("UNASSIGNED", 0.0, False)

            # Sort by score descending
            sorted_refs = sorted(scores.items(), key=lambda x: -x[1])
            best_ref, best_score = sorted_refs[0]

            # Check minimum score threshold
            if best_score < self.min_alignment_score:
                return ("UNASSIGNED", best_score, False)

            # Check for ambiguity (if second-best is within threshold)
            is_ambiguous = False
            if len(sorted_refs) >= 2:
                second_score = sorted_refs[1][1]
                if best_score > 0:
                    score_diff = (best_score - second_score) / best_score
                    is_ambiguous = score_diff < self.ambiguity_threshold

            return (best_ref, best_score, is_ambiguous)

        except subprocess.TimeoutExpired:
            return ("UNASSIGNED", 0.0, False)
        except Exception:
            return ("UNASSIGNED", 0.0, False)
        finally:
            # Clean up temp file
            if os.path.exists(query_file):
                os.remove(query_file)

    def get_reference_sequence(self, ref_id: str) -> str:
        """
        Get DNA sequence for a reference.

        Args:
            ref_id: Reference identifier

        Returns:
            DNA sequence string, or empty string if ref_id not found
        """
        return self._references.get(ref_id, "")

    def get_reference_protein(self, ref_id: str) -> str:
        """
        Get protein sequence for a reference.

        Args:
            ref_id: Reference identifier

        Returns:
            Protein sequence string, or empty string if ref_id not found
        """
        return self._proteins.get(ref_id, "")

    def get_reference_file(self, ref_id: str) -> Optional[str]:
        """
        Get path to a temporary FASTA file containing just one reference.

        Note: The returned file is temporary and should be cleaned up after use.

        Args:
            ref_id: Reference identifier

        Returns:
            Path to temporary FASTA file, or None if ref_id not found
        """
        if ref_id not in self._references:
            return None

        fd, path = tempfile.mkstemp(suffix='.fasta', prefix=f'ref_{ref_id}_')
        os.close(fd)

        with open(path, 'w') as f:
            f.write(f">{ref_id}\n{self._references[ref_id]}\n")

        return path

    def count(self) -> int:
        """Return number of reference sequences."""
        return len(self._references)

    def is_multi_reference(self) -> bool:
        """Return True if multiple references are loaded."""
        return len(self._references) > 1

    def get_reference_ids(self) -> List[str]:
        """Return list of reference IDs in order they were loaded."""
        return list(self._ref_order)

    def get_default_reference_id(self) -> str:
        """Return the first (default) reference ID."""
        return self._ref_order[0] if self._ref_order else ""

    def get_reference_info(self) -> str:
        """Return a formatted string with reference information."""
        lines = [f"Loaded {self.count()} reference sequence(s):"]
        for ref_id in self._ref_order:
            seq_len = len(self._references[ref_id])
            prot_len = len(self._proteins[ref_id])
            lines.append(f"  - {ref_id}: {seq_len} bp ({prot_len} aa)")
        return '\n'.join(lines)

    def get_original_fasta_path(self) -> str:
        """Return path to the original reference FASTA file."""
        return self.reference_fasta

    def get_combined_fasta_path(self) -> Optional[str]:
        """
        Return path to combined reference FASTA (for multi-ref mode).

        For single-reference mode, returns the original file path.
        """
        if self._combined_ref_file:
            return self._combined_ref_file
        return self.reference_fasta
