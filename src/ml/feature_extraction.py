"""Feature extraction for real-time AMR prediction.

This module extracts k-mer features from DNA sequences for prediction.
Uses the same k-mer vocabulary as the trained model.
"""

import json
import gzip
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from collections import Counter

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "ncbi" / "ncbi_amr_metadata.json"


class KmerFeatureExtractor:
    """Extract k-mer features from DNA sequences using trained vocabulary.

    This extractor uses the same k-mer vocabulary that was used during model
    training to ensure consistent feature extraction for inference.
    """

    def __init__(self, metadata_path: Optional[str] = None):
        """Initialize the feature extractor.

        Args:
            metadata_path: Path to metadata JSON containing feature_names.
                          If None, uses default path.
        """
        self.metadata_path = Path(metadata_path) if metadata_path else METADATA_PATH
        self.feature_names: List[str] = []
        self.k: int = 6
        self.kmer_to_idx: dict = {}

        self._load_vocabulary()

    def _load_vocabulary(self):
        """Load k-mer vocabulary from metadata file."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with open(self.metadata_path) as f:
            metadata = json.load(f)

        self.feature_names = metadata.get("feature_names", [])
        self.k = metadata.get("k", 6)

        if not self.feature_names:
            raise ValueError("No feature_names found in metadata")

        self.kmer_to_idx = {kmer: idx for idx, kmer in enumerate(self.feature_names)}

        logger.info(f"Loaded {len(self.feature_names)} k-mer features (k={self.k})")

    def extract_features(self, sequence: str) -> np.ndarray:
        """Extract k-mer features from a single DNA sequence.

        Args:
            sequence: DNA sequence string (A, C, G, T characters)

        Returns:
            Feature vector of shape (n_features,) with k-mer frequencies
        """
        sequence = sequence.upper().replace('\n', '').replace(' ', '')
        seq_len = len(sequence) - self.k + 1

        if seq_len <= 0:
            logger.warning(f"Sequence too short (len={len(sequence)}, need >={self.k})")
            return np.zeros(len(self.feature_names))

        # Count k-mers
        features = np.zeros(len(self.feature_names))
        valid_count = 0

        for i in range(seq_len):
            kmer = sequence[i:i + self.k]
            # Only count valid DNA k-mers
            if all(c in "ACGT" for c in kmer):
                valid_count += 1
                if kmer in self.kmer_to_idx:
                    features[self.kmer_to_idx[kmer]] += 1

        # Normalize by total valid k-mers
        if valid_count > 0:
            features = features / valid_count

        return features

    def extract_features_batch(self, sequences: List[str]) -> np.ndarray:
        """Extract k-mer features from multiple sequences.

        Args:
            sequences: List of DNA sequence strings

        Returns:
            Feature matrix of shape (n_sequences, n_features)
        """
        return np.array([self.extract_features(seq) for seq in sequences])

    def parse_fasta(self, content: str) -> List[Tuple[str, str]]:
        """Parse FASTA format content.

        Args:
            content: FASTA file content as string

        Returns:
            List of (header, sequence) tuples
        """
        sequences = []
        current_header = None
        current_seq = []

        for line in content.strip().split('\n'):
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    sequences.append((current_header, ''.join(current_seq)))
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

        if current_header is not None:
            sequences.append((current_header, ''.join(current_seq)))

        return sequences

    def parse_fastq(self, content: str) -> List[Tuple[str, str]]:
        """Parse FASTQ format content.

        Args:
            content: FASTQ file content as string

        Returns:
            List of (header, sequence) tuples
        """
        sequences = []
        lines = content.strip().split('\n')

        i = 0
        while i < len(lines):
            if lines[i].startswith('@'):
                header = lines[i][1:]
                sequence = lines[i + 1] if i + 1 < len(lines) else ''
                sequences.append((header, sequence))
                i += 4  # Skip quality lines
            else:
                i += 1

        return sequences

    def extract_from_file_content(
        self,
        content: str,
        file_format: str = "fasta"
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract features from file content.

        Args:
            content: File content as string
            file_format: Either 'fasta' or 'fastq'

        Returns:
            Tuple of (feature_matrix, sequence_headers)
        """
        if file_format.lower() in ['fastq', 'fq']:
            sequences = self.parse_fastq(content)
        else:
            sequences = self.parse_fasta(content)

        if not sequences:
            raise ValueError("No sequences found in file content")

        headers = [h for h, _ in sequences]
        seqs = [s for _, s in sequences]

        # For multiple sequences, concatenate them (typical for assembled genomes)
        if len(seqs) > 1:
            logger.info(f"Found {len(seqs)} sequences, concatenating for feature extraction")
            combined_seq = ''.join(seqs)
            features = self.extract_features(combined_seq)
            return features.reshape(1, -1), [f"combined_{len(seqs)}_sequences"]
        else:
            features = self.extract_features(seqs[0])
            return features.reshape(1, -1), headers

    @property
    def n_features(self) -> int:
        """Number of features (k-mers) in vocabulary."""
        return len(self.feature_names)


# Global extractor instance (lazy loaded)
_extractor: Optional[KmerFeatureExtractor] = None


def get_extractor() -> KmerFeatureExtractor:
    """Get or create the global feature extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = KmerFeatureExtractor()
    return _extractor


def extract_features_from_sequence(sequence: str) -> np.ndarray:
    """Convenience function to extract features from a sequence.

    Args:
        sequence: DNA sequence string

    Returns:
        Feature vector of shape (n_features,)
    """
    return get_extractor().extract_features(sequence)


def extract_features_from_fasta(content: str) -> np.ndarray:
    """Convenience function to extract features from FASTA content.

    Args:
        content: FASTA file content

    Returns:
        Feature vector of shape (n_features,)
    """
    features, _ = get_extractor().extract_from_file_content(content, "fasta")
    return features[0]  # Return first (and typically only) row
