"""ResFinder database preprocessor for AMR prediction modeling."""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResFinderPreprocessor:
    """Preprocess ResFinder data for AMR prediction models."""

    def __init__(
        self,
        resfinder_dir: str = "data/raw/resfinder",
        output_dir: str = "data/processed/resfinder",
    ):
        self.resfinder_dir = Path(resfinder_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.genes_df: Optional[pd.DataFrame] = None
        self.phenotypes_df: Optional[pd.DataFrame] = None
        self.sequences: Dict[str, str] = {}
        self.label_encoders: Dict = {}

    def load_data(self) -> None:
        """Load all ResFinder data files."""
        logger.info("Loading ResFinder data...")

        # Load sequences from FASTA files
        self._load_sequences()

        # Load phenotype information
        self._load_phenotypes()

        # Create gene index
        self._create_gene_index()

    def _load_sequences(self) -> None:
        """Load gene sequences from FASTA files."""
        for fasta_file in self.resfinder_dir.glob("*.fsa"):
            # Skip the 'all.fsa' file to avoid duplicates
            if fasta_file.stem == "all":
                continue

            drug_class = fasta_file.stem
            logger.info(f"Loading sequences from {drug_class}.fsa")

            try:
                for record in SeqIO.parse(fasta_file, "fasta"):
                    self.sequences[record.id] = {
                        "sequence": str(record.seq),
                        "description": record.description,
                        "drug_class": drug_class,
                    }
            except Exception as e:
                logger.error(f"Error parsing {fasta_file}: {e}")

        logger.info(f"Loaded {len(self.sequences)} gene sequences")

    def _load_phenotypes(self) -> None:
        """Load phenotype information from phenotypes.txt."""
        phenotypes_file = self.resfinder_dir / "phenotypes.txt"
        if not phenotypes_file.exists():
            logger.warning(f"Phenotypes file not found: {phenotypes_file}")
            return

        # Read tab-separated file
        self.phenotypes_df = pd.read_csv(
            phenotypes_file,
            sep="\t",
            dtype=str,
            na_values=["", "NA", "N/A"],
        )

        # Clean column names
        self.phenotypes_df.columns = [
            col.strip().replace(" ", "_").lower()
            for col in self.phenotypes_df.columns
        ]

        # Rename columns for consistency
        column_mapping = {
            "gene_accession_no.": "gene_id",
            "gene_accession no.": "gene_id",
        }
        self.phenotypes_df.rename(columns=column_mapping, inplace=True)

        logger.info(f"Loaded {len(self.phenotypes_df)} phenotype records")

    def _create_gene_index(self) -> None:
        """Create comprehensive gene index combining sequences and phenotypes."""
        genes = []

        for gene_id, seq_info in self.sequences.items():
            # Parse gene name from ID (format: gene_variant_accession)
            parts = gene_id.split("_")
            gene_name = parts[0] if parts else gene_id

            gene_data = {
                "gene_id": gene_id,
                "gene_name": gene_name,
                "drug_class": seq_info["drug_class"],
                "sequence_length": len(seq_info["sequence"]),
                "description": seq_info["description"],
            }

            # Add phenotype info if available
            if self.phenotypes_df is not None and "gene_id" in self.phenotypes_df.columns:
                pheno_match = self.phenotypes_df[
                    self.phenotypes_df["gene_id"] == gene_id
                ]
                if not pheno_match.empty:
                    row = pheno_match.iloc[0]
                    gene_data["phenotype"] = row.get("phenotype", "")
                    gene_data["mechanism"] = row.get("mechanism_of_resistance", "")
                    gene_data["pmid"] = row.get("pmid", "")
                    gene_data["notes"] = row.get("notes", "")

            genes.append(gene_data)

        self.genes_df = pd.DataFrame(genes)
        logger.info(f"Created gene index with {len(self.genes_df)} entries")

    def get_drug_class_statistics(self) -> pd.DataFrame:
        """Get statistics about drug classes in the dataset."""
        if self.genes_df is None:
            self.load_data()

        stats = self.genes_df.groupby("drug_class").agg(
            gene_count=("gene_id", "count"),
            unique_genes=("gene_name", "nunique"),
            mean_seq_length=("sequence_length", "mean"),
            min_seq_length=("sequence_length", "min"),
            max_seq_length=("sequence_length", "max"),
        ).reset_index()

        stats = stats.sort_values("gene_count", ascending=False)
        return stats

    def get_mechanism_statistics(self) -> pd.DataFrame:
        """Get statistics about resistance mechanisms."""
        if self.genes_df is None:
            self.load_data()

        if "mechanism" not in self.genes_df.columns:
            logger.warning("Mechanism information not available")
            return pd.DataFrame()

        stats = self.genes_df.groupby("mechanism").agg(
            gene_count=("gene_id", "count"),
            drug_classes=("drug_class", lambda x: list(x.unique())),
        ).reset_index()

        stats = stats.sort_values("gene_count", ascending=False)
        return stats

    def get_phenotype_statistics(self) -> pd.DataFrame:
        """Get statistics about specific antibiotic phenotypes."""
        if self.genes_df is None:
            self.load_data()

        if "phenotype" not in self.genes_df.columns:
            logger.warning("Phenotype information not available")
            return pd.DataFrame()

        # Parse phenotypes (comma-separated antibiotics)
        antibiotic_counts = Counter()
        for phenotype in self.genes_df["phenotype"].dropna():
            for ab in phenotype.split(","):
                ab = ab.strip()
                if ab:
                    antibiotic_counts[ab] += 1

        stats = pd.DataFrame([
            {"antibiotic": ab, "gene_count": count}
            for ab, count in antibiotic_counts.most_common()
        ])

        return stats

    def compute_gc_content(self, sequence: str) -> float:
        """Calculate GC content of a DNA sequence."""
        sequence = sequence.upper()
        if len(sequence) == 0:
            return 0.0
        gc_count = sequence.count("G") + sequence.count("C")
        return gc_count / len(sequence)

    def get_sequence_statistics(self) -> pd.DataFrame:
        """Get statistics about gene sequences."""
        if not self.sequences:
            self._load_sequences()

        stats = []
        for gene_id, seq_info in self.sequences.items():
            seq = seq_info["sequence"]
            seq_upper = seq.upper()

            stats.append({
                "gene_id": gene_id,
                "drug_class": seq_info["drug_class"],
                "length": len(seq),
                "gc_content": self.compute_gc_content(seq),
                "a_count": seq_upper.count("A"),
                "t_count": seq_upper.count("T"),
                "g_count": seq_upper.count("G"),
                "c_count": seq_upper.count("C"),
            })

        return pd.DataFrame(stats).sort_values("length", ascending=False)

    def extract_kmer_features(
        self,
        sequences: List[str],
        k: int = 6,
        max_features: int = 1000,
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract k-mer frequency features from DNA sequences.

        Args:
            sequences: List of DNA sequences.
            k: k-mer size (default 6 for DNA).
            max_features: Maximum number of k-mer features.

        Returns:
            Tuple of (feature_matrix, feature_names).
        """
        logger.info(f"Extracting {k}-mer features from {len(sequences)} sequences...")

        # Count all k-mers across sequences to find most common
        all_kmers = Counter()
        for seq in sequences:
            seq = seq.upper()
            for i in range(len(seq) - k + 1):
                kmer = seq[i : i + k]
                # Valid DNA nucleotides only
                if all(c in "ACGT" for c in kmer):
                    all_kmers[kmer] += 1

        # Select top k-mers as features
        top_kmers = [kmer for kmer, _ in all_kmers.most_common(max_features)]
        logger.info(f"Selected {len(top_kmers)} k-mer features")

        # Create feature matrix
        feature_matrix = np.zeros((len(sequences), len(top_kmers)))
        kmer_to_idx = {kmer: idx for idx, kmer in enumerate(top_kmers)}

        for seq_idx, seq in enumerate(sequences):
            seq = seq.upper()
            seq_len = len(seq) - k + 1
            if seq_len <= 0:
                continue
            for i in range(seq_len):
                kmer = seq[i : i + k]
                if kmer in kmer_to_idx:
                    feature_matrix[seq_idx, kmer_to_idx[kmer]] += 1
            # Normalize by sequence length
            if seq_len > 0:
                feature_matrix[seq_idx] /= seq_len

        return feature_matrix, top_kmers

    def prepare_drug_class_data(
        self,
        k: int = 6,
        max_features: int = 500,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        min_samples_per_class: int = 10,
    ) -> Dict:
        """Prepare dataset for drug class prediction (multiclass).

        Args:
            k: k-mer size for feature extraction.
            max_features: Maximum number of k-mer features.
            test_size: Proportion for testing.
            val_size: Proportion of training data for validation.
            random_state: Random seed.
            min_samples_per_class: Minimum samples to include a class.

        Returns:
            Dictionary with train/val/test splits and metadata.
        """
        if self.genes_df is None:
            self.load_data()

        logger.info("Preparing drug class prediction data...")

        # Filter to classes with enough samples
        class_counts = self.genes_df["drug_class"].value_counts()
        valid_classes = class_counts[class_counts >= min_samples_per_class].index.tolist()

        df = self.genes_df[self.genes_df["drug_class"].isin(valid_classes)].copy()
        logger.info(f"Using {len(valid_classes)} drug classes with >= {min_samples_per_class} samples")
        logger.info(f"Total samples: {len(df)}")

        # Get sequences
        sequences = [self.sequences[gid]["sequence"] for gid in df["gene_id"]]

        # Extract features
        X, feature_names = self.extract_kmer_features(
            sequences, k=k, max_features=max_features
        )

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(df["drug_class"])
        class_names = list(le.classes_)
        self.label_encoders["drug_class"] = le

        logger.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")
        logger.info(f"Classes: {class_names}")

        # Split data with stratification
        try:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed ({e}), using random split")
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=random_state
            )

        logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "feature_names": feature_names,
            "class_names": class_names,
            "task_type": "multiclass",
            "metadata": {
                "k": k,
                "max_features": max_features,
                "n_samples": len(df),
                "n_features": X.shape[1],
                "n_classes": len(class_names),
            },
        }

    def prepare_multilabel_antibiotic_data(
        self,
        k: int = 6,
        max_features: int = 500,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        min_samples_per_antibiotic: int = 20,
    ) -> Dict:
        """Prepare dataset for multi-label antibiotic prediction.

        Args:
            k: k-mer size for feature extraction.
            max_features: Maximum number of k-mer features.
            test_size: Proportion for testing.
            val_size: Proportion of training data for validation.
            random_state: Random seed.
            min_samples_per_antibiotic: Minimum samples to include an antibiotic.

        Returns:
            Dictionary with train/val/test splits and metadata.
        """
        if self.genes_df is None:
            self.load_data()

        logger.info("Preparing multi-label antibiotic prediction data...")

        # Filter to genes with phenotype information
        df = self.genes_df[self.genes_df["phenotype"].notna()].copy()
        logger.info(f"Genes with phenotype info: {len(df)}")

        if len(df) == 0:
            raise ValueError("No genes with phenotype information available")

        # Parse antibiotics for each gene
        gene_antibiotics = []
        for _, row in df.iterrows():
            antibiotics = [ab.strip() for ab in str(row["phenotype"]).split(",") if ab.strip()]
            gene_antibiotics.append({
                "gene_id": row["gene_id"],
                "antibiotics": antibiotics,
            })

        # Count antibiotic occurrences
        ab_counts = Counter()
        for ga in gene_antibiotics:
            for ab in ga["antibiotics"]:
                ab_counts[ab] += 1

        # Filter to antibiotics with enough samples
        valid_antibiotics = [ab for ab, count in ab_counts.items() if count >= min_samples_per_antibiotic]
        valid_antibiotics = sorted(valid_antibiotics)
        logger.info(f"Valid antibiotics (>= {min_samples_per_antibiotic} samples): {len(valid_antibiotics)}")

        if len(valid_antibiotics) == 0:
            raise ValueError(f"No antibiotics with >= {min_samples_per_antibiotic} samples")

        # Filter genes to those with at least one valid antibiotic
        filtered_genes = []
        for ga in gene_antibiotics:
            valid_abs = [ab for ab in ga["antibiotics"] if ab in valid_antibiotics]
            if valid_abs:
                filtered_genes.append({
                    "gene_id": ga["gene_id"],
                    "antibiotics": valid_abs,
                })

        logger.info(f"Genes with valid antibiotics: {len(filtered_genes)}")

        # Get sequences and create label matrix
        gene_ids = [g["gene_id"] for g in filtered_genes]
        sequences = [self.sequences[gid]["sequence"] for gid in gene_ids]

        # Extract features
        X, feature_names = self.extract_kmer_features(
            sequences, k=k, max_features=max_features
        )

        # Create multi-label target matrix
        mlb = MultiLabelBinarizer(classes=valid_antibiotics)
        y = mlb.fit_transform([g["antibiotics"] for g in filtered_genes])
        self.label_encoders["antibiotics"] = mlb

        logger.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")

        # Split data (can't stratify with multi-label)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state
        )

        logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "feature_names": feature_names,
            "class_names": valid_antibiotics,
            "task_type": "multilabel",
            "metadata": {
                "k": k,
                "max_features": max_features,
                "n_samples": len(filtered_genes),
                "n_features": X.shape[1],
                "n_antibiotics": len(valid_antibiotics),
            },
        }

    def prepare_mechanism_data(
        self,
        k: int = 6,
        max_features: int = 500,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        min_samples_per_class: int = 20,
    ) -> Dict:
        """Prepare dataset for resistance mechanism prediction.

        Args:
            k: k-mer size for feature extraction.
            max_features: Maximum number of k-mer features.
            test_size: Proportion for testing.
            val_size: Proportion of training data for validation.
            random_state: Random seed.
            min_samples_per_class: Minimum samples to include a mechanism.

        Returns:
            Dictionary with train/val/test splits and metadata.
        """
        if self.genes_df is None:
            self.load_data()

        logger.info("Preparing resistance mechanism prediction data...")

        # Filter to genes with mechanism information
        df = self.genes_df[self.genes_df["mechanism"].notna()].copy()
        df = df[df["mechanism"] != ""].copy()
        logger.info(f"Genes with mechanism info: {len(df)}")

        if len(df) == 0:
            raise ValueError("No genes with mechanism information available")

        # Filter to mechanisms with enough samples
        mech_counts = df["mechanism"].value_counts()
        valid_mechs = mech_counts[mech_counts >= min_samples_per_class].index.tolist()

        df = df[df["mechanism"].isin(valid_mechs)].copy()
        logger.info(f"Valid mechanisms: {len(valid_mechs)}")
        logger.info(f"Samples after filtering: {len(df)}")

        # Get sequences
        sequences = [self.sequences[gid]["sequence"] for gid in df["gene_id"]]

        # Extract features
        X, feature_names = self.extract_kmer_features(
            sequences, k=k, max_features=max_features
        )

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(df["mechanism"])
        class_names = list(le.classes_)
        self.label_encoders["mechanism"] = le

        logger.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")

        # Split data
        try:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed ({e}), using random split")
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=random_state
            )

        logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "feature_names": feature_names,
            "class_names": class_names,
            "task_type": "multiclass",
            "metadata": {
                "k": k,
                "max_features": max_features,
                "n_samples": len(df),
                "n_features": X.shape[1],
                "n_classes": len(class_names),
            },
        }

    def save_processed_data(self, data: Dict, prefix: str = "resfinder") -> None:
        """Save processed data to disk."""
        logger.info(f"Saving processed data to {self.output_dir}")

        # Save numpy arrays
        np.save(self.output_dir / f"{prefix}_X_train.npy", data["X_train"])
        np.save(self.output_dir / f"{prefix}_X_val.npy", data["X_val"])
        np.save(self.output_dir / f"{prefix}_X_test.npy", data["X_test"])
        np.save(self.output_dir / f"{prefix}_y_train.npy", data["y_train"])
        np.save(self.output_dir / f"{prefix}_y_val.npy", data["y_val"])
        np.save(self.output_dir / f"{prefix}_y_test.npy", data["y_test"])

        # Save metadata
        metadata = {
            "feature_names": data["feature_names"],
            "class_names": data["class_names"],
            "task_type": data["task_type"],
            **data["metadata"],
        }
        with open(self.output_dir / f"{prefix}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Data saved successfully!")

    def get_data_summary(self) -> Dict:
        """Get comprehensive summary of the ResFinder dataset."""
        if self.genes_df is None:
            self.load_data()

        summary = {
            "total_genes": len(self.genes_df),
            "total_sequences": len(self.sequences),
            "unique_gene_names": self.genes_df["gene_name"].nunique(),
            "drug_classes": sorted(self.genes_df["drug_class"].unique().tolist()),
            "n_drug_classes": self.genes_df["drug_class"].nunique(),
        }

        # Sequence statistics
        if self.sequences:
            seq_lengths = [len(s["sequence"]) for s in self.sequences.values()]
            summary["sequence_stats"] = {
                "count": len(seq_lengths),
                "min_length": min(seq_lengths),
                "max_length": max(seq_lengths),
                "mean_length": int(np.mean(seq_lengths)),
                "median_length": int(np.median(seq_lengths)),
            }

        # Phenotype stats
        if "phenotype" in self.genes_df.columns:
            summary["genes_with_phenotype"] = self.genes_df["phenotype"].notna().sum()

        # Mechanism stats
        if "mechanism" in self.genes_df.columns:
            summary["genes_with_mechanism"] = self.genes_df["mechanism"].notna().sum()
            summary["unique_mechanisms"] = self.genes_df["mechanism"].dropna().nunique()

        return summary


def main():
    """Main preprocessing pipeline."""
    preprocessor = ResFinderPreprocessor()

    # Load data
    preprocessor.load_data()

    # Show summary
    print("\n=== ResFinder Data Summary ===")
    summary = preprocessor.get_data_summary()
    for key, value in summary.items():
        if key not in ["drug_classes"]:
            print(f"{key}: {value}")

    # Show drug class statistics
    print("\n=== Drug Class Statistics ===")
    stats = preprocessor.get_drug_class_statistics()
    print(stats.to_string(index=False))

    # Prepare drug class prediction data
    print("\n=== Preparing Drug Class Prediction Data ===")
    try:
        data = preprocessor.prepare_drug_class_data(
            k=6,
            max_features=500,
            min_samples_per_class=20,
        )
        preprocessor.save_processed_data(data, prefix="resfinder_drug_class")
        print("Saved drug class prediction data")
    except Exception as e:
        print(f"Error preparing drug class data: {e}")

    # Prepare mechanism prediction data
    print("\n=== Preparing Mechanism Prediction Data ===")
    try:
        data = preprocessor.prepare_mechanism_data(
            k=6,
            max_features=500,
            min_samples_per_class=20,
        )
        preprocessor.save_processed_data(data, prefix="resfinder_mechanism")
        print("Saved mechanism prediction data")
    except Exception as e:
        print(f"Error preparing mechanism data: {e}")

    # Prepare multi-label antibiotic data
    print("\n=== Preparing Multi-label Antibiotic Data ===")
    try:
        data = preprocessor.prepare_multilabel_antibiotic_data(
            k=6,
            max_features=500,
            min_samples_per_antibiotic=20,
        )
        preprocessor.save_processed_data(data, prefix="resfinder_antibiotic")
        print("Saved antibiotic prediction data")
    except Exception as e:
        print(f"Error preparing antibiotic data: {e}")

    print("\n=== Preprocessing Complete ===")
    print(f"Output directory: {preprocessor.output_dir}")


if __name__ == "__main__":
    main()
