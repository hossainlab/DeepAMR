"""PATRIC database preprocessor for AMR prediction modeling."""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PATRICPreprocessor:
    """Preprocess PATRIC data for AMR prediction models."""

    def __init__(
        self,
        patric_dir: str = "data/raw/patric",
        output_dir: str = "data/processed/patric",
    ):
        self.patric_dir = Path(patric_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.amr_phenotypes: Optional[pd.DataFrame] = None
        self.genomes_metadata: Optional[pd.DataFrame] = None
        self.sequences: dict = {}
        self.label_encoders: dict = {}

    def load_data(self) -> None:
        """Load all PATRIC data files."""
        logger.info("Loading PATRIC data...")

        # Load AMR phenotypes
        amr_file = self.patric_dir / "amr_phenotypes.csv"
        if amr_file.exists():
            self.amr_phenotypes = pd.read_csv(amr_file)
            logger.info(f"Loaded {len(self.amr_phenotypes)} AMR phenotype records")
        else:
            raise FileNotFoundError(f"AMR phenotypes file not found: {amr_file}")

        # Load genome metadata
        meta_file = self.patric_dir / "genomes_metadata.csv"
        if meta_file.exists():
            self.genomes_metadata = pd.read_csv(meta_file)
            logger.info(f"Loaded {len(self.genomes_metadata)} genome metadata records")

        # Load genome sequences
        self._load_sequences()

    def _load_sequences(self) -> None:
        """Load genome sequences from FASTA files."""
        genomes_dir = self.patric_dir / "genomes"
        if not genomes_dir.exists():
            logger.warning(f"Genomes directory not found: {genomes_dir}")
            return

        fasta_files = list(genomes_dir.glob("*.fasta"))
        logger.info(f"Found {len(fasta_files)} genome FASTA files")

        for fasta_file in fasta_files:
            genome_id = fasta_file.stem
            sequences = []
            current_seq = []

            with open(fasta_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if current_seq:
                            sequences.append("".join(current_seq))
                            current_seq = []
                    else:
                        current_seq.append(line)
                if current_seq:
                    sequences.append("".join(current_seq))

            # Concatenate all contigs for this genome
            self.sequences[genome_id] = "".join(sequences)

        logger.info(f"Loaded sequences for {len(self.sequences)} genomes")

    def create_amr_dataset(
        self,
        antibiotic: Optional[str] = None,
        min_samples_per_class: int = 10,
    ) -> pd.DataFrame:
        """Create dataset mapping genomes to AMR phenotypes.

        Args:
            antibiotic: Specific antibiotic to filter. If None, uses all.
            min_samples_per_class: Minimum samples per class to include an antibiotic.

        Returns:
            DataFrame with genome_id, antibiotic, phenotype, and sequence.
        """
        if self.amr_phenotypes is None:
            self.load_data()

        df = self.amr_phenotypes.copy()

        # Filter to records with resistance phenotypes (Resistant/Susceptible)
        df = df[df["resistant_phenotype"].isin(["Resistant", "Susceptible"])].copy()
        logger.info(f"Records with R/S phenotypes: {len(df)}")

        # Filter by antibiotic if specified
        if antibiotic:
            df = df[df["antibiotic"] == antibiotic]
            logger.info(f"Records for {antibiotic}: {len(df)}")

        # Convert genome_id to string for matching with sequence keys
        df["genome_id"] = df["genome_id"].astype(str)

        # Add sequence data
        df["sequence"] = df["genome_id"].apply(
            lambda x: self.sequences.get(x, "")
        )

        # Filter to genomes with sequences
        df = df[df["sequence"].str.len() > 0].copy()
        logger.info(f"Records with genome sequences: {len(df)}")

        # Filter antibiotics with enough samples per class
        if not antibiotic:
            valid_antibiotics = []
            for ab in df["antibiotic"].unique():
                ab_df = df[df["antibiotic"] == ab]
                class_counts = ab_df["resistant_phenotype"].value_counts()
                if all(count >= min_samples_per_class for count in class_counts.values):
                    valid_antibiotics.append(ab)

            df = df[df["antibiotic"].isin(valid_antibiotics)]
            logger.info(f"Antibiotics with sufficient samples: {len(valid_antibiotics)}")
            logger.info(f"Valid antibiotics: {valid_antibiotics}")

        return df

    def extract_kmer_features(
        self, sequences: list, k: int = 6, max_features: int = 1000
    ) -> tuple:
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

    def prepare_single_antibiotic_data(
        self,
        antibiotic: str,
        k: int = 6,
        max_features: int = 1000,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> dict:
        """Prepare dataset for a single antibiotic prediction.

        Args:
            antibiotic: Name of the antibiotic.
            k: k-mer size for feature extraction.
            max_features: Maximum number of k-mer features.
            test_size: Proportion for testing.
            val_size: Proportion of training data for validation.
            random_state: Random seed.

        Returns:
            Dictionary with train/val/test splits and metadata.
        """
        logger.info(f"Preparing data for {antibiotic} prediction...")

        # Create dataset for this antibiotic
        df = self.create_amr_dataset(antibiotic=antibiotic, min_samples_per_class=5)

        if len(df) < 20:
            raise ValueError(f"Not enough samples for {antibiotic}: {len(df)}")

        # Get unique genomes (one row per genome for this antibiotic)
        df_unique = df.drop_duplicates(subset=["genome_id"]).copy()
        logger.info(f"Unique genomes: {len(df_unique)}")

        # Extract features
        sequences = df_unique["sequence"].tolist()
        X, feature_names = self.extract_kmer_features(
            sequences, k=k, max_features=max_features
        )

        # Encode labels (binary: Resistant=1, Susceptible=0)
        le = LabelEncoder()
        y = le.fit_transform(df_unique["resistant_phenotype"])
        class_names = list(le.classes_)
        self.label_encoders[antibiotic] = le

        logger.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")
        logger.info(f"Class distribution: {dict(zip(class_names, np.bincount(y)))}")

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
            "task_type": "binary",
            "metadata": {
                "antibiotic": antibiotic,
                "k": k,
                "max_features": max_features,
                "n_samples": len(df_unique),
                "n_features": X.shape[1],
                "n_classes": len(class_names),
            },
        }

    def prepare_multi_antibiotic_data(
        self,
        antibiotics: Optional[list] = None,
        k: int = 6,
        max_features: int = 1000,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> dict:
        """Prepare dataset for multi-label antibiotic resistance prediction.

        Args:
            antibiotics: List of antibiotics to include. If None, uses all valid ones.
            k: k-mer size for feature extraction.
            max_features: Maximum number of k-mer features.
            test_size: Proportion for testing.
            val_size: Proportion of training data for validation.
            random_state: Random seed.

        Returns:
            Dictionary with train/val/test splits and metadata.
        """
        logger.info("Preparing multi-antibiotic prediction data...")

        # Get valid dataset
        df = self.create_amr_dataset(min_samples_per_class=5)

        if antibiotics:
            df = df[df["antibiotic"].isin(antibiotics)]

        # Get unique antibiotics
        unique_antibiotics = sorted(df["antibiotic"].unique())
        logger.info(f"Using {len(unique_antibiotics)} antibiotics")

        # Create genome-level features and multi-label targets
        genome_ids = df["genome_id"].unique()
        logger.info(f"Unique genomes: {len(genome_ids)}")

        # Build genome-antibiotic resistance matrix
        genome_sequences = {}
        genome_labels = {gid: {} for gid in genome_ids}

        for _, row in df.iterrows():
            gid = row["genome_id"]
            ab = row["antibiotic"]
            phenotype = row["resistant_phenotype"]

            if gid not in genome_sequences:
                genome_sequences[gid] = row["sequence"]

            # Store resistance (1) or susceptible (0)
            genome_labels[gid][ab] = 1 if phenotype == "Resistant" else 0

        # Create feature matrix and label matrix
        sequences = [genome_sequences[gid] for gid in genome_ids]
        X, feature_names = self.extract_kmer_features(
            sequences, k=k, max_features=max_features
        )

        # Create multi-label target matrix
        y = np.zeros((len(genome_ids), len(unique_antibiotics)))
        for i, gid in enumerate(genome_ids):
            for j, ab in enumerate(unique_antibiotics):
                if ab in genome_labels[gid]:
                    y[i, j] = genome_labels[gid][ab]

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
            "class_names": unique_antibiotics,
            "task_type": "multilabel",
            "metadata": {
                "antibiotics": unique_antibiotics,
                "k": k,
                "max_features": max_features,
                "n_samples": len(genome_ids),
                "n_features": X.shape[1],
                "n_classes": len(unique_antibiotics),
            },
        }

    def save_processed_data(self, data: dict, prefix: str = "patric") -> None:
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

    def get_antibiotic_statistics(self) -> pd.DataFrame:
        """Get statistics about antibiotics in the dataset."""
        if self.amr_phenotypes is None:
            self.load_data()

        # Filter to R/S phenotypes
        df = self.amr_phenotypes[
            self.amr_phenotypes["resistant_phenotype"].isin(["Resistant", "Susceptible"])
        ].copy()

        # Count by antibiotic
        stats = []
        for ab in df["antibiotic"].unique():
            ab_df = df[df["antibiotic"] == ab]
            r_count = (ab_df["resistant_phenotype"] == "Resistant").sum()
            s_count = (ab_df["resistant_phenotype"] == "Susceptible").sum()
            total = r_count + s_count
            r_ratio = r_count / total if total > 0 else 0

            stats.append({
                "antibiotic": ab,
                "resistant": r_count,
                "susceptible": s_count,
                "total": total,
                "resistance_rate": round(r_ratio, 3),
            })

        stats_df = pd.DataFrame(stats)
        stats_df = stats_df.sort_values("total", ascending=False)
        return stats_df

    def compute_gc_content(self, sequence: str) -> float:
        """Calculate GC content of a DNA sequence.

        Args:
            sequence: DNA sequence string.

        Returns:
            GC content as a fraction (0-1).
        """
        sequence = sequence.upper()
        if len(sequence) == 0:
            return 0.0
        gc_count = sequence.count("G") + sequence.count("C")
        return gc_count / len(sequence)

    def get_sequence_statistics(self) -> pd.DataFrame:
        """Get statistics about genome sequences.

        Returns:
            DataFrame with sequence statistics per genome.
        """
        if not self.sequences:
            self._load_sequences()

        stats = []
        for genome_id, seq in self.sequences.items():
            seq_len = len(seq)
            gc_content = self.compute_gc_content(seq)

            # Count nucleotides
            seq_upper = seq.upper()
            a_count = seq_upper.count("A")
            t_count = seq_upper.count("T")
            g_count = seq_upper.count("G")
            c_count = seq_upper.count("C")
            n_count = seq_upper.count("N")
            other_count = seq_len - a_count - t_count - g_count - c_count - n_count

            stats.append({
                "genome_id": genome_id,
                "length": seq_len,
                "gc_content": round(gc_content, 4),
                "a_count": a_count,
                "t_count": t_count,
                "g_count": g_count,
                "c_count": c_count,
                "n_count": n_count,
                "other_count": other_count,
            })

        return pd.DataFrame(stats).sort_values("length", ascending=False)

    def get_organism_statistics(self) -> pd.DataFrame:
        """Get statistics about organisms in the dataset.

        Returns:
            DataFrame with organism-level statistics.
        """
        if self.genomes_metadata is None:
            self.load_data()

        if self.genomes_metadata is None:
            logger.warning("No genome metadata available")
            return pd.DataFrame()

        # Group by species/organism
        if "species" in self.genomes_metadata.columns:
            group_col = "species"
        elif "organism_name" in self.genomes_metadata.columns:
            group_col = "organism_name"
        else:
            logger.warning("No species or organism_name column found")
            return pd.DataFrame()

        stats = self.genomes_metadata.groupby(group_col).agg({
            "genome_id": "count",
        }).reset_index()
        stats.columns = [group_col, "genome_count"]
        stats = stats.sort_values("genome_count", ascending=False)

        return stats

    def get_phenotype_by_organism(self) -> pd.DataFrame:
        """Get resistance statistics broken down by organism.

        Returns:
            DataFrame with resistance rates per organism and antibiotic.
        """
        if self.amr_phenotypes is None or self.genomes_metadata is None:
            self.load_data()

        # Merge phenotypes with metadata
        df = self.amr_phenotypes.merge(
            self.genomes_metadata[["genome_id", "species"]].drop_duplicates(),
            on="genome_id",
            how="left"
        )

        # Filter to R/S phenotypes
        df = df[df["resistant_phenotype"].isin(["Resistant", "Susceptible"])].copy()

        # Calculate resistance rate per organism-antibiotic pair
        stats = []
        for (species, ab), group in df.groupby(["species", "antibiotic"]):
            r_count = (group["resistant_phenotype"] == "Resistant").sum()
            s_count = (group["resistant_phenotype"] == "Susceptible").sum()
            total = r_count + s_count
            r_rate = r_count / total if total > 0 else 0

            stats.append({
                "species": species,
                "antibiotic": ab,
                "resistant": r_count,
                "susceptible": s_count,
                "total": total,
                "resistance_rate": round(r_rate, 3),
            })

        return pd.DataFrame(stats).sort_values(["species", "total"], ascending=[True, False])

    def extract_combined_features(
        self,
        sequences: List[str],
        k: int = 6,
        max_features: int = 1000,
        include_gc: bool = True,
        include_length: bool = True,
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract k-mer features combined with sequence statistics.

        Args:
            sequences: List of DNA sequences.
            k: k-mer size.
            max_features: Maximum number of k-mer features.
            include_gc: Whether to include GC content feature.
            include_length: Whether to include normalized sequence length.

        Returns:
            Tuple of (feature_matrix, feature_names).
        """
        # Get k-mer features
        kmer_features, kmer_names = self.extract_kmer_features(
            sequences, k=k, max_features=max_features
        )

        feature_names = kmer_names.copy()
        additional_features = []

        if include_gc:
            gc_features = np.array([self.compute_gc_content(seq) for seq in sequences])
            additional_features.append(gc_features.reshape(-1, 1))
            feature_names.append("gc_content")

        if include_length:
            lengths = np.array([len(seq) for seq in sequences])
            # Log-normalize length
            log_lengths = np.log1p(lengths)
            # Scale to 0-1 range
            if log_lengths.max() > log_lengths.min():
                log_lengths = (log_lengths - log_lengths.min()) / (log_lengths.max() - log_lengths.min())
            additional_features.append(log_lengths.reshape(-1, 1))
            feature_names.append("log_length_normalized")

        if additional_features:
            additional_matrix = np.hstack(additional_features)
            feature_matrix = np.hstack([kmer_features, additional_matrix])
        else:
            feature_matrix = kmer_features

        logger.info(f"Combined features shape: {feature_matrix.shape}")
        return feature_matrix, feature_names

    def get_data_summary(self) -> Dict:
        """Get comprehensive summary of the PATRIC dataset.

        Returns:
            Dictionary containing dataset summary statistics.
        """
        if self.amr_phenotypes is None:
            self.load_data()

        summary = {
            "total_amr_records": len(self.amr_phenotypes),
            "total_genomes_with_sequences": len(self.sequences),
            "unique_antibiotics": self.amr_phenotypes["antibiotic"].nunique(),
            "antibiotics_list": sorted(self.amr_phenotypes["antibiotic"].unique().tolist()),
        }

        # R/S phenotype breakdown
        rs_df = self.amr_phenotypes[
            self.amr_phenotypes["resistant_phenotype"].isin(["Resistant", "Susceptible"])
        ]
        summary["resistant_records"] = int((rs_df["resistant_phenotype"] == "Resistant").sum())
        summary["susceptible_records"] = int((rs_df["resistant_phenotype"] == "Susceptible").sum())
        summary["records_with_rs_phenotype"] = len(rs_df)

        # Genome metadata
        if self.genomes_metadata is not None:
            summary["total_genome_metadata_records"] = len(self.genomes_metadata)
            if "species" in self.genomes_metadata.columns:
                summary["unique_species"] = self.genomes_metadata["species"].nunique()
                summary["species_list"] = sorted(
                    self.genomes_metadata["species"].dropna().unique().tolist()
                )

        # Sequence statistics
        if self.sequences:
            seq_lengths = [len(seq) for seq in self.sequences.values()]
            summary["sequence_stats"] = {
                "count": len(seq_lengths),
                "min_length": min(seq_lengths),
                "max_length": max(seq_lengths),
                "mean_length": int(np.mean(seq_lengths)),
                "median_length": int(np.median(seq_lengths)),
            }

        return summary


def main():
    """Main preprocessing pipeline."""
    preprocessor = PATRICPreprocessor()

    # Load data
    preprocessor.load_data()

    # Show statistics
    print("\n=== Antibiotic Statistics ===")
    stats = preprocessor.get_antibiotic_statistics()
    print(stats.head(20).to_string(index=False))

    # Prepare multi-antibiotic prediction data
    print("\n=== Preparing Multi-Antibiotic Prediction Data ===")
    try:
        data = preprocessor.prepare_multi_antibiotic_data(
            k=6,
            max_features=500,
            test_size=0.2,
            val_size=0.1,
        )
        preprocessor.save_processed_data(data, prefix="patric_multilabel")
        print("Saved multi-label prediction data")
    except Exception as e:
        print(f"Error preparing multi-antibiotic data: {e}")

    # Prepare single antibiotic models for top antibiotics
    print("\n=== Preparing Single-Antibiotic Models ===")
    top_antibiotics = stats.head(5)["antibiotic"].tolist()

    for ab in top_antibiotics:
        try:
            print(f"\nProcessing: {ab}")
            data = preprocessor.prepare_single_antibiotic_data(
                antibiotic=ab,
                k=6,
                max_features=500,
            )
            # Clean antibiotic name for filename
            ab_clean = ab.replace("/", "_").replace(" ", "_").lower()
            preprocessor.save_processed_data(data, prefix=f"patric_{ab_clean}")
            print(f"Saved {ab} prediction data")
        except Exception as e:
            print(f"Error preparing {ab} data: {e}")

    print("\n=== Preprocessing Complete ===")
    print(f"Output directory: {preprocessor.output_dir}")


if __name__ == "__main__":
    main()
