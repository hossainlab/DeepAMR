"""NCBI Pathogen Detection database preprocessor for AMR prediction modeling."""

import gzip
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NCBIPreprocessor:
    """Preprocess NCBI Pathogen Detection data for AMR prediction models.

    This preprocessor handles genome sequences and metadata from NCBI's
    Pathogen Detection database. Since AMR phenotype data may be sparse
    or unavailable, it supports:
    1. Organism classification (using available metadata)
    2. AMR phenotype prediction (when data is available)
    3. Feature extraction for downstream modeling
    """

    def __init__(
        self,
        ncbi_dir: str = "data/raw/ncbi",
        output_dir: str = "data/processed/ncbi",
    ):
        self.ncbi_dir = Path(ncbi_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.metadata: Optional[pd.DataFrame] = None
        self.sequences: dict = {}
        self.label_encoders: dict = {}

    def load_data(self) -> None:
        """Load all NCBI data files."""
        logger.info("Loading NCBI Pathogen Detection data...")

        # Load metadata from all organism-specific files
        metadata_dir = self.ncbi_dir / "metadata"
        all_dfs = []

        if metadata_dir.exists():
            for csv_file in metadata_dir.glob("*.csv"):
                if not csv_file.name.startswith("."):
                    df = pd.read_csv(csv_file)
                    all_dfs.append(df)
                    logger.info(f"Loaded {len(df)} records from {csv_file.name}")

        # Also load complete metadata if exists
        complete_file = self.ncbi_dir / "complete_metadata.csv"
        if complete_file.exists():
            df = pd.read_csv(complete_file)
            all_dfs.append(df)
            logger.info(f"Loaded {len(df)} records from complete_metadata.csv")

        if not all_dfs:
            raise FileNotFoundError(f"No metadata files found in {self.ncbi_dir}")

        # Combine and deduplicate
        self.metadata = pd.concat(all_dfs, ignore_index=True)
        self.metadata = self.metadata.drop_duplicates(subset=["biosample_id"])
        logger.info(f"Total unique records: {len(self.metadata)}")

        # Load genome sequences
        self._load_sequences()

    def _load_sequences(self) -> None:
        """Load genome sequences from compressed FASTA files."""
        genomes_dir = self.ncbi_dir / "genomes"
        if not genomes_dir.exists():
            logger.warning(f"Genomes directory not found: {genomes_dir}")
            return

        fasta_files = list(genomes_dir.glob("*.fna.gz"))
        logger.info(f"Found {len(fasta_files)} genome FASTA files")

        for fasta_file in fasta_files:
            # Extract biosample_id from filename (e.g., 54802194.fna.gz)
            biosample_id = fasta_file.stem.replace(".fna", "")

            try:
                sequences = []
                current_seq = []

                with gzip.open(fasta_file, "rt") as f:
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
                self.sequences[biosample_id] = "".join(sequences)
            except Exception as e:
                logger.warning(f"Error loading {fasta_file}: {e}")

        logger.info(f"Loaded sequences for {len(self.sequences)} genomes")

    def create_organism_dataset(
        self,
        min_samples_per_class: int = 10,
    ) -> pd.DataFrame:
        """Create dataset mapping genomes to organisms for classification.

        Args:
            min_samples_per_class: Minimum samples per organism to include.

        Returns:
            DataFrame with biosample_id, organism, and sequence.
        """
        if self.metadata is None:
            self.load_data()

        df = self.metadata.copy()

        # Convert biosample_id to string for matching
        df["biosample_id"] = df["biosample_id"].astype(str)

        # Add sequence data
        df["sequence"] = df["biosample_id"].apply(
            lambda x: self.sequences.get(x, "")
        )

        # Filter to records with sequences
        df = df[df["sequence"].str.len() > 0].copy()
        logger.info(f"Records with genome sequences: {len(df)}")

        # Use organism_query as the organism label
        df["organism"] = df["organism_query"].fillna("unknown")

        # Filter organisms with enough samples
        organism_counts = df["organism"].value_counts()
        valid_organisms = organism_counts[
            organism_counts >= min_samples_per_class
        ].index.tolist()

        df = df[df["organism"].isin(valid_organisms)]
        logger.info(f"Organisms with >= {min_samples_per_class} samples: {len(valid_organisms)}")
        logger.info(f"Records after filtering: {len(df)}")

        return df

    def create_amr_dataset(
        self,
        min_samples_per_class: int = 10,
    ) -> pd.DataFrame:
        """Create dataset mapping genomes to AMR phenotypes.

        Args:
            min_samples_per_class: Minimum samples per class to include.

        Returns:
            DataFrame with AMR data, or empty DataFrame if no AMR data available.
        """
        if self.metadata is None:
            self.load_data()

        df = self.metadata.copy()

        # Check if AMR data is available
        has_amr = (
            df["amr_phenotypes"].notna() & (df["amr_phenotypes"] != "")
        ) | (
            df["amr_genotypes"].notna() & (df["amr_genotypes"] != "")
        )

        amr_count = has_amr.sum()
        if amr_count == 0:
            logger.warning(
                "No AMR phenotype data available in NCBI metadata. "
                "Consider using organism classification instead, or "
                "linking to external AMR databases."
            )
            return pd.DataFrame()

        logger.info(f"Records with AMR data: {amr_count}")
        df = df[has_amr].copy()

        # Convert biosample_id to string for matching
        df["biosample_id"] = df["biosample_id"].astype(str)

        # Add sequence data
        df["sequence"] = df["biosample_id"].apply(
            lambda x: self.sequences.get(x, "")
        )

        # Filter to records with sequences
        df = df[df["sequence"].str.len() > 0].copy()
        logger.info(f"AMR records with genome sequences: {len(df)}")

        return df

    def extract_kmer_features(
        self,
        sequences: List[str],
        k: int = 6,
        max_features: int = 1000,
        sample_size: int = 100000,
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract k-mer frequency features from DNA sequences.

        For large genome sequences, this method samples positions to make
        k-mer extraction tractable while still capturing the sequence signature.

        Args:
            sequences: List of DNA sequences.
            k: k-mer size (default 6 for DNA).
            max_features: Maximum number of k-mer features.
            sample_size: Number of positions to sample per sequence for large genomes.
                        If sequence length <= sample_size, uses entire sequence.

        Returns:
            Tuple of (feature_matrix, feature_names).
        """
        logger.info(f"Extracting {k}-mer features from {len(sequences)} sequences...")

        # Count all k-mers across sequences to find most common
        # Use sampling for large sequences
        all_kmers = Counter()
        for seq_idx, seq in enumerate(sequences):
            seq = seq.upper()
            seq_len = len(seq) - k + 1

            if seq_len <= 0:
                continue

            # Sample positions for large sequences
            if seq_len > sample_size:
                np.random.seed(42 + seq_idx)  # Reproducible sampling
                positions = np.random.choice(seq_len, size=sample_size, replace=False)
            else:
                positions = range(seq_len)

            for i in positions:
                kmer = seq[i : i + k]
                # Valid DNA nucleotides only
                if all(c in "ACGT" for c in kmer):
                    all_kmers[kmer] += 1

            if (seq_idx + 1) % 100 == 0:
                logger.info(f"Processed {seq_idx + 1}/{len(sequences)} sequences for k-mer counting")

        # Select top k-mers as features
        top_kmers = [kmer for kmer, _ in all_kmers.most_common(max_features)]
        logger.info(f"Selected {len(top_kmers)} k-mer features")

        # Create feature matrix using sampling
        feature_matrix = np.zeros((len(sequences), len(top_kmers)))
        kmer_to_idx = {kmer: idx for idx, kmer in enumerate(top_kmers)}

        for seq_idx, seq in enumerate(sequences):
            seq = seq.upper()
            seq_len = len(seq) - k + 1

            if seq_len <= 0:
                continue

            # Sample positions for large sequences
            if seq_len > sample_size:
                np.random.seed(42 + seq_idx)  # Same seed for reproducibility
                positions = np.random.choice(seq_len, size=sample_size, replace=False)
                normalizer = sample_size
            else:
                positions = range(seq_len)
                normalizer = seq_len

            for i in positions:
                kmer = seq[i : i + k]
                if kmer in kmer_to_idx:
                    feature_matrix[seq_idx, kmer_to_idx[kmer]] += 1

            # Normalize by number of k-mers counted
            if normalizer > 0:
                feature_matrix[seq_idx] /= normalizer

            if (seq_idx + 1) % 100 == 0:
                logger.info(f"Processed {seq_idx + 1}/{len(sequences)} sequences for features")

        return feature_matrix, top_kmers

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

        feature_names = list(kmer_names)
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
                log_lengths = (log_lengths - log_lengths.min()) / (
                    log_lengths.max() - log_lengths.min()
                )
            additional_features.append(log_lengths.reshape(-1, 1))
            feature_names.append("log_length_normalized")

        if additional_features:
            additional_matrix = np.hstack(additional_features)
            feature_matrix = np.hstack([kmer_features, additional_matrix])
        else:
            feature_matrix = kmer_features

        logger.info(f"Combined features shape: {feature_matrix.shape}")
        return feature_matrix, feature_names

    def prepare_organism_classification_data(
        self,
        k: int = 6,
        max_features: int = 1000,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        min_samples_per_class: int = 10,
    ) -> dict:
        """Prepare dataset for organism classification.

        Args:
            k: k-mer size for feature extraction.
            max_features: Maximum number of k-mer features.
            test_size: Proportion for testing.
            val_size: Proportion of training data for validation.
            random_state: Random seed.
            min_samples_per_class: Minimum samples per organism.

        Returns:
            Dictionary with train/val/test splits and metadata.
        """
        logger.info("Preparing organism classification data...")

        # Create dataset
        df = self.create_organism_dataset(min_samples_per_class=min_samples_per_class)

        if len(df) < 20:
            raise ValueError(f"Not enough samples: {len(df)}")

        # Extract features
        sequences = df["sequence"].tolist()
        X, feature_names = self.extract_kmer_features(
            sequences, k=k, max_features=max_features
        )

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(df["organism"])
        class_names = list(le.classes_)
        self.label_encoders["organism"] = le

        logger.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")
        logger.info(f"Number of classes: {len(class_names)}")
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
                "target": "organism",
                "k": k,
                "max_features": max_features,
                "n_samples": len(df),
                "n_features": X.shape[1],
                "n_classes": len(class_names),
            },
        }

    def prepare_amr_prediction_data(
        self,
        amr_labels_file: str = "data/raw/ncbi/amrfinder_results/amr_labels.csv",
        k: int = 6,
        max_features: int = 1000,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        min_samples_per_drug: int = 10,
    ) -> dict:
        """Prepare dataset for AMR phenotype prediction using AMRFinderPlus labels.

        This method uses AMR labels generated by AMRFinderPlus to create
        a multi-label classification dataset for predicting resistance to
        different drug classes.

        Args:
            amr_labels_file: Path to AMR labels CSV from AMRFinderPlus.
            k: k-mer size for feature extraction.
            max_features: Maximum number of k-mer features.
            test_size: Proportion for testing.
            val_size: Proportion of training data for validation.
            random_state: Random seed.
            min_samples_per_drug: Minimum samples per drug class to include.

        Returns:
            Dictionary with train/val/test splits and metadata.
        """
        logger.info("Preparing AMR prediction data from AMRFinderPlus labels...")

        amr_labels_path = Path(amr_labels_file)
        if not amr_labels_path.exists():
            raise FileNotFoundError(
                f"AMR labels file not found: {amr_labels_file}\n"
                "Please run AMRFinderPlus first:\n"
                "  from src.data_collection.amrfinder_annotator import AMRFinderAnnotator\n"
                "  annotator = AMRFinderAnnotator()\n"
                "  annotator.run_on_all_genomes()\n"
                "  annotator.create_amr_labels()"
            )

        # Load AMR labels
        amr_labels = pd.read_csv(amr_labels_path)
        amr_labels["biosample_id"] = amr_labels["biosample_id"].astype(str)
        logger.info(f"Loaded AMR labels for {len(amr_labels)} samples")

        # Get drug columns (all columns except biosample_id)
        drug_columns = [c for c in amr_labels.columns if c != "biosample_id"]

        # Filter drugs with enough samples
        drug_counts = amr_labels[drug_columns].sum()
        valid_drugs = drug_counts[drug_counts >= min_samples_per_drug].index.tolist()

        if not valid_drugs:
            raise ValueError(
                f"No drug classes with >= {min_samples_per_drug} resistant samples. "
                f"Drug counts: {dict(drug_counts)}"
            )

        logger.info(f"Using {len(valid_drugs)} drug classes with >= {min_samples_per_drug} samples")
        for drug in valid_drugs:
            logger.info(f"  {drug}: {drug_counts[drug]} resistant samples")

        # Load sequences
        if self.metadata is None:
            self.load_data()

        # Merge with sequences
        self.metadata["biosample_id"] = self.metadata["biosample_id"].astype(str)
        merged = amr_labels.merge(
            self.metadata[["biosample_id"]],
            on="biosample_id",
            how="inner",
        )

        # Add sequences
        merged["sequence"] = merged["biosample_id"].apply(
            lambda x: self.sequences.get(x, "")
        )

        # Filter to records with sequences
        merged = merged[merged["sequence"].str.len() > 0].copy()
        logger.info(f"Samples with sequences: {len(merged)}")

        if len(merged) < 20:
            raise ValueError(f"Not enough samples with sequences: {len(merged)}")

        # Extract features
        sequences = merged["sequence"].tolist()
        X, feature_names = self.extract_kmer_features(
            sequences, k=k, max_features=max_features
        )

        # Create multi-label target matrix
        y = merged[valid_drugs].values.astype(int)

        logger.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")
        logger.info(f"Drug classes: {valid_drugs}")

        # Split data (random split for multi-label)
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
            "class_names": valid_drugs,
            "task_type": "multilabel",
            "metadata": {
                "target": "amr_drug_class",
                "k": k,
                "max_features": max_features,
                "n_samples": len(merged),
                "n_features": X.shape[1],
                "n_classes": len(valid_drugs),
                "drug_classes": valid_drugs,
            },
        }

    def save_processed_data(self, data: dict, prefix: str = "ncbi") -> None:
        """Save processed data to disk.

        Args:
            data: Dictionary with train/val/test splits and metadata.
            prefix: Filename prefix for saved files.
        """
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

    def get_organism_statistics(self) -> pd.DataFrame:
        """Get statistics about organisms in the dataset.

        Returns:
            DataFrame with organism-level statistics.
        """
        if self.metadata is None:
            self.load_data()

        # Count by organism
        stats = self.metadata["organism_query"].value_counts().reset_index()
        stats.columns = ["organism", "total_records"]

        # Count with sequences
        seq_counts = {}
        for org in stats["organism"]:
            org_df = self.metadata[self.metadata["organism_query"] == org].copy()
            org_df["biosample_id"] = org_df["biosample_id"].astype(str)
            with_seq = sum(1 for bid in org_df["biosample_id"] if bid in self.sequences)
            seq_counts[org] = with_seq

        stats["with_sequences"] = stats["organism"].map(seq_counts)
        stats = stats.sort_values("total_records", ascending=False)

        return stats

    def get_sequence_statistics(self) -> pd.DataFrame:
        """Get statistics about genome sequences.

        Returns:
            DataFrame with sequence statistics.
        """
        if not self.sequences:
            self._load_sequences()

        stats = []
        for biosample_id, seq in self.sequences.items():
            seq_len = len(seq)
            gc_content = self.compute_gc_content(seq)

            # Count nucleotides
            seq_upper = seq.upper()
            a_count = seq_upper.count("A")
            t_count = seq_upper.count("T")
            g_count = seq_upper.count("G")
            c_count = seq_upper.count("C")
            n_count = seq_upper.count("N")

            stats.append({
                "biosample_id": biosample_id,
                "length": seq_len,
                "gc_content": round(gc_content, 4),
                "a_count": a_count,
                "t_count": t_count,
                "g_count": g_count,
                "c_count": c_count,
                "n_count": n_count,
            })

        return pd.DataFrame(stats).sort_values("length", ascending=False)

    def get_metadata_statistics(self) -> Dict:
        """Get comprehensive summary of the NCBI dataset.

        Returns:
            Dictionary containing dataset summary statistics.
        """
        if self.metadata is None:
            self.load_data()

        summary = {
            "total_records": len(self.metadata),
            "total_genomes_with_sequences": len(self.sequences),
            "unique_organisms": self.metadata["organism_query"].nunique(),
            "organisms_list": sorted(
                self.metadata["organism_query"].dropna().unique().tolist()
            ),
        }

        # AMR data availability
        has_amr_geno = (
            self.metadata["amr_genotypes"].notna()
            & (self.metadata["amr_genotypes"] != "")
        ).sum()
        has_amr_pheno = (
            self.metadata["amr_phenotypes"].notna()
            & (self.metadata["amr_phenotypes"] != "")
        ).sum()

        summary["records_with_amr_genotypes"] = int(has_amr_geno)
        summary["records_with_amr_phenotypes"] = int(has_amr_pheno)

        # Assembly availability
        has_assembly = (
            self.metadata["assembly_accession"].notna()
            & (self.metadata["assembly_accession"] != "")
        ).sum()
        summary["records_with_assembly"] = int(has_assembly)

        # Geographic distribution
        if "geo_loc_name" in self.metadata.columns:
            summary["unique_locations"] = self.metadata["geo_loc_name"].nunique()

        # Isolation source distribution
        if "isolation_source" in self.metadata.columns:
            summary["unique_isolation_sources"] = self.metadata[
                "isolation_source"
            ].nunique()

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

    def get_geographic_distribution(self) -> pd.DataFrame:
        """Get geographic distribution of samples.

        Returns:
            DataFrame with location counts.
        """
        if self.metadata is None:
            self.load_data()

        if "geo_loc_name" not in self.metadata.columns:
            return pd.DataFrame()

        stats = self.metadata["geo_loc_name"].value_counts().reset_index()
        stats.columns = ["location", "count"]
        return stats

    def get_isolation_source_distribution(self) -> pd.DataFrame:
        """Get distribution of isolation sources.

        Returns:
            DataFrame with isolation source counts.
        """
        if self.metadata is None:
            self.load_data()

        if "isolation_source" not in self.metadata.columns:
            return pd.DataFrame()

        stats = self.metadata["isolation_source"].value_counts().reset_index()
        stats.columns = ["isolation_source", "count"]
        return stats


def main():
    """Main preprocessing pipeline."""
    preprocessor = NCBIPreprocessor()

    # Load data
    preprocessor.load_data()

    # Show statistics
    print("\n=== Dataset Summary ===")
    summary = preprocessor.get_metadata_statistics()
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, list):
            print(f"{key}: {len(value)} items")
            if len(value) <= 10:
                for item in value:
                    print(f"  - {item}")
        else:
            print(f"{key}: {value}")

    print("\n=== Organism Statistics ===")
    org_stats = preprocessor.get_organism_statistics()
    print(org_stats.to_string(index=False))

    # Prepare organism classification data
    print("\n=== Preparing Organism Classification Data ===")
    try:
        data = preprocessor.prepare_organism_classification_data(
            k=6,
            max_features=500,
            test_size=0.2,
            val_size=0.1,
            min_samples_per_class=5,
        )
        preprocessor.save_processed_data(data, prefix="ncbi_organism")
        print("Saved organism classification data")

        # Print class distribution
        print("\nClass distribution in training set:")
        train_counts = np.bincount(data["y_train"])
        for i, count in enumerate(train_counts):
            print(f"  {data['class_names'][i]}: {count}")

    except Exception as e:
        print(f"Error preparing organism classification data: {e}")

    # Check AMR data availability
    print("\n=== AMR Data Status ===")
    if summary.get("records_with_amr_phenotypes", 0) > 0:
        print("AMR phenotype data available - can prepare AMR prediction data")
    else:
        print(
            "No AMR phenotype data available in NCBI metadata.\n"
            "Options:\n"
            "  1. Use organism classification as a proxy task\n"
            "  2. Link to external AMR databases (PATRIC, CARD)\n"
            "  3. Wait for AMR data to be added to NCBI records"
        )

    print("\n=== Preprocessing Complete ===")
    print(f"Output directory: {preprocessor.output_dir}")


if __name__ == "__main__":
    main()
