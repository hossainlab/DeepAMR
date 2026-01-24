"""CARD database preprocessor for AMR prediction modeling."""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CARDPreprocessor:
    """Preprocess CARD database for AMR prediction models."""

    def __init__(
        self,
        card_dir: str = "data/raw/card-data",
        output_dir: str = "data/processed/card",
    ):
        self.card_dir = Path(card_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.aro_index: Optional[pd.DataFrame] = None
        self.aro_categories: Optional[pd.DataFrame] = None
        self.sequences: dict = {}
        self.label_encoders: dict = {}

    def load_data(self) -> None:
        """Load all CARD data files."""
        logger.info("Loading CARD data...")

        # Load ARO index (main gene-drug-mechanism mapping)
        self.aro_index = pd.read_csv(
            self.card_dir / "aro_index.tsv", sep="\t", low_memory=False
        )
        logger.info(f"Loaded {len(self.aro_index)} ARO entries")

        # Load ARO categories index (detailed categorization)
        self.aro_categories = pd.read_csv(
            self.card_dir / "aro_categories_index.tsv", sep="\t", low_memory=False
        )
        logger.info(f"Loaded {len(self.aro_categories)} category mappings")

        # Load sequences from FASTA files
        self._load_sequences()

    def _load_sequences(self) -> None:
        """Load protein and nucleotide sequences from FASTA files."""
        fasta_files = {
            "protein_homolog": "protein_fasta_protein_homolog_model.fasta",
            "protein_variant": "protein_fasta_protein_variant_model.fasta",
            "protein_knockout": "protein_fasta_protein_knockout_model.fasta",
            "protein_overexpression": "protein_fasta_protein_overexpression_model.fasta",
            "nucleotide_homolog": "nucleotide_fasta_protein_homolog_model.fasta",
            "nucleotide_variant": "nucleotide_fasta_protein_variant_model.fasta",
        }

        for seq_type, filename in fasta_files.items():
            fasta_path = self.card_dir / filename
            if fasta_path.exists():
                self.sequences[seq_type] = {}
                for record in SeqIO.parse(fasta_path, "fasta"):
                    # Extract ARO accession from header
                    # Format: ">gb|ACCESSION|ARO:XXXXXX|NAME [Species]"
                    header_parts = record.description.split("|")
                    aro_acc = None
                    for part in header_parts:
                        if part.startswith("ARO:"):
                            aro_acc = part.strip()
                            break
                    if aro_acc:
                        self.sequences[seq_type][aro_acc] = str(record.seq)
                logger.info(f"Loaded {len(self.sequences[seq_type])} {seq_type} sequences")

    def create_drug_resistance_dataset(self) -> pd.DataFrame:
        """Create dataset mapping genes to drug classes and resistance mechanisms."""
        if self.aro_index is None:
            self.load_data()

        df = self.aro_index.copy()

        # Clean and standardize drug classes
        df["Drug Class"] = df["Drug Class"].fillna("unknown")
        df["Drug Classes"] = df["Drug Class"].apply(self._split_drug_classes)

        # Clean resistance mechanisms
        df["Resistance Mechanism"] = df["Resistance Mechanism"].fillna("unknown")

        # Clean gene families
        df["AMR Gene Family"] = df["AMR Gene Family"].fillna("unknown")

        # Add sequence data
        df["protein_sequence"] = df["ARO Accession"].apply(
            lambda x: self.sequences.get("protein_homolog", {}).get(x, "")
        )
        df["nucleotide_sequence"] = df["ARO Accession"].apply(
            lambda x: self.sequences.get("nucleotide_homolog", {}).get(x, "")
        )

        # Filter entries with sequences
        df_with_seq = df[df["protein_sequence"].str.len() > 0].copy()
        logger.info(f"Entries with protein sequences: {len(df_with_seq)}")

        return df_with_seq

    def _split_drug_classes(self, drug_string: str) -> list:
        """Split drug class string into list."""
        if pd.isna(drug_string) or drug_string == "unknown":
            return []
        return [d.strip() for d in drug_string.split(";")]

    def extract_kmer_features(
        self, sequences: list, k: int = 3, max_features: int = 1000
    ) -> np.ndarray:
        """Extract k-mer frequency features from sequences."""
        logger.info(f"Extracting {k}-mer features from {len(sequences)} sequences...")

        # Count all k-mers across sequences to find most common
        all_kmers = Counter()
        for seq in sequences:
            seq = seq.upper()
            for i in range(len(seq) - k + 1):
                kmer = seq[i : i + k]
                if all(c in "ACDEFGHIKLMNPQRSTVWY" for c in kmer):  # Valid amino acids
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

    def encode_labels(
        self, df: pd.DataFrame, target_col: str = "Resistance Mechanism"
    ) -> tuple:
        """Encode categorical labels."""
        le = LabelEncoder()
        labels = le.fit_transform(df[target_col])
        self.label_encoders[target_col] = le
        return labels, le.classes_

    def encode_multilabels(
        self, df: pd.DataFrame, target_col: str = "Drug Classes"
    ) -> tuple:
        """Encode multi-label targets (e.g., multiple drug classes)."""
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(df[target_col])
        self.label_encoders[target_col] = mlb
        return labels, mlb.classes_

    def prepare_modeling_data(
        self,
        target: str = "drug_class",
        k: int = 3,
        max_features: int = 1000,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> dict:
        """Prepare complete dataset for modeling.

        Args:
            target: 'drug_class', 'mechanism', or 'gene_family'
            k: k-mer size for feature extraction
            max_features: maximum number of k-mer features
            test_size: proportion of data for testing
            val_size: proportion of training data for validation
            random_state: random seed for reproducibility

        Returns:
            Dictionary with train/val/test splits and metadata
        """
        logger.info(f"Preparing modeling data with target: {target}")

        # Create base dataset
        df = self.create_drug_resistance_dataset()

        # Extract features
        sequences = df["protein_sequence"].tolist()
        X, feature_names = self.extract_kmer_features(sequences, k=k, max_features=max_features)

        # Encode labels based on target
        if target == "drug_class":
            y, class_names = self.encode_multilabels(df, "Drug Classes")
            task_type = "multilabel"
        elif target == "mechanism":
            y, class_names = self.encode_labels(df, "Resistance Mechanism")
            task_type = "multiclass"
        elif target == "gene_family":
            y, class_names = self.encode_labels(df, "AMR Gene Family")
            task_type = "multiclass"
        else:
            raise ValueError(f"Unknown target: {target}")

        logger.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")
        logger.info(f"Number of classes: {len(class_names)}")

        # Split data (try stratified, fall back to random if classes too small)
        try:
            stratify = y if task_type == "multiclass" else None
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify
            )
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp if task_type == "multiclass" else None
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

        # Prepare result
        result = {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "feature_names": feature_names,
            "class_names": list(class_names),
            "task_type": task_type,
            "metadata": {
                "target": target,
                "k": k,
                "max_features": max_features,
                "n_samples": len(df),
                "n_features": X.shape[1],
                "n_classes": len(class_names),
            },
        }

        return result

    def save_processed_data(self, data: dict, prefix: str = "card") -> None:
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

    def get_drug_class_statistics(self) -> pd.DataFrame:
        """Get statistics about drug classes in the dataset."""
        if self.aro_index is None:
            self.load_data()

        # Count drug classes
        drug_counts = Counter()
        for drugs in self.aro_index["Drug Class"].dropna():
            for drug in drugs.split(";"):
                drug_counts[drug.strip()] += 1

        stats = pd.DataFrame(
            [{"Drug Class": drug, "Count": count} for drug, count in drug_counts.most_common()]
        )
        return stats

    def get_mechanism_statistics(self) -> pd.DataFrame:
        """Get statistics about resistance mechanisms."""
        if self.aro_index is None:
            self.load_data()

        stats = self.aro_index["Resistance Mechanism"].value_counts().reset_index()
        stats.columns = ["Resistance Mechanism", "Count"]
        return stats


def main():
    """Main preprocessing pipeline."""
    preprocessor = CARDPreprocessor()

    # Load data
    preprocessor.load_data()

    # Show statistics
    print("\n=== Drug Class Statistics ===")
    drug_stats = preprocessor.get_drug_class_statistics()
    print(drug_stats.head(20))

    print("\n=== Resistance Mechanism Statistics ===")
    mech_stats = preprocessor.get_mechanism_statistics()
    print(mech_stats)

    # Prepare data for different modeling tasks
    for target in ["mechanism", "drug_class", "gene_family"]:
        print(f"\n=== Preparing {target} prediction data ===")
        try:
            data = preprocessor.prepare_modeling_data(
                target=target,
                k=3,
                max_features=500,
                test_size=0.2,
                val_size=0.1,
            )
            preprocessor.save_processed_data(data, prefix=f"card_{target}")
            print(f"Saved {target} prediction data")
        except Exception as e:
            print(f"Error preparing {target} data: {e}")

    print("\n=== Preprocessing Complete ===")
    print(f"Output directory: {preprocessor.output_dir}")


if __name__ == "__main__":
    main()
