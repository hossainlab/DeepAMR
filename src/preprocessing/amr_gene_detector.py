"""AMR gene detection using ResFinder database.

This module detects antimicrobial resistance genes in genome sequences
by searching for ResFinder reference sequences. No external tools required.
"""

import gzip
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AMRGeneDetector:
    """Detect AMR genes in genomes using ResFinder database.

    Uses k-mer based sequence matching to identify resistance genes
    without requiring external alignment tools.
    """

    def __init__(
        self,
        resfinder_dir: str = "data/raw/resfinder",
        card_dir: str = "data/raw/card-data",
        kmer_size: int = 31,
        min_identity: float = 0.8,
    ):
        """Initialize AMR gene detector.

        Args:
            resfinder_dir: Path to ResFinder database
            card_dir: Path to CARD database
            kmer_size: Size of k-mers for matching (default 31)
            min_identity: Minimum fraction of k-mers that must match (default 0.8)
        """
        self.resfinder_dir = Path(resfinder_dir)
        self.card_dir = Path(card_dir)
        self.kmer_size = kmer_size
        self.min_identity = min_identity

        # Reference data
        self.resistance_genes: Dict[str, Dict] = {}  # gene_name -> {sequence, drug_class, ...}
        self.drug_classes: Set[str] = set()
        self.gene_kmers: Dict[str, Set[str]] = {}  # gene_name -> set of k-mers

    def load_resfinder_database(self) -> None:
        """Load resistance genes from ResFinder database."""
        logger.info("Loading ResFinder database...")

        # Map of file names to drug classes
        drug_class_files = {
            "aminoglycoside.fsa": "aminoglycoside",
            "beta-lactam.fsa": "beta-lactam",
            "colistin.fsa": "colistin",
            "fosfomycin.fsa": "fosfomycin",
            "fusidicacid.fsa": "fusidic_acid",
            "glycopeptide.fsa": "glycopeptide",
            "macrolide.fsa": "macrolide",
            "nitroimidazole.fsa": "nitroimidazole",
            "oxazolidinone.fsa": "oxazolidinone",
            "phenicol.fsa": "phenicol",
            "quinolone.fsa": "quinolone",
            "rifampicin.fsa": "rifampicin",
            "sulphonamide.fsa": "sulfonamide",
            "tetracycline.fsa": "tetracycline",
            "trimethoprim.fsa": "trimethoprim",
        }

        for filename, drug_class in drug_class_files.items():
            fasta_path = self.resfinder_dir / filename
            if fasta_path.exists():
                genes = self._parse_fasta(fasta_path)
                for gene_name, sequence in genes.items():
                    self.resistance_genes[gene_name] = {
                        "sequence": sequence,
                        "drug_class": drug_class,
                        "source": "resfinder",
                    }
                    self.drug_classes.add(drug_class)
                logger.info(f"  Loaded {len(genes)} genes from {filename}")

        logger.info(f"Total resistance genes loaded: {len(self.resistance_genes)}")
        logger.info(f"Drug classes: {sorted(self.drug_classes)}")

    def _parse_fasta(self, fasta_path: Path) -> Dict[str, str]:
        """Parse a FASTA file and return gene sequences."""
        genes = {}
        current_gene = None
        current_seq = []

        with open(fasta_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_gene and current_seq:
                        genes[current_gene] = "".join(current_seq).upper()
                    # Parse gene name from header
                    # Format: >gene_name_variant additional info
                    header = line[1:].split()[0]
                    current_gene = header
                    current_seq = []
                else:
                    current_seq.append(line)

            if current_gene and current_seq:
                genes[current_gene] = "".join(current_seq).upper()

        return genes

    def build_kmer_index(self) -> None:
        """Build k-mer index for all resistance genes."""
        logger.info(f"Building {self.kmer_size}-mer index for {len(self.resistance_genes)} genes...")

        for gene_name, gene_data in self.resistance_genes.items():
            seq = gene_data["sequence"]
            kmers = set()
            for i in range(len(seq) - self.kmer_size + 1):
                kmer = seq[i:i + self.kmer_size]
                if all(c in "ACGT" for c in kmer):
                    kmers.add(kmer)
                    # Also add reverse complement
                    kmers.add(self._reverse_complement(kmer))
            self.gene_kmers[gene_name] = kmers

        logger.info("K-mer index built successfully")

    @staticmethod
    def _reverse_complement(seq: str) -> str:
        """Get reverse complement of a DNA sequence."""
        complement = {"A": "T", "T": "A", "G": "C", "C": "G"}
        return "".join(complement.get(b, "N") for b in reversed(seq))

    def detect_genes_in_sequence(
        self,
        sequence: str,
        min_coverage: float = 0.5,
    ) -> List[Dict]:
        """Detect resistance genes in a genome sequence.

        Args:
            sequence: Genome sequence (DNA)
            min_coverage: Minimum fraction of gene k-mers that must be found

        Returns:
            List of detected genes with metadata
        """
        sequence = sequence.upper()

        # Build k-mer set for the genome
        genome_kmers = set()
        for i in range(len(sequence) - self.kmer_size + 1):
            kmer = sequence[i:i + self.kmer_size]
            if all(c in "ACGT" for c in kmer):
                genome_kmers.add(kmer)

        detected = []
        for gene_name, gene_kmers in self.gene_kmers.items():
            if not gene_kmers:
                continue

            # Count matching k-mers
            matching = len(gene_kmers & genome_kmers)
            coverage = matching / len(gene_kmers)

            if coverage >= min_coverage:
                gene_data = self.resistance_genes[gene_name]
                detected.append({
                    "gene": gene_name,
                    "drug_class": gene_data["drug_class"],
                    "coverage": round(coverage, 3),
                    "matching_kmers": matching,
                    "total_kmers": len(gene_kmers),
                })

        return detected

    def annotate_genomes(
        self,
        genomes_dir: str = "data/raw/ncbi/genomes",
        metadata_file: str = "data/raw/ncbi/complete_metadata.csv",
        output_dir: str = "data/raw/ncbi/amr_annotations",
        min_coverage: float = 0.5,
        max_genomes: Optional[int] = None,
    ) -> pd.DataFrame:
        """Annotate all genomes with AMR gene detection.

        Args:
            genomes_dir: Directory containing genome FASTA files
            metadata_file: Path to metadata CSV
            output_dir: Directory to save results
            min_coverage: Minimum k-mer coverage for gene detection
            max_genomes: Maximum number of genomes to process (for testing)

        Returns:
            DataFrame with AMR annotations for all genomes
        """
        genomes_path = Path(genomes_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load database and build index
        if not self.resistance_genes:
            self.load_resfinder_database()
        if not self.gene_kmers:
            self.build_kmer_index()

        # Get genome files
        genome_files = list(genomes_path.glob("*.fna.gz"))
        if max_genomes:
            genome_files = genome_files[:max_genomes]

        logger.info(f"Processing {len(genome_files)} genomes...")

        all_annotations = []
        genome_drug_classes = {}  # biosample_id -> set of drug classes

        for i, genome_file in enumerate(genome_files):
            biosample_id = genome_file.stem.replace(".fna", "")

            try:
                # Load genome sequence
                with gzip.open(genome_file, "rt") as f:
                    sequences = []
                    current_seq = []
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

                    full_sequence = "".join(sequences)

                # Detect genes
                detected = self.detect_genes_in_sequence(full_sequence, min_coverage)

                # Record drug classes for this genome
                drug_classes_found = set()
                for gene_info in detected:
                    gene_info["biosample_id"] = biosample_id
                    all_annotations.append(gene_info)
                    drug_classes_found.add(gene_info["drug_class"])

                genome_drug_classes[biosample_id] = drug_classes_found

            except Exception as e:
                logger.warning(f"Error processing {biosample_id}: {e}")

            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(genome_files)} genomes")

        # Create annotations DataFrame
        if all_annotations:
            annotations_df = pd.DataFrame(all_annotations)
            annotations_df.to_csv(output_path / "all_amr_annotations.csv", index=False)
            logger.info(f"Total AMR genes detected: {len(annotations_df)}")
        else:
            annotations_df = pd.DataFrame()
            logger.warning("No AMR genes detected in any genome")

        # Create labels DataFrame (multi-label format)
        labels_data = []
        for biosample_id, drug_classes in genome_drug_classes.items():
            row = {"biosample_id": biosample_id}
            for dc in sorted(self.drug_classes):
                row[dc] = 1 if dc in drug_classes else 0
            labels_data.append(row)

        if labels_data:
            labels_df = pd.DataFrame(labels_data)
            labels_df.to_csv(output_path / "amr_labels.csv", index=False)

            # Summary statistics
            logger.info("\nAMR Detection Summary:")
            logger.info(f"  Genomes processed: {len(genome_drug_classes)}")
            logger.info(f"  Genomes with AMR genes: {sum(1 for dc in genome_drug_classes.values() if dc)}")
            logger.info("\n  Resistance by drug class:")
            for dc in sorted(self.drug_classes):
                count = labels_df[dc].sum()
                logger.info(f"    {dc}: {count} genomes")
        else:
            labels_df = pd.DataFrame()

        # Save drug class mapping
        drug_mapping = {dc: i for i, dc in enumerate(sorted(self.drug_classes))}
        with open(output_path / "drug_class_mapping.json", "w") as f:
            json.dump(drug_mapping, f, indent=2)

        return labels_df

    def get_labels(self, output_dir: str = "data/raw/ncbi/amr_annotations") -> pd.DataFrame:
        """Load AMR labels from saved file."""
        labels_file = Path(output_dir) / "amr_labels.csv"
        if labels_file.exists():
            return pd.read_csv(labels_file)
        else:
            raise FileNotFoundError(
                f"Labels file not found: {labels_file}\n"
                "Run annotate_genomes() first."
            )


def main():
    """Run AMR gene detection on NCBI genomes."""
    print("=" * 60)
    print("AMR Gene Detection using ResFinder Database")
    print("=" * 60)

    detector = AMRGeneDetector(
        kmer_size=31,
        min_identity=0.8,
    )

    # Load database
    print("\nLoading ResFinder database...")
    detector.load_resfinder_database()

    # Build k-mer index
    print("\nBuilding k-mer index...")
    detector.build_kmer_index()

    # Annotate genomes
    print("\nAnnotating NCBI genomes...")
    labels_df = detector.annotate_genomes(
        min_coverage=0.5,
    )

    if len(labels_df) > 0:
        print("\n" + "=" * 60)
        print("Detection Complete!")
        print("=" * 60)
        print(f"\nResults saved to: data/raw/ncbi/amr_annotations/")
        print(f"  - amr_labels.csv: Multi-label resistance matrix")
        print(f"  - all_amr_annotations.csv: Detailed gene annotations")
        print(f"  - drug_class_mapping.json: Drug class to index mapping")
    else:
        print("\nNo AMR genes detected. Check genome files.")


if __name__ == "__main__":
    main()
