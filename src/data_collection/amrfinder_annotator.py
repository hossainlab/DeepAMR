"""AMRFinderPlus integration for annotating NCBI genomes with AMR predictions.

This module runs AMRFinderPlus on genome sequences to detect AMR genes
and predict resistance phenotypes, which can then be used as labels
for machine learning models.
"""

import gzip
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AMRFinderAnnotator:
    """Annotate genomes with AMR predictions using AMRFinderPlus."""

    # Mapping of organisms to AMRFinderPlus organism codes
    ORGANISM_CODES = {
        "Acinetobacter baumannii": "Acinetobacter_baumannii",
        "Campylobacter jejuni": "Campylobacter",
        "Campylobacter coli": "Campylobacter",
        "Clostridioides difficile": "Clostridioides_difficile",
        "Enterococcus faecalis": "Enterococcus_faecalis",
        "Enterococcus faecium": "Enterococcus_faecium",
        "Escherichia coli": "Escherichia",
        "Klebsiella pneumoniae": "Klebsiella_pneumoniae",
        "Neisseria gonorrhoeae": "Neisseria_gonorrhoeae",
        "Neisseria meningitidis": "Neisseria_meningitidis",
        "Pseudomonas aeruginosa": "Pseudomonas_aeruginosa",
        "Salmonella enterica": "Salmonella",
        "Staphylococcus aureus": "Staphylococcus_aureus",
        "Staphylococcus pseudintermedius": "Staphylococcus_pseudintermedius",
        "Streptococcus agalactiae": "Streptococcus_agalactiae",
        "Streptococcus pneumoniae": "Streptococcus_pneumoniae",
        "Streptococcus pyogenes": "Streptococcus_pyogenes",
        "Vibrio cholerae": "Vibrio_cholerae",
    }

    def __init__(
        self,
        genomes_dir: str = "data/raw/ncbi/genomes",
        metadata_file: str = "data/raw/ncbi/complete_metadata.csv",
        output_dir: str = "data/raw/ncbi/amrfinder_results",
    ):
        self.genomes_dir = Path(genomes_dir)
        self.metadata_file = Path(metadata_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metadata: Optional[pd.DataFrame] = None
        self.amr_results: Dict[str, pd.DataFrame] = {}

    @staticmethod
    def check_installation() -> bool:
        """Check if AMRFinderPlus is installed."""
        try:
            result = subprocess.run(
                ["amrfinder", "--version"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info(f"AMRFinderPlus version: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            pass
        return False

    @staticmethod
    def install_amrfinder() -> bool:
        """Install AMRFinderPlus using conda."""
        logger.info("Installing AMRFinderPlus via conda...")
        try:
            # Try conda install
            result = subprocess.run(
                ["conda", "install", "-y", "-c", "bioconda", "ncbi-amrfinderplus"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info("AMRFinderPlus installed successfully")
                # Update database
                subprocess.run(["amrfinder", "-u"], capture_output=True)
                return True
            else:
                logger.error(f"Conda install failed: {result.stderr}")
        except FileNotFoundError:
            logger.error("Conda not found. Please install conda first.")
        return False

    @staticmethod
    def update_database() -> bool:
        """Update AMRFinderPlus database."""
        logger.info("Updating AMRFinderPlus database...")
        try:
            result = subprocess.run(
                ["amrfinder", "-u"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info("Database updated successfully")
                return True
            else:
                logger.warning(f"Database update warning: {result.stderr}")
                return True  # May already be up to date
        except Exception as e:
            logger.error(f"Database update failed: {e}")
            return False

    def load_metadata(self) -> pd.DataFrame:
        """Load genome metadata."""
        if self.metadata is None:
            # Load from all metadata files
            metadata_dir = self.metadata_file.parent / "metadata"
            all_dfs = []

            if metadata_dir.exists():
                for csv_file in metadata_dir.glob("*.csv"):
                    if not csv_file.name.startswith("."):
                        df = pd.read_csv(csv_file)
                        all_dfs.append(df)

            if self.metadata_file.exists():
                df = pd.read_csv(self.metadata_file)
                all_dfs.append(df)

            if all_dfs:
                self.metadata = pd.concat(all_dfs, ignore_index=True)
                self.metadata = self.metadata.drop_duplicates(subset=["biosample_id"])
                self.metadata["biosample_id"] = self.metadata["biosample_id"].astype(str)
                logger.info(f"Loaded metadata for {len(self.metadata)} samples")
            else:
                raise FileNotFoundError("No metadata files found")

        return self.metadata

    def get_organism_for_sample(self, biosample_id: str) -> Optional[str]:
        """Get AMRFinderPlus organism code for a sample."""
        if self.metadata is None:
            self.load_metadata()

        row = self.metadata[self.metadata["biosample_id"] == biosample_id]
        if len(row) == 0:
            return None

        organism = row.iloc[0].get("organism_query", "")
        return self.ORGANISM_CODES.get(organism)

    def run_amrfinder_on_genome(
        self,
        genome_file: Path,
        biosample_id: str,
        organism_code: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Run AMRFinderPlus on a single genome.

        Args:
            genome_file: Path to genome FASTA file (can be gzipped)
            biosample_id: Sample identifier
            organism_code: AMRFinderPlus organism code (optional)

        Returns:
            DataFrame with AMR results or None if failed
        """
        output_file = self.output_dir / f"{biosample_id}_amrfinder.tsv"

        # Skip if already processed
        if output_file.exists():
            try:
                return pd.read_csv(output_file, sep="\t")
            except Exception:
                pass

        # Decompress if needed
        temp_file = None
        if str(genome_file).endswith(".gz"):
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".fna", delete=False, mode="w"
            )
            with gzip.open(genome_file, "rt") as f_in:
                temp_file.write(f_in.read())
            temp_file.close()
            input_file = temp_file.name
        else:
            input_file = str(genome_file)

        try:
            # Build command
            cmd = [
                "amrfinder",
                "-n", input_file,  # Nucleotide input
                "-o", str(output_file),
                "--plus",  # Include stress/virulence genes
            ]

            # Add organism-specific options if available
            if organism_code:
                cmd.extend(["--organism", organism_code])

            # Run AMRFinderPlus
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0 and output_file.exists():
                df = pd.read_csv(output_file, sep="\t")
                df["biosample_id"] = biosample_id
                return df
            else:
                logger.warning(f"AMRFinder failed for {biosample_id}: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.warning(f"AMRFinder timeout for {biosample_id}")
            return None
        except Exception as e:
            logger.error(f"Error running AMRFinder for {biosample_id}: {e}")
            return None
        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    def run_on_all_genomes(
        self,
        max_samples: Optional[int] = None,
        use_organism: bool = True,
    ) -> pd.DataFrame:
        """Run AMRFinderPlus on all genomes.

        Args:
            max_samples: Maximum number of samples to process (for testing)
            use_organism: Whether to use organism-specific detection

        Returns:
            Combined DataFrame with all AMR results
        """
        if not self.check_installation():
            logger.error(
                "AMRFinderPlus not installed. Install with:\n"
                "  conda install -c bioconda ncbi-amrfinderplus\n"
                "Then update database with:\n"
                "  amrfinder -u"
            )
            raise RuntimeError("AMRFinderPlus not installed")

        self.load_metadata()

        # Get genome files
        genome_files = list(self.genomes_dir.glob("*.fna.gz"))
        if max_samples:
            genome_files = genome_files[:max_samples]

        logger.info(f"Processing {len(genome_files)} genomes...")

        all_results = []
        for i, genome_file in enumerate(genome_files):
            biosample_id = genome_file.stem.replace(".fna", "")

            # Get organism code
            organism_code = None
            if use_organism:
                organism_code = self.get_organism_for_sample(biosample_id)

            # Run AMRFinderPlus
            result = self.run_amrfinder_on_genome(
                genome_file, biosample_id, organism_code
            )

            if result is not None and len(result) > 0:
                all_results.append(result)
                self.amr_results[biosample_id] = result

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(genome_files)} genomes")

        # Combine results
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            combined.to_csv(self.output_dir / "all_amr_results.csv", index=False)
            logger.info(f"Found {len(combined)} AMR genes across {len(all_results)} genomes")
            return combined
        else:
            logger.warning("No AMR genes found in any genome")
            return pd.DataFrame()

    def create_amr_labels(
        self,
        min_samples_per_drug: int = 10,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Create AMR labels from AMRFinderPlus results.

        Converts AMR gene detections into drug resistance labels.

        Args:
            min_samples_per_drug: Minimum samples with resistance to include a drug

        Returns:
            Tuple of (labels DataFrame, drug class mapping)
        """
        # Load all results
        results_file = self.output_dir / "all_amr_results.csv"
        if not results_file.exists():
            raise FileNotFoundError(
                "No AMR results found. Run run_on_all_genomes() first."
            )

        df = pd.read_csv(results_file)
        logger.info(f"Loaded {len(df)} AMR annotations")

        # Filter to AMR genes only (not stress/virulence)
        amr_df = df[df["Element type"] == "AMR"].copy()
        logger.info(f"AMR genes: {len(amr_df)}")

        if len(amr_df) == 0:
            logger.warning("No AMR genes found in results")
            return pd.DataFrame(), {}

        # Get unique drug classes
        # AMRFinderPlus uses "Class" and "Subclass" columns
        drug_classes = set()
        for _, row in amr_df.iterrows():
            drug_class = row.get("Class", "")
            if pd.notna(drug_class) and drug_class:
                drug_classes.add(drug_class)

        logger.info(f"Drug classes found: {drug_classes}")

        # Create label matrix
        biosample_ids = amr_df["biosample_id"].unique()
        labels = []

        for biosample_id in biosample_ids:
            sample_amr = amr_df[amr_df["biosample_id"] == biosample_id]
            sample_drugs = set(sample_amr["Class"].dropna().unique())

            row = {"biosample_id": biosample_id}
            for drug in drug_classes:
                row[drug] = 1 if drug in sample_drugs else 0
            labels.append(row)

        labels_df = pd.DataFrame(labels)

        # Filter drugs with enough samples
        drug_counts = labels_df.drop(columns=["biosample_id"]).sum()
        valid_drugs = drug_counts[drug_counts >= min_samples_per_drug].index.tolist()

        logger.info(f"Drugs with >= {min_samples_per_drug} resistant samples: {len(valid_drugs)}")
        for drug in valid_drugs:
            logger.info(f"  {drug}: {drug_counts[drug]} samples")

        # Create drug class mapping
        drug_mapping = {drug: i for i, drug in enumerate(sorted(valid_drugs))}

        # Save labels
        labels_df.to_csv(self.output_dir / "amr_labels.csv", index=False)
        with open(self.output_dir / "drug_mapping.json", "w") as f:
            json.dump(drug_mapping, f, indent=2)

        return labels_df, drug_mapping

    def get_phenotype_labels(self) -> pd.DataFrame:
        """Get resistance phenotype labels for preprocessing.

        Returns DataFrame with columns:
        - biosample_id
        - One column per drug class (1=resistant, 0=susceptible/unknown)
        """
        labels_file = self.output_dir / "amr_labels.csv"
        if labels_file.exists():
            return pd.read_csv(labels_file)
        else:
            labels_df, _ = self.create_amr_labels()
            return labels_df


def main():
    """Main function to run AMR annotation pipeline."""
    annotator = AMRFinderAnnotator()

    # Check installation
    if not annotator.check_installation():
        print("\n" + "=" * 60)
        print("AMRFinderPlus is not installed!")
        print("=" * 60)
        print("\nTo install AMRFinderPlus:")
        print("  1. Using conda (recommended):")
        print("     conda install -c bioconda ncbi-amrfinderplus")
        print("\n  2. Using docker:")
        print("     docker pull ncbi/amr")
        print("\n  3. Manual installation:")
        print("     https://github.com/ncbi/amr/wiki/Installing-AMRFinder")
        print("\nAfter installation, update the database:")
        print("  amrfinder -u")
        print("=" * 60)
        return

    # Run on all genomes
    print("\nRunning AMRFinderPlus on all genomes...")
    results = annotator.run_on_all_genomes()

    if len(results) > 0:
        print(f"\nFound {len(results)} AMR genes")

        # Create labels
        print("\nCreating AMR labels...")
        labels_df, drug_mapping = annotator.create_amr_labels()

        print(f"\nCreated labels for {len(labels_df)} samples")
        print(f"Drug classes: {list(drug_mapping.keys())}")

        print(f"\nResults saved to: {annotator.output_dir}")
    else:
        print("\nNo AMR genes detected. Check genome files and AMRFinderPlus installation.")


if __name__ == "__main__":
    main()
