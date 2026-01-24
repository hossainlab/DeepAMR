"""ResFinder database collector."""

import subprocess
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from .base import BaseCollector
from .config import CollectionConfig


class ResFinderCollector(BaseCollector):
    """Download and process ResFinder database."""

    REPO_URL = "https://bitbucket.org/genomicepidemiology/resfinder_db.git"

    def __init__(self, config: CollectionConfig):
        super().__init__(config)

    @property
    def name(self) -> str:
        return "ResFinder"

    @property
    def output_dir(self) -> Path:
        return self.config.resfinder_dir

    def collect(self) -> pd.DataFrame:
        """Clone/update ResFinder repository and return gene index."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if (self.output_dir / ".git").exists():
            self._update_repo()
        else:
            self._clone_repo()

        # Parse and return gene information
        genes_df = self._parse_gene_files()
        if not genes_df.empty:
            genes_df.to_csv(self.output_dir / "gene_index.csv", index=False)
            self.log_summary(genes_df)

        return genes_df

    def _clone_repo(self) -> None:
        """Clone ResFinder repository."""
        self.logger.info("Cloning ResFinder database...")
        try:
            subprocess.run(
                ["git", "clone", self.REPO_URL, str(self.output_dir)],
                check=True,
                capture_output=True,
            )
            self.logger.info("Clone complete")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Clone failed: {e.stderr.decode()}")

    def _update_repo(self) -> None:
        """Update existing ResFinder repository."""
        self.logger.info("Updating ResFinder database...")
        try:
            subprocess.run(
                ["git", "pull"],
                cwd=self.output_dir,
                check=True,
                capture_output=True,
            )
            self.logger.info("Update complete")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Update failed: {e.stderr.decode()}")

    def _parse_gene_files(self) -> pd.DataFrame:
        """Parse FASTA files to extract gene information."""
        genes = []

        for fasta_file in self.output_dir.glob("*.fsa"):
            drug_class = fasta_file.stem
            self.logger.info(f"Parsing: {drug_class}")

            try:
                for record in SeqIO.parse(fasta_file, "fasta"):
                    # Parse header: gene_accession_description
                    parts = record.description.split("_", 2)
                    gene_name = parts[0] if parts else record.id

                    genes.append(
                        {
                            "gene_id": record.id,
                            "gene_name": gene_name,
                            "drug_class": drug_class,
                            "description": record.description,
                            "sequence_length": len(record.seq),
                        }
                    )

            except Exception as e:
                self.logger.error(f"Parse error for {fasta_file}: {e}")

        df = pd.DataFrame(genes)
        self.logger.info(f"Parsed {len(df)} resistance genes")
        return df

    def get_genes_by_drug_class(self, drug_class: str) -> pd.DataFrame:
        """Get resistance genes for a specific drug class."""
        index_file = self.output_dir / "gene_index.csv"
        if not index_file.exists():
            return pd.DataFrame()

        df = pd.read_csv(index_file)
        return df[df["drug_class"] == drug_class]

    def get_gene_sequence(self, gene_id: str) -> str | None:
        """Get sequence for a specific gene."""
        for fasta_file in self.output_dir.glob("*.fsa"):
            for record in SeqIO.parse(fasta_file, "fasta"):
                if record.id == gene_id:
                    return str(record.seq)
        return None
