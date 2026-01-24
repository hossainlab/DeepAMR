"""CARD (Comprehensive Antibiotic Resistance Database) collector."""

import tarfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from .base import BaseCollector
from .config import CollectionConfig


class CARDCollector(BaseCollector):
    """Download and process CARD database."""

    BASE_URL = "https://card.mcmaster.ca/latest"

    FILES = {
        "card_data.tar.bz2": "/data",
        "aro_index.tsv": "/aro_index.tsv",
        "aro_categories.tsv": "/aro_categories.tsv",
        "aro_categories_index.tsv": "/aro_categories_index.tsv",
        "protein_fasta_protein_homolog_model.fasta": "/protein_fasta_protein_homolog_model.fasta",
        "nucleotide_fasta_protein_homolog_model.fasta": "/nucleotide_fasta_protein_homolog_model.fasta",
    }

    def __init__(self, config: CollectionConfig):
        super().__init__(config)

    @property
    def name(self) -> str:
        return "CARD"

    @property
    def output_dir(self) -> Path:
        return self.config.card_dir

    def collect(self) -> pd.DataFrame:
        """Download CARD database and return ARO index."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for filename, endpoint in self.FILES.items():
            output_file = self.output_dir / filename
            if output_file.exists():
                self.logger.info(f"Already exists: {filename}")
                continue

            url = f"{self.BASE_URL}{endpoint}"
            self._download_file(url, output_file)

        # Extract tar archive
        tar_file = self.output_dir / "card_data.tar.bz2"
        if tar_file.exists():
            self._extract_archive(tar_file)

        # Load and return ARO index
        aro_df = self.load_aro_index()
        if aro_df is not None:
            self.log_summary(aro_df)
        return aro_df if aro_df is not None else pd.DataFrame()

    def _download_file(self, url: str, output_path: Path) -> None:
        """Download a file with progress bar."""
        self.logger.info(f"Downloading: {output_path.name}")

        try:
            response = requests.get(url, stream=True, timeout=600)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(output_path, "wb") as f:
                with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

            self.logger.info(f"Downloaded: {output_path.name}")

        except Exception as e:
            self.logger.error(f"Download failed for {output_path.name}: {e}")

    def _extract_archive(self, tar_path: Path) -> None:
        """Extract tar.bz2 archive."""
        extracted_marker = self.output_dir / ".extracted"
        if extracted_marker.exists():
            self.logger.info("Archive already extracted")
            return

        self.logger.info("Extracting CARD archive...")
        try:
            with tarfile.open(tar_path, "r:bz2") as tar:
                tar.extractall(self.output_dir)
            extracted_marker.touch()
            self.logger.info("Extraction complete")
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")

    def load_aro_index(self) -> pd.DataFrame | None:
        """Load ARO (Antibiotic Resistance Ontology) index."""
        aro_file = self.output_dir / "aro_index.tsv"
        if not aro_file.exists():
            self.logger.error("ARO index not found")
            return None

        df = pd.read_csv(aro_file, sep="\t")
        self.logger.info(f"Loaded {len(df)} ARO entries")
        return df

    def load_aro_categories(self) -> pd.DataFrame | None:
        """Load ARO categories."""
        cat_file = self.output_dir / "aro_categories.tsv"
        if not cat_file.exists():
            return None
        return pd.read_csv(cat_file, sep="\t")

    def get_resistance_genes(self) -> pd.DataFrame:
        """Get list of resistance genes from CARD."""
        aro_df = self.load_aro_index()
        if aro_df is None:
            return pd.DataFrame()

        # Filter for protein homolog models (acquired resistance genes)
        if "Model Type" in aro_df.columns:
            genes = aro_df[aro_df["Model Type"] == "protein homolog model"]
            return genes
        return aro_df
