"""PATRIC (Pathosystems Resource Integration Center) data collector."""

import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from .base import BaseCollector
from .config import CollectionConfig


class PATRICCollector(BaseCollector):
    """Collect genome and AMR data from PATRIC/BV-BRC."""

    # BV-BRC Data API endpoint
    API_BASE = "https://www.bv-brc.org/api"

    def __init__(self, config: CollectionConfig):
        super().__init__(config)
        self._genomes_dir = self.output_dir / "genomes"

    @property
    def name(self) -> str:
        return "PATRIC/BV-BRC"

    @property
    def output_dir(self) -> Path:
        return self.config.patric_dir

    def collect(self) -> pd.DataFrame:
        """Run full PATRIC collection pipeline."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._genomes_dir.mkdir(exist_ok=True)

        all_genomes = []
        all_amr = []

        for organism in self.config.target_organisms:
            self.logger.info(f"Processing: {organism}")

            genomes = self._search_genomes(organism)
            if genomes.empty:
                continue

            genome_ids = genomes["genome_id"].tolist()
            amr_data = self._fetch_amr_phenotypes(genome_ids[:100])

            all_genomes.append(genomes)
            if not amr_data.empty:
                all_amr.append(amr_data)

        if not all_genomes:
            return pd.DataFrame()

        genomes_df = pd.concat(all_genomes, ignore_index=True)
        genomes_df.to_csv(self.output_dir / "genomes_metadata.csv", index=False)

        if all_amr:
            amr_df = pd.concat(all_amr, ignore_index=True)
            amr_df.to_csv(self.output_dir / "amr_phenotypes.csv", index=False)

            # Merge AMR data with genomes
            result = genomes_df.merge(
                amr_df.groupby("genome_id").first().reset_index(),
                on="genome_id",
                how="left",
            )
        else:
            result = genomes_df

        result.to_csv(self.output_dir / "complete_dataset.csv", index=False)
        self.log_summary(result)
        return result

    def _search_genomes(self, organism: str) -> pd.DataFrame:
        """Search BV-BRC for genomes."""
        self.logger.info(f"Searching BV-BRC for: {organism}")

        # BV-BRC API uses keyword() and limit() in URL
        # Encode the organism name for URL
        from urllib.parse import quote

        encoded_organism = quote(organism)
        url = f"{self.API_BASE}/genome/?keyword({encoded_organism})&limit({self.config.max_samples_per_organism})"

        headers = {"Accept": "application/json"}

        try:
            response = requests.get(url, headers=headers, timeout=120)
            response.raise_for_status()

            data = response.json()
            df = pd.DataFrame(data)
            self.logger.info(f"Found {len(df)} genomes")
            return df

        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return pd.DataFrame()

    def _fetch_amr_phenotypes(self, genome_ids: list[str]) -> pd.DataFrame:
        """Fetch AMR phenotype data for genomes."""
        if not genome_ids:
            return pd.DataFrame()

        self.logger.info(f"Fetching AMR data for {len(genome_ids)} genomes")
        all_amr = []

        for genome_id in tqdm(genome_ids, desc="Fetching AMR"):
            try:
                url = f"{self.API_BASE}/genome_amr/"
                headers = {"Accept": "application/json"}

                # Simple query by genome_id
                search_url = f"{url}?eq(genome_id,{genome_id})&limit(1000)"
                response = requests.get(search_url, headers=headers, timeout=30)
                response.raise_for_status()

                data = response.json()
                all_amr.extend(data)
                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"AMR fetch error for {genome_id}: {e}")

        df = pd.DataFrame(all_amr)
        self.logger.info(f"Retrieved {len(df)} AMR records")
        return df

    def download_genome(self, genome_id: str) -> Path | None:
        """Download a single genome FASTA file using BV-BRC API."""
        output_file = self._genomes_dir / f"{genome_id}.fasta"
        if output_file.exists() and output_file.stat().st_size > 100:
            return output_file

        try:
            # Get genome sequences (contigs) from genome_sequence endpoint
            url = f"{self.API_BASE}/genome_sequence/"
            query = f"?eq(genome_id,{genome_id})&select(sequence_id,sequence,accession,length)"
            headers = {"Accept": "application/json"}

            response = requests.get(url + query, headers=headers, timeout=180)
            response.raise_for_status()

            data = response.json()
            if not data:
                self.logger.warning(f"No contigs found for {genome_id}")
                return None

            # Build FASTA format
            fasta_content = []
            total_length = 0
            for contig in data:
                seq = contig.get("sequence", "")
                if seq:
                    seq_id = contig.get("sequence_id", contig.get("accession", "unknown"))
                    length = contig.get("length", len(seq))
                    fasta_content.append(f">{seq_id} length={length}")
                    # Format sequence in 80-char lines
                    for i in range(0, len(seq), 80):
                        fasta_content.append(seq[i:i+80])
                    total_length += len(seq)

            if not fasta_content:
                self.logger.warning(f"No sequence data for {genome_id}")
                return None

            with open(output_file, "w") as f:
                f.write("\n".join(fasta_content))

            self.logger.info(f"Downloaded {genome_id}: {len(data)} contigs, {total_length:,} bp")
            return output_file

        except Exception as e:
            self.logger.error(f"Download error for {genome_id}: {e}")
            return None

    def download_genomes_with_amr(self, max_genomes: int | None = None) -> list[Path]:
        """Download genome sequences for all genomes with AMR data.

        Args:
            max_genomes: Maximum number of genomes to download. If None, downloads all.

        Returns:
            List of paths to downloaded FASTA files.
        """
        self._genomes_dir.mkdir(parents=True, exist_ok=True)

        # Load AMR phenotypes to get genome IDs
        amr_file = self.output_dir / "amr_phenotypes.csv"
        if not amr_file.exists():
            self.logger.error("AMR phenotypes file not found. Run collect() first.")
            return []

        amr_df = pd.read_csv(amr_file)

        # Get unique genome IDs that have resistance phenotype data
        genome_ids = amr_df[amr_df["resistant_phenotype"].notna()]["genome_id"].unique()
        self.logger.info(f"Found {len(genome_ids)} genomes with AMR phenotype data")

        if max_genomes:
            genome_ids = genome_ids[:max_genomes]
            self.logger.info(f"Limiting to {max_genomes} genomes")

        downloaded = []
        failed = []

        for genome_id in tqdm(genome_ids, desc="Downloading genomes"):
            result = self.download_genome(str(genome_id))
            if result:
                downloaded.append(result)
            else:
                failed.append(genome_id)
            time.sleep(self.config.request_delay)

        self.logger.info(f"Downloaded {len(downloaded)} genomes, {len(failed)} failed")

        # Save download summary
        summary = {
            "total_with_amr": len(genome_ids),
            "downloaded": len(downloaded),
            "failed": len(failed),
            "failed_ids": failed[:100]  # Save first 100 failed IDs
        }

        import json
        with open(self.output_dir / "download_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return downloaded
