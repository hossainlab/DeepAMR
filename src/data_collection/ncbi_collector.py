"""NCBI Pathogen Detection data collector."""

import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from Bio import Entrez
from tqdm import tqdm

from .base import BaseCollector
from .config import CollectionConfig


class NCBICollector(BaseCollector):
    """Collect genome and AMR data from NCBI Pathogen Detection."""

    def __init__(self, config: CollectionConfig):
        super().__init__(config)
        if not config.ncbi_email:
            raise ValueError("NCBI email is required. Set ncbi_email in config.")

        Entrez.email = config.ncbi_email
        if config.ncbi_api_key:
            Entrez.api_key = config.ncbi_api_key

        self._genomes_dir = self.output_dir / "genomes"
        self._metadata_dir = self.output_dir / "metadata"

    @property
    def name(self) -> str:
        return "NCBI Pathogen Detection"

    @property
    def output_dir(self) -> Path:
        return self.config.ncbi_dir

    def collect(self) -> pd.DataFrame:
        """Run full NCBI collection pipeline."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._genomes_dir.mkdir(exist_ok=True)
        self._metadata_dir.mkdir(exist_ok=True)

        all_data = []

        for organism in self.config.target_organisms:
            self.logger.info(f"Processing: {organism}")

            biosample_ids = self._search_biosamples(organism)
            if not biosample_ids:
                continue

            metadata = self._fetch_metadata(biosample_ids)
            assemblies = self._find_assemblies(biosample_ids)

            if not metadata.empty:
                combined = metadata.merge(assemblies, on="biosample_id", how="left")
                combined["organism_query"] = organism
                all_data.append(combined)

                # Save intermediate results
                safe_name = organism.replace(" ", "_")
                combined.to_csv(
                    self._metadata_dir / f"{safe_name}_metadata.csv", index=False
                )

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result.to_csv(self.output_dir / "complete_metadata.csv", index=False)

        # Download genomes for samples with assemblies
        samples_with_assemblies = result[result["assembly_accession"].notna()]
        if not samples_with_assemblies.empty:
            self._download_genomes(samples_with_assemblies)

        self.log_summary(result)
        return result

    def _search_biosamples(self, organism: str) -> list[str]:
        """Search NCBI for BioSample IDs."""
        # Simple organism search - filter results later
        query = f'"{organism}"[Organism]'
        self.logger.info(f"Search query: {query}")

        try:
            handle = Entrez.esearch(
                db="biosample",
                term=query,
                retmax=self.config.max_samples_per_organism,
                usehistory="y",
            )
            results = Entrez.read(handle)
            handle.close()

            ids = results.get("IdList", [])
            self.logger.info(f"Found {len(ids)} BioSample records")
            time.sleep(self.config.request_delay)
            return ids

        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return []

    def _fetch_metadata(self, biosample_ids: list[str]) -> pd.DataFrame:
        """Fetch metadata for BioSample IDs."""
        self.logger.info(f"Fetching metadata for {len(biosample_ids)} samples")
        records = []
        batch_size = 100

        for i in tqdm(range(0, len(biosample_ids), batch_size), desc="Fetching"):
            batch = biosample_ids[i : i + batch_size]

            try:
                handle = Entrez.efetch(db="biosample", id=batch, rettype="xml")
                xml_data = handle.read()
                handle.close()

                # Parse XML using ElementTree instead of Entrez.read()
                root = ET.fromstring(xml_data)
                for biosample in root.findall(".//BioSample"):
                    parsed = self._parse_biosample_xml(biosample)
                    if parsed:
                        records.append(parsed)

                time.sleep(self.config.request_delay)

            except Exception as e:
                self.logger.error(f"Fetch error for batch {i}: {e}")

        return pd.DataFrame(records)

    def _parse_biosample_xml(self, biosample: ET.Element) -> dict | None:
        """Parse a BioSample XML element using ElementTree."""
        try:
            # Get numeric biosample ID from the 'id' attribute (matches search results)
            biosample_id = biosample.get("id", "")

            # Get accession from attribute (e.g., SAMN12345678)
            accession = biosample.get("accession", "")

            # Parse attributes
            attrs = {}
            for attr in biosample.findall(".//Attribute"):
                name = attr.get("attribute_name", "") or attr.get("harmonized_name", "")
                value = attr.text or ""
                if name:
                    attrs[name] = value

            # Parse links
            links = {}
            for link in biosample.findall(".//Link"):
                label = link.get("label", "")
                target = link.get("target", "")
                if label and target:
                    links[label] = target

            return {
                "biosample_id": biosample_id,
                "accession": accession,
                "organism": attrs.get("organism", ""),
                "strain": attrs.get("strain", ""),
                "isolation_source": attrs.get("isolation_source", ""),
                "host": attrs.get("host", ""),
                "collection_date": attrs.get("collection_date", ""),
                "geo_loc_name": attrs.get("geo_loc_name", ""),
                "amr_genotypes": attrs.get("AMR_genotypes", ""),
                "amr_phenotypes": attrs.get("AMR_phenotypes", ""),
                "disease": attrs.get("disease", ""),
                "sra_accession": links.get("SRA", ""),
            }

        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            return None

    def _find_assemblies(self, biosample_ids: list[str]) -> pd.DataFrame:
        """Find genome assemblies for BioSample IDs."""
        self.logger.info(f"Finding assemblies for {len(biosample_ids)} samples")
        assemblies = []

        for biosample_id in tqdm(biosample_ids, desc="Finding assemblies"):
            try:
                handle = Entrez.esearch(
                    db="assembly", term=f"{biosample_id}[BioSample]"
                )
                results = Entrez.read(handle)
                handle.close()

                assembly_ids = results.get("IdList", [])
                if not assembly_ids:
                    continue

                handle = Entrez.esummary(db="assembly", id=assembly_ids[0])
                summary = Entrez.read(handle)
                handle.close()

                info = summary["DocumentSummarySet"]["DocumentSummary"][0]
                assemblies.append(
                    {
                        "biosample_id": biosample_id,
                        "assembly_accession": info.get("AssemblyAccession", ""),
                        "assembly_status": info.get("AssemblyStatus", ""),
                        "ftp_path_refseq": info.get("FtpPath_RefSeq", ""),
                        "ftp_path_genbank": info.get("FtpPath_GenBank", ""),
                    }
                )

                time.sleep(self.config.request_delay)

            except Exception as e:
                self.logger.error(f"Assembly error for {biosample_id}: {e}")

        # Ensure DataFrame has proper columns even if empty
        if not assemblies:
            return pd.DataFrame(columns=[
                "biosample_id", "assembly_accession", "assembly_status",
                "ftp_path_refseq", "ftp_path_genbank"
            ])
        return pd.DataFrame(assemblies)

    def _download_genomes(self, df: pd.DataFrame) -> None:
        """Download genome FASTA files."""
        self.logger.info(f"Downloading up to {self.config.max_downloads} genomes")
        downloaded = 0

        for _, row in tqdm(df.iterrows(), total=min(len(df), self.config.max_downloads)):
            if downloaded >= self.config.max_downloads:
                break

            ftp_path = row["ftp_path_refseq"] or row["ftp_path_genbank"]
            if not ftp_path:
                continue

            try:
                assembly_name = ftp_path.split("/")[-1]
                fasta_url = f"{ftp_path}/{assembly_name}_genomic.fna.gz"
                fasta_url = fasta_url.replace("ftp://", "https://")

                output_file = self._genomes_dir / f"{row['biosample_id']}.fna.gz"
                if output_file.exists():
                    downloaded += 1
                    continue

                response = requests.get(fasta_url, stream=True, timeout=300)
                response.raise_for_status()

                with open(output_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                downloaded += 1
                time.sleep(1)

            except Exception as e:
                self.logger.error(f"Download error for {row['biosample_id']}: {e}")

        self.logger.info(f"Downloaded {downloaded} genomes")
