"""Configuration for data collection."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CollectionConfig:
    """Configuration for data collection pipelines."""

    # Base directories
    base_dir: Path = field(default_factory=lambda: Path("data"))

    # NCBI settings
    ncbi_email: str = ""
    ncbi_api_key: Optional[str] = None

    # Target organisms (focus on common pathogens in Bangladesh)
    # Prioritized by clinical relevance and AMR burden
    target_organisms: list[str] = field(
        default_factory=lambda: [
            # ESKAPE pathogens (high AMR, hospital-acquired)
            "Escherichia coli",
            "Klebsiella pneumoniae",
            "Staphylococcus aureus",
            "Acinetobacter baumannii",
            "Pseudomonas aeruginosa",
            "Enterococcus faecium",
            "Enterococcus faecalis",
            # Enteric pathogens (high burden in Bangladesh)
            "Salmonella enterica",
            "Vibrio cholerae",
            "Campylobacter jejuni",
            "Clostridioides difficile",
            # Respiratory pathogens
            "Streptococcus pneumoniae",
            "Mycobacterium tuberculosis",
            "Haemophilus influenzae",
            "Streptococcus pyogenes",
            # Other clinically relevant
            "Neisseria gonorrhoeae",
            "Neisseria meningitidis",
            "Listeria monocytogenes",
            "Serratia marcescens",
            "Citrobacter freundii",
        ]
    )

    # Collection limits
    max_samples_per_organism: int = 500
    max_downloads: int = 200

    # Rate limiting (seconds between requests)
    request_delay: float = 0.4

    @property
    def raw_dir(self) -> Path:
        return self.base_dir / "raw"

    @property
    def external_dir(self) -> Path:
        return self.base_dir / "external"

    @property
    def processed_dir(self) -> Path:
        return self.base_dir / "processed"

    @property
    def ncbi_dir(self) -> Path:
        return self.raw_dir / "ncbi"

    @property
    def patric_dir(self) -> Path:
        return self.raw_dir / "patric"

    @property
    def card_dir(self) -> Path:
        return self.raw_dir / "card-data"

    @property
    def resfinder_dir(self) -> Path:
        return self.external_dir / "resfinder"

    def ensure_directories(self) -> None:
        """Create all required directories."""
        for dir_path in [
            self.raw_dir,
            self.external_dir,
            self.processed_dir,
            self.ncbi_dir,
            self.patric_dir,
            self.card_dir,
            self.resfinder_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "CollectionConfig":
        """Create config from environment variables."""
        import os

        return cls(
            ncbi_email=os.getenv("NCBI_EMAIL", ""),
            ncbi_api_key=os.getenv("NCBI_API_KEY"),
        )
