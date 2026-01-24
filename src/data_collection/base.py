"""Base class for data collectors."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from .config import CollectionConfig


class BaseCollector(ABC):
    """Abstract base class for all data collectors."""

    def __init__(self, config: CollectionConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the data source."""
        pass

    @property
    @abstractmethod
    def output_dir(self) -> Path:
        """Return the output directory for this collector."""
        pass

    @abstractmethod
    def collect(self) -> pd.DataFrame:
        """Run the collection pipeline and return metadata DataFrame."""
        pass

    def setup_logging(self, log_file: Path | None = None) -> None:
        """Configure logging for this collector."""
        handlers = [logging.StreamHandler()]
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file))

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=handlers,
        )

    def log_summary(self, df: pd.DataFrame) -> None:
        """Log a summary of collected data."""
        self.logger.info(f"Collection complete for {self.name}")
        self.logger.info(f"Total records: {len(df)}")
        if not df.empty:
            self.logger.info(f"Columns: {', '.join(df.columns[:10])}")
