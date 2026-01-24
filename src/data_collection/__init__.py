"""
Data collection module for AMR research.

Provides collectors for various bacterial genome and AMR databases:
- NCBI Pathogen Detection
- CARD (Comprehensive Antibiotic Resistance Database)
- PATRIC (Pathosystems Resource Integration Center)
- ResFinder
"""

from .config import CollectionConfig
from .base import BaseCollector
from .ncbi_collector import NCBICollector
from .card_collector import CARDCollector
from .patric_collector import PATRICCollector
from .resfinder_collector import ResFinderCollector

__all__ = [
    "CollectionConfig",
    "BaseCollector",
    "NCBICollector",
    "CARDCollector",
    "PATRICCollector",
    "ResFinderCollector",
]
