import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class LendingDataLoader:
    """Load and preprocess lending data from various sources."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.supported_formats = ['.csv', '.xlsx', '.json', '.parquet']
    
    def load_data(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load data from file with automatic format detection.
        
        Args:
            filename: Name of the data file
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            Loaded DataFrame
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            # Try different subdirectories
            for subdir in ['raw', 'processed', 'synthetic']:
                alt_path = self.data_dir / subdir / filename
                if alt_path.exists():
                    filepath = alt_path
                    break
            else:
                raise FileNotFoundError(f"Data file not found: {filename}")
        
        # Determine file format and load
        suffix = filepath.suffix.lower()
        
        try:
            if suffix == '.csv':
                df = pd.read_csv(filepath, **kwargs)
            elif suffix == '.xlsx':
                df = pd.read_excel(filepath, **kwargs)
            elif suffix == '.json':
                df = pd.read_json(filepath, **kwargs)
            elif suffix == '.parquet':
                df = pd.read_parquet(filepath, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
            
            logger.info(f"Loaded {len(df)} records from {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data from {filepath}: {e}")
            raise
    
    def save_data(self, df: pd.DataFrame, filename: str, 
                  subdir: str = 'processed', **kwargs):
        """Save DataFrame to file."""
        output_dir = self.data_dir / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / filename
        suffix = filepath.suffix.lower()
        
        try:
            if suffix == '.csv':
                df.to_csv(filepath, index=False, **kwargs)
            elif suffix == '.xlsx':
                df.to_excel(filepath, index=False, **kwargs)
            elif suffix == '.json':
                df.to_json(filepath, **kwargs)
            elif suffix == '.parquet':
                df.to_parquet(filepath, **kwargs)
            else:
                raise ValueError(f"Unsupported output format: {suffix}")
            
            logger.info(f"Saved {len(df)} records to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save data to {filepath}: {e}")
            raise
    
    def load_sample_data(self, sample_size: int = 1000) -> pd.DataFrame:
        """Load sample data for testing."""
        try:
            # Try to load existing synthetic data
            sample_file = self.data_dir / 'synthetic' / 'sample_data.csv'
            
            if sample_file.exists():
                df = pd.read_csv(sample_file)
                logger.info(f"Loaded sample data: {len(df)} records")
                return df.sample(n=min(sample_size, len(df)), random_state=42)
            else:
                logger.warning("Sample data file not found")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to load sample data: {e}")
            return pd.DataFrame()