import logging
import logging.config
import yaml
from pathlib import Path
import os

def setup_logger(name: str, config_path: str = "config/logging_config.yaml") -> logging.Logger:
    """
    Set up logger with configuration from YAML file.
    
    Args:
        name: Logger name
        config_path: Path to logging configuration file
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Load logging configuration
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        logging.config.dictConfig(config)
    else:
        # Fallback to basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler('logs/fair_ai.log'),
                logging.StreamHandler()
            ]
        )
    
    return logging.getLogger(name)