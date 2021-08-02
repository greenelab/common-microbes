"""Path definitions"""
from pathlib import Path

# Path to this repository
PROJECT_DIR = Path(__file__).parents[1]

# Path to local directory where data files will be stored
LOCAL_DIR = Path.home()
LOCAL_DATA_DIR = LOCAL_DIR / "Documents" / "Data" / "Common_microbes"

# Location of raw microbiome data
RAW_MICROBIOME_DATA = LOCAL_DATA_DIR / "studies_consolidated.txt"
