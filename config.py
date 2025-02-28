from pathlib import Path

# path data
BASE_DIR = Path("/home/alism/projects/dr/data")
NUM_IMG_CSV = BASE_DIR / "num_img.csv"
CSV_FOLDER_PATH = BASE_DIR / "json_files/coverage_files"
OUTPUT_CSV = BASE_DIR / "balanced_data/balancedata_new_7900.csv"

# num of imag for balanced
TARGET_IMAGES = 12000
