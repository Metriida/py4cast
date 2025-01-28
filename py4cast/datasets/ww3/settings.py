from pathlib import Path
import yaml

FORMATSTR = "%Y%m%d%H"
SCRATCH_PATH = Path("/scratch/shared/ww3/datas")
DEFAULT_CONFIG = Path(__file__).parents[3] / "config/datasets/ww3.json"

with open(Path(__file__).parents[0] / "metadata.yaml", "r") as file:
    METADATA = yaml.safe_load(file)