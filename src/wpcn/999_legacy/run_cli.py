import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(str(ROOT))
from wpcn.cli.main import main
main()
