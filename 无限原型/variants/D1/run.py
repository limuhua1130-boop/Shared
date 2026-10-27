"""Compatibility entry; use Share/share.py for all documented commands."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
from share import main
if __name__=="__main__":
    sys.argv[1:1]=["run","--variant",Path(__file__).resolve().parent.name]
    main()
