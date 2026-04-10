import sys
import pandas as pd
from pathlib import Path

OUTPUTS_DIR = Path(__file__).parent / "outputs"


def csv_to_latex(table_name: str) -> str:
    path = OUTPUTS_DIR / table_name
    if not path.suffix:
        path = path.with_suffix(".csv")
    df = pd.read_csv(path)
    return df.to_latex(index=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python to_latex_table.py <table_name.csv>", file=sys.stderr)
        sys.exit(1)
    latex = csv_to_latex(sys.argv[1])
    import subprocess
    subprocess.run("pbcopy", input=latex.encode(), check=True)
    print("Copied to clipboard.")
