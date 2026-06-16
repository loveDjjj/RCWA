from __future__ import annotations

import argparse
from pathlib import Path

from fdtd import run_fdtd_main


DEFAULT_DEFAULTS = "configs/fdtd/defaults.yaml"
DEFAULT_STRUCTURE = "configs/fdtd/structure.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Run an FDTD validation case from YAML configs.")
    parser.add_argument("--defaults", default=DEFAULT_DEFAULTS, help="Path to the FDTD defaults YAML config.")
    parser.add_argument("--structure", default=DEFAULT_STRUCTURE, help="Path to the FDTD structure YAML config.")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_fdtd_main(Path(args.defaults), Path(args.structure))
    if result.get("config_sources"):
        print(f"Completed FDTD case: defaults={result['config_sources']['defaults']}")
        print(f"Structure config: {result['config_sources']['structure']}")
    else:
        print(f"Completed FDTD case: {result['config']}")
    print(f"Outputs: {result['output_dir']}")


if __name__ == "__main__":
    main()
