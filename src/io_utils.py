# src/io_utils.py

import csv
import json
from pathlib import Path


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_csv(rows, output_path):
    """
    Save list[dict] to CSV.

    Returns:
        output_path as Path
    """

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    if not rows:
        raise ValueError("rows is empty. Nothing to save.")

    # Preserve the key order from the first row, then append any extra keys.
    fieldnames = list(rows[0].keys())

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    return output_path


def save_metadata(metadata, output_path):
    """
    Save metadata dictionary to JSON.
    """

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return output_path