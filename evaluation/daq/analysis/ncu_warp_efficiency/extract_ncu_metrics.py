#!/usr/bin/env python3
import argparse
import csv
import sys

METRIC_COLUMNS = [
    "gpu__time_duration.sum",
    "smsp__inst_executed.sum",
    "smsp__thread_inst_executed.sum",
    "smsp__thread_inst_executed_per_inst_executed.ratio",
]


def ncu_csv_rows(path):
    with open(path, newline="") as f:
        lines = [line for line in f if not line.startswith("==")]

    header_idx = None
    for idx, line in enumerate(lines):
        if line.startswith('"ID",') or line.startswith("ID,"):
            header_idx = idx
            break
    if header_idx is None:
        return

    yield from csv.DictReader(lines[header_idx:])


def normalize_value(value):
    return value.replace(",", "").strip()


def main():
    parser = argparse.ArgumentParser(
        description="Extract Nsight Compute raw CSV metric rows into a run summary."
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--queues", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--raw-csv", required=True)
    args = parser.parse_args()

    rows = []
    for row in ncu_csv_rows(args.raw_csv):
        if not row.get("ID"):
            continue

        metric_name = row.get("Metric Name") or row.get("Metric Name ")
        metric_value = row.get("Metric Value") or row.get("Metric Value ")
        if metric_name and metric_value is not None:
            metric_items = [(metric_name.strip(), row.get("Metric Unit", ""), metric_value)]
        else:
            metric_items = [
                (metric, "", row[metric])
                for metric in METRIC_COLUMNS
                if metric in row and row[metric] != ""
            ]

        kernel_name = row.get("Kernel Name", "")
        for name, unit, value in metric_items:
            rows.append(
                [
                    args.benchmark,
                    args.variant,
                    args.queues,
                    args.input,
                    args.cutoff,
                    kernel_name,
                    name,
                    unit.strip(),
                    normalize_value(value),
                    args.raw_csv,
                ]
            )

    writer = csv.writer(sys.stdout)
    writer.writerows(rows)


if __name__ == "__main__":
    main()
