#!/usr/bin/env python3
"""Convert a real-session mcap bag (scripts/record_real.sh) to CSV.

Columns: the VM receive time (bag timestamp = ROS system clock on the VM = the one shared clock), the RAW firmware
degrees for all 7 axes (index 6 = EE), and the 6 arm joints converted to URDF radians by control.stepper_map.

The mapping table is a PARAMETER: by default the recording's own meta.json table is used (so the CSV is produced with
the table the data was taken under), but --table-version default or a custom meta re-converts the SAME raw degrees with
a different table. This is why the bag keeps RAW degrees: axis 4's gear ratio is wrong and the offsets may change, so
old recordings stay re-convertible.

    source /opt/ros/jazzy/setup.bash && source .../RoverFlake2/install/setup.bash
    python3 real_bag_to_csv.py data/real_sessions/<ts>_<label>/bag            # -> <...>/feedback.csv
    python3 real_bag_to_csv.py <bag> -o out.csv --meta <session>/meta.json --topic /arm/feedback
"""
import argparse
import csv
import json
import os
import sys

# stepper_map is the single source of truth for the mapping (pure stdlib, imports on the system python ROS uses).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from rover_intent.control.stepper_map import DEFAULT_TABLE, StepperTable, fw_to_q  # noqa: E402

import rosbag2_py  # noqa: E402
from rclpy.serialization import deserialize_message  # noqa: E402
from rosidl_runtime_py.utilities import get_message  # noqa: E402


def load_table(meta_path, bag):
    """Table from an explicit --meta, else the session's own meta.json (bag's parent dir), else the built-in default."""
    if meta_path is None:
        guess = os.path.join(os.path.dirname(os.path.abspath(bag.rstrip("/"))), "meta.json")
        meta_path = guess if os.path.exists(guess) else None
    if meta_path and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if "stepper_table" in meta:
            return StepperTable.from_dict(meta["stepper_table"]), meta_path
    return DEFAULT_TABLE, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="bag directory (contains metadata.yaml + *.mcap)")
    ap.add_argument("-o", "--out", default=None, help="output CSV (default: <bag parent>/<topic>.csv)")
    ap.add_argument("--meta", default=None, help="meta.json with the stepper_table to convert with (default: the "
                                                 "session's own; pass a different one to re-convert)")
    ap.add_argument("--topic", default="/arm/feedback", help="topic to export (default /arm/feedback)")
    a = ap.parse_args()

    table, used_meta = load_table(a.meta, a.bag)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.bag.rstrip("/"))),
                                a.topic.strip("/").replace("/", "_") + ".csv")

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=a.bag, storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    typemap = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if a.topic not in typemap:
        sys.exit(f"topic {a.topic} not in bag (have: {sorted(typemap)})")
    msg_cls = get_message(typemap[a.topic])

    n = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_vm"] + [f"fw{i+1}" for i in range(7)] + [f"q{i+1}" for i in range(6)] + ["table_version"])
        while reader.has_next():
            topic, data, t_ns = reader.read_next()
            if topic != a.topic:
                continue
            msg = deserialize_message(data, msg_cls)
            fw = [float(x) for x in msg.positions]
            fw = (fw + [0.0] * 7)[:7]                      # pad/truncate to 7 (index 6 = EE)
            q = fw_to_q(fw, table)
            w.writerow([f"{t_ns / 1e9:.6f}"] + [f"{x:.6f}" for x in fw] + [f"{x:.6f}" for x in q] + [table.version])
            n += 1
    print(f"wrote {n} rows from {a.topic} -> {out} (table {table.version}"
          f"{', from ' + used_meta if used_meta else ', built-in default'})")


if __name__ == "__main__":
    main()
