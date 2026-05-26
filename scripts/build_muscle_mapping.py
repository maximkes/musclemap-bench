#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

CURATED = [
    {"rajagopal": "glut_max1_r", "kinesis": "glut_max1_r"},
    {"rajagopal": "glut_max2_r", "kinesis": "glut_max2_r"},
    {"rajagopal": "glut_max3_r", "kinesis": "glut_max3_r"},
    {"rajagopal": "glut_med1_r", "kinesis": "glut_med1_r"},
    {"rajagopal": "glut_med2_r", "kinesis": "glut_med2_r"},
    {"rajagopal": "glut_med3_r", "kinesis": "glut_med3_r"},
    {"rajagopal": "semimem_r", "kinesis": "semimem_r"},
    {"rajagopal": "semiten_r", "kinesis": "semiten_r"},
    {"rajagopal": "bifemlh_r", "kinesis": "bifemlh_r"},
    {"rajagopal": "bifemsh_r", "kinesis": "bifemsh_r"},
    {"rajagopal": "rect_fem_r", "kinesis": "rect_fem_r"},
    {"rajagopal": "vas_med_r", "kinesis": "vas_med_r"},
    {"rajagopal": "vas_int_r", "kinesis": "vas_int_r"},
    {"rajagopal": "vas_lat_r", "kinesis": "vas_lat_r"},
    {"rajagopal": "med_gas_r", "kinesis": "med_gas_r"},
    {"rajagopal": "lat_gas_r", "kinesis": "lat_gas_r"},
    {"rajagopal": "soleus_r", "kinesis": "soleus_r"},
    {"rajagopal": "tib_post_r", "kinesis": "tib_post_r"},
    {"rajagopal": "tib_ant_r", "kinesis": "tib_ant_r"},
    {"rajagopal": "per_brev_r", "kinesis": "per_brev_r"},
    {"rajagopal": "per_long_r", "kinesis": "per_long_r"},
    {"rajagopal": "glut_max1_l", "kinesis": "glut_max1_l"},
    {"rajagopal": "glut_max2_l", "kinesis": "glut_max2_l"},
    {"rajagopal": "glut_max3_l", "kinesis": "glut_max3_l"},
    {"rajagopal": "glut_med1_l", "kinesis": "glut_med1_l"},
    {"rajagopal": "glut_med2_l", "kinesis": "glut_med2_l"},
    {"rajagopal": "glut_med3_l", "kinesis": "glut_med3_l"},
    {"rajagopal": "semimem_l", "kinesis": "semimem_l"},
    {"rajagopal": "semiten_l", "kinesis": "semiten_l"},
    {"rajagopal": "bifemlh_l", "kinesis": "bifemlh_l"},
    {"rajagopal": "bifemsh_l", "kinesis": "bifemsh_l"},
    {"rajagopal": "rect_fem_l", "kinesis": "rect_fem_l"},
    {"rajagopal": "vas_med_l", "kinesis": "vas_med_l"},
    {"rajagopal": "vas_int_l", "kinesis": "vas_int_l"},
    {"rajagopal": "vas_lat_l", "kinesis": "vas_lat_l"},
    {"rajagopal": "med_gas_l", "kinesis": "med_gas_l"},
    {"rajagopal": "lat_gas_l", "kinesis": "lat_gas_l"},
    {"rajagopal": "soleus_l", "kinesis": "soleus_l"},
    {"rajagopal": "tib_post_l", "kinesis": "tib_post_l"},
    {"rajagopal": "tib_ant_l", "kinesis": "tib_ant_l"},
    {"rajagopal": "per_brev_l", "kinesis": "per_brev_l"},
    {"rajagopal": "per_long_l", "kinesis": "per_long_l"},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_path = Path(cfg["layer1"]["muscle_mapping_json"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "Curated Rajagopal ↔ Kinesis muscle mapping",
        "mapping": CURATED,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
