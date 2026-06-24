#!/usr/bin/env python3
from cot_safety.cli import main

if __name__ == "__main__":
    main(["config", "show", "--config", "configs/experiment/full_four_stage.yaml"])
