#!/usr/bin/env python3
from cot_safety.cli import main

if __name__ == "__main__":
    main(["steer", "validate-scope", "--config", "configs/experiment/stage4_pause_steering.yaml"])
