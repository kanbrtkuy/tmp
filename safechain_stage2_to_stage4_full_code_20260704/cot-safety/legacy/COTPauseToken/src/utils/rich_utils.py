from pathlib import Path
from typing import Sequence

import rich
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf


def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "rl_algorithm",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    tree = rich.tree.Tree("CONFIG", style="dim", guide_style="dim")

    queue = [field for field in print_order if field in cfg]
    queue.extend(field for field in cfg if field not in queue)

    for field in queue:
        branch = tree.add(field, style="dim", guide_style="dim")
        value = cfg[field]
        content = OmegaConf.to_yaml(value, resolve=resolve) if isinstance(value, DictConfig) else str(value)
        branch.add(rich.syntax.Syntax(content, "yaml"))

    rich.print(tree)

    if save_to_file and cfg.get("paths"):
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w", encoding="utf-8") as f:
            rich.print(tree, file=f)
