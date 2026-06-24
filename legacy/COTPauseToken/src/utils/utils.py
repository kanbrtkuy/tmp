import warnings
from typing import Any, Callable, Dict, Optional, Tuple

from omegaconf import DictConfig

from src.utils.pylogger import RankedLogger
from src.utils.rich_utils import print_config_tree

log = RankedLogger(__name__)


def extras(cfg: DictConfig) -> None:
    if not cfg.get("extras"):
        return
    if cfg.extras.get("ignore_warnings"):
        warnings.filterwarnings("ignore")
    if cfg.extras.get("print_config"):
        print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        try:
            return task_func(cfg=cfg)
        finally:
            if cfg.get("paths"):
                log.info(f"Output dir: {cfg.paths.output_dir}")

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    if not metric_name:
        return None
    if metric_name not in metric_dict:
        raise KeyError(f"Metric {metric_name!r} was not found in trainer metrics")
    metric = metric_dict[metric_name]
    return metric.item() if hasattr(metric, "item") else metric


def make_trainable_params_summary(model: Any) -> str:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen = total - trainable
    pct = 100.0 * trainable / total if total else 0.0
    return (
        f"total={total:,}, trainable={trainable:,} ({pct:.2f}%), "
        f"frozen={frozen:,}"
    )
