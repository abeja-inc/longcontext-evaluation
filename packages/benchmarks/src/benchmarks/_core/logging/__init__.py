from .local_logger import log_results_local


try:
    from .wandb_logger import log_results_wandb
except ImportError:
    log_results_wandb = None
