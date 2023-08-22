import os
import json
import logging
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs
# from transformers.utils import logging
from utils.config import args
from utils.process_manager import ProcessManager
import gc
import torch
from datetime import timedelta
import time


kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, log_with="wandb", kwargs_handlers=[kwargs])

# Initialize Logger
accelerator.init_trackers(
    project_name="preference-ranking-optimization",
    config=kwargs,
    init_kwargs={
        "wandb": {
            "name": f"pro_{str(time.time())}",
        },
    }
)

args.gradient_accumulation_steps = accelerator.gradient_accumulation_steps
log_filename = ""
if args.do_train:
    log_filename = "train.log"
else:
    log_filename = "bin.log"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    filename=os.path.join(args.log_path, log_filename)
)

logger = get_logger(__name__)
args_message = '\n'+'\n'.join([f'{k:<40}: {v}' for k, v in vars(args).items()])

logger.info(args_message, main_process_only=True)
accelerator.print(args_message)

process_manager = ProcessManager(
    accelerator, 
)

# Run!
accelerator.wait_for_everyone()

if args.do_train:
    model = process_manager.train()