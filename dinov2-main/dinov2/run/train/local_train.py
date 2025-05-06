import os
import sys
import logging
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dinov2.train import get_args_parser as get_train_args_parser

logger = logging.getLogger("dinov2")

def setup(rank, world_size):
    # 设置所有必要的分布式环境变量
    os.environ.update({
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "29500",
        "RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "LOCAL_RANK": str(rank),
        "LOCAL_WORLD_SIZE": str(world_size)
    })
    
    # dist.init_process_group(
    #     backend="nccl",
    #     init_method="env://",
    #     rank=rank,
    #     world_size=world_size
    # )
    torch.cuda.set_device(rank)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

class Trainer:
    def __init__(self, args):
        self.args = args

    def __call__(self, rank, world_size):
        try:
            setup(rank, world_size)
            self._setup_logging(rank)
            self._setup_args(rank, world_size)
            self.train()
        except Exception as e:
            logger.error(f"Rank {rank} failed: {e}", exc_info=True)
        finally:
            cleanup()

    def _setup_logging(self, rank):
        """动态配置每个进程的日志"""
        handlers = []
        if rank == 0:
            # 主进程记录到文件和终端
            handlers.append(logging.FileHandler(os.path.join(self.args.output_dir, "main.log")))
            handlers.append(logging.StreamHandler())
        else:
            # 其他进程只记录警告及以上级别到单独文件
            handlers.append(logging.FileHandler(os.path.join(self.args.output_dir, f"worker_{rank}.log")))
        
        logging.basicConfig(
            level=logging.INFO if rank == 0 else logging.WARNING,
            format=f"[%(asctime)s] [Rank {rank}] %(levelname)s: %(message)s",
            handlers=handlers
        )

    def _setup_args(self, rank, world_size):
        # 注入分布式参数
        self.args.rank = rank
        self.args.world_size = world_size
        self.args.local_rank = rank
        self.args.device = f"cuda:{rank}"
        logger.info(f"Initialized process {rank}/{world_size}")

    def train(self):
        from dinov2.train import main as train_main
        train_main(self.args)

def main():
    parser = argparse.ArgumentParser(
        description="Local DINOv2 Training",
        parents=[get_train_args_parser(add_help=False)]
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=torch.cuda.device_count(),
        help="Number of GPUs per node"
    )
    args = parser.parse_args()

    # 预创建输出目录（避免多进程竞争）
    os.makedirs(args.output_dir, exist_ok=True)

    # 父进程日志配置
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[logging.StreamHandler()]
    )

    mp.spawn(
        Trainer(args).__call__,
        args=(args.nproc_per_node,),
        nprocs=args.nproc_per_node,
        join=True
    )

if __name__ == "__main__":
    sys.exit(main())