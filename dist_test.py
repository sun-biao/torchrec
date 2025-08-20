import os
import torch
import torch.distributed as dist

# 初始化分布式环境
def setup(rank, world_size):
    # BACKEND: nccl 是 NVIDIA GPU 推荐的后端
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank) 
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % torch.cuda.device_count()) # 每个进程使用一个GPU
    

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_distributed_operation():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Rank {rank}/{world_size} initialized on device {torch.cuda.current_device()}")

    # 创建一个张量并发送给 rank 0
    data = torch.tensor([rank], dtype=torch.float32).to(torch.cuda.current_device())
    print(f"Rank {rank}: Initial data = {data.item()}")

    if rank == 0:
        # rank 0 接收来自所有其他 rank 的数据
        gather_list = [torch.empty_like(data).to(torch.cuda.current_device()) for _ in range(world_size)]
        dist.gather(data, gather_list=gather_list, dst=0)
        print(f"Rank {rank}: Received data from all ranks: {[x.item() for x in gather_list]}")

    else:
        # 其他 rank 发送数据给 rank 0
        dist.gather(data, dst=0)
        print(f"Rank {rank}: Sending data = {data.item()}")

    print(f"Rank {rank}: Operation completed.")

if __name__ == "__main__":
    # 在 TorchX 运行时，RANK 和 WORLD_SIZE 环境变量由 TorchX 自动设置。
    # 这里为了本地测试或手动运行，我们假设这两个变量已设置。
    # 在 GKE 上，TorchX会负责传递这些。
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 验证是否有可用 GPU
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires GPUs.")
        exit(1)

    setup(rank, world_size)
    #print(f"Rank {rank} (local_rank: {os.environ['CUDA_VISIBLE_DEVICES']}) is running on device index {torch.cuda.current_device()}")
    run_distributed_operation()
    cleanup()
    print(f"Rank {rank} exited.")
