#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
from typing import Dict, Iterator, List
import time
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torchrec import EmbeddingBagCollection, KeyedJaggedTensor
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.models.dlrm import DLRM
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import datetime
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.types import ModuleSharder, ShardingType
from typing import Type, List  
import torch.nn as nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel 
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torchrec.distributed.planner.types import ParameterConstraints
from typing import cast
from torch.profiler import profile, record_function, ProfilerActivity
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from torchrec.distributed.planner.storage_reservations import FixedPercentageStorageReservation
from datetime import datetime
import pytz
from torchrec.distributed.types import (
    BoundsCheckMode,
    CacheAlgorithm,
    CacheParams,
    DataType,
    ModuleSharder,
    QuantizedCommCodecs,
    ShardingEnv,
    ShardingPlan,
    ShardingType,
)

IS_WEIGHTED = False
device = None

def get_beijing_time():
  """获取当前北京时间，包含年月日时分秒。"""
  beijing_timezone = pytz.timezone('Asia/Shanghai')  # 北京时间使用上海时区
  beijing_time = datetime.now(beijing_timezone)
  return beijing_time.strftime("%Y-%m-%d %H:%M:%S")

# 打印当前北京时间
print(get_beijing_time())


if sys.platform not in ["linux", "linux2"]:
    raise EnvironmentError(
        f"Torchrec does not currently support {sys.platform}. Only linux is supported."
    )
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

#rank = int(os.environ["RANK"])

class DLRMSharder(ModuleSharder):
    # def sharding_types(self, compute_device_type: str):
    #     # 支持 ROW_WISE 分片
    #     return [ShardingType.ROW_WISE]

    def shardable_modules(self) -> List[Type[nn.Module]]:
        # 指定可以分片的模块类型
        return [DLRM]

    def shard(self, module: DLRM, *args, **kwargs):
        return module

    @property
    def module_type(self):
        return DLRM
    
    # def parameter_constraints(self) -> Dict[str, ParameterConstraints]:
    #     # 添加参数约束以启用 UVM caching
    #     return {
    #         "embedding_bag_collection.embedding_bags.table0.weight": ParameterConstraints(
    #             compute_kernels=[EmbeddingComputeKernel.FUSED_UVM_CACHING.value],
    #         ),
    #     }    


class RandomDataset(Dataset):
    def __init__(
        self, 
        sample_size: int, 
        num_dense: int, 
        num_sparse: int, 
        num_embeddings: int, 
        num_ids_per_feature: int = 1,
        num_cpu_workers: int = 1,
        batch_size: int = 512,
        embedding_scope = 0,
    ) -> None:
        self.sample_size = sample_size
        self.num_dense = num_dense
        self.num_sparse = num_sparse
        self.sparse_keys = [f"feature{id}" for id in range(self.num_sparse)]
        self.num_embeddings = num_embeddings
        self.num_ids_per_feature = num_ids_per_feature
        self.num_ids_total = self.num_sparse * self.num_ids_per_feature
        self.batch_size = batch_size
        self.embedding_scope = num_embeddings if embedding_scope==0 else embedding_scope
        # 单线程预生成所有数据（原始数值格式）
        self.data = []
        torch.manual_seed(77)  # 保持随机性
        # for idx in range(sample_size):
        for idx in range(batch_size): #重复使用一个batch_size的数据，节省内存
            
            # 生成稀疏ID
            sparse_ids = torch.randint(
                low=0,
                high=self.num_embeddings,
                size=(self.num_ids_total,)
            )
            if torch.all(sparse_ids == 0):
                sparse_ids[0] = 1
            
            # 存储原始数值（避免复杂对象序列化）
            self.data.append({
                "float": torch.randn(num_dense),
                "sparse_ids": sparse_ids,
                "offsets": torch.arange(
                    0, 
                    self.num_ids_total + 1, 
                    self.num_ids_per_feature, 
                    dtype=torch.int32
                ),
                "label": torch.randint(low=0, high=2, size=(1,))
            })

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        #torch.manual_seed(idx)  # 保持随机性
        item = self.data[idx%self.batch_size] #反复使用一个batch 数据
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=self.sparse_keys,
            values=item["sparse_ids"],
            offsets=item["offsets"],
        )
        return (item["float"], sparse_features, item["label"])

def custom_collate_fn(batch):

    # 提取密集特征和标签
    dense_features = torch.stack([item[0] for item in batch])
    labels = torch.cat([item[2] for item in batch])

    # 提取所有稀疏特征
    sparse_features_list = [item[1] for item in batch]
    
    # 预加载所有 values 和 offsets
    all_values_list = [sf.values() for sf in sparse_features_list]
    all_offsets_list = [sf.offsets() for sf in sparse_features_list]
    
    # 计算累积长度（用于调整 offsets）
    lengths = [len(values) for values in all_values_list]
    cumulative_lengths = torch.cumsum(torch.tensor([0] + lengths[:-1], dtype=torch.int32), dim=0)
    
    # 调整每个样本的 offsets
    adjusted_offsets = []
    for i, offsets in enumerate(all_offsets_list):
        adjusted = offsets[:-1] + cumulative_lengths[i]
        adjusted_offsets.append(adjusted)
    
    # 拼接所有 values 和 offsets
    all_values = torch.cat(all_values_list)
    all_offsets = torch.cat(adjusted_offsets)
    all_offsets = torch.cat([all_offsets, torch.tensor([all_values.size(0)], dtype=torch.int32)])

# 构造批量化 KeyedJaggedTensor

    batch_sparse_features = KeyedJaggedTensor.from_offsets_sync(
        keys=sparse_features_list[0].keys(),
        values=all_values,
        offsets=all_offsets,
        weights=torch.full_like(all_values, 0.5, dtype=torch.float32)
    )

    return dense_features, batch_sparse_features, labels


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TorchRec test installation")
    parser.add_argument("--cpu_only", action="store_true")
    parser.add_argument('--samples', type=int, default=None, help='Number of samples')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--num_sparse', type=int, default=None, help='Number of sparse features')
    parser.add_argument('--num_embeddings', type=int, default=None, help='Number of embeddings')
    parser.add_argument('--embedding_scope', type=int, default=None, help='Embedding scope')
    parser.add_argument('--compute_kernel', type=str, choices=[member.value for member in EmbeddingComputeKernel],default=None, help='Compute kernel') # 1 fused, 2 fused_uvm, 3 fused_uvm_caching
    parser.add_argument('--profilling', action='store_true', help='Enable profiling')
    parser.add_argument('--shard_type', type=str, choices=[member.value for member in ShardingType], default=None, help='shard type') # 1 table wise, 3 row wise
    parser.add_argument('--ids_feature', type=int, default=10, help='ids per feature')
    parser.add_argument('--weighted', action='store_true', help='Enable weighted embedding')
    return parser.parse_args(argv)

def rank0print(msg: str) -> None:
    if dist.get_rank() == 0:
        print(msg)

@record
def main(argv: List[str]) -> None:
    args = parse_args(argv)
    SAMPLES = 10_240_000//10 if args.samples is None else args.samples
    batch_size = 2048 if args.batch_size is None else args.batch_size
    num_dense = 20 #dense数量
    num_sparse = 1 if args.num_sparse is None else args.num_sparse
    #1_060_000_000 --> 64G
    num_embeddings = 1_060_000_000//2//1 if args.num_embeddings is None else args.num_embeddings
    num_ids_per_feature = 10 if args.ids_feature is None else args.ids_feature
    num_cpu_workers = min(16, os.cpu_count())
    print(f"num_cpu_workers: {num_cpu_workers}")
    profilling =args.profilling
    IS_WEIGHTED = args.weighted
    embeding_scope = num_embeddings if args.embedding_scope is None else args.embedding_scope
    # FUSED = "fused"  1
    # FUSED_UVM = "fused_uvm"  2
    # FUSED_UVM_CACHING = "fused_uvm_caching"  3
    compute_kernel = EmbeddingComputeKernel.FUSED_UVM_CACHING.value if args.compute_kernel is None else args.compute_kernel
    # row wise = 3
    # table wise = 1
    shard_type = ShardingType.ROW_WISE.value if args.shard_type is None else args.shard_type

    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')


    configs = [
        EmbeddingBagConfig(
            name=f"table{id}",
            embedding_dim=16,#16 dimentions
            num_embeddings=num_embeddings,
            feature_names=[f"feature{id}"],
        )
        for id in range(num_sparse)
    ]
    
    
    if not args.cpu_only and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
        device_str = "cuda"
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        backend = "gloo"
        device_str = "cpu"

        print(
            "\033[92m"
            + f"WARNING: Running in CPU mode. Cuda is available: {torch.cuda.is_available()}. CPU only: {args.cpu_only}"
        )
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    print(f"{rank} is running")


    topology = Topology(world_size=dist.get_world_size(), compute_device=device_str,hbm_cap = 40*1024**3, local_world_size=2)
    constraints = {
        "table"+str(i): ParameterConstraints(
            sharding_types=[shard_type],
            compute_kernels=[compute_kernel],
            cache_params=CacheParams(
                algorithm=CacheAlgorithm.LRU,
                load_factor=0.2,
                reserved_memory=0.0,
                precision=DataType.FP32,
            )
        ) for i in range(num_sparse)
    }
    fused_params_for_uvm = {
        "cache_load_factor": 0.1,  # 这是一个与缓存相关的参数，可能需要调整
        "cache_ops": EmbeddingComputeKernel.FUSED_UVM_CACHING.value, # 有时也在这里指定
        # 其他 fused optimizer 参数:
        "optimizer": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
        "learning_rate": 0.1,
        "eps": 1e-8,
        "caching_ratio": 0.5,  # <--- caching_ratio 放在这里！
        "use_uvm": True, # 确保UVM被启用，虽然FUSED_UVM_CACHING核心已经暗示了
    }

    rank0print(f"""*******************fused_params  cache_load_factor: {fused_params_for_uvm.get("cache_load_factor")}; caching_ratio: {fused_params_for_uvm.get("caching_ratio")}; load_factor: {constraints['table0'].cache_params.load_factor}*******************""")

    ebc=EmbeddingBagCollection(
        tables=configs, device=torch.device("meta"),is_weighted=IS_WEIGHTED,
    )

    plan = EmbeddingShardingPlanner(
        topology=topology, constraints=constraints, storage_reservation=FixedPercentageStorageReservation(percentage=0.05)
    ).plan(
        ebc, [cast(ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder(fused_params=fused_params_for_uvm))]
    )
    print(plan)
    model = DistributedModelParallel(
        ebc,
        device=device,
        plan=plan
    )
    dist_starttime = time.time()


    rank0print("*****************************************************")
    rank0print(f"Sharding Plan generated: {plan}")
    print(f"SAMPLE: {SAMPLES:,}, BATCH_SIZE: {batch_size}, DENSE: {num_dense}, SPARSE: {num_sparse}, EMBEDDINGS: {num_embeddings:,}, NUM_IDS_PER_FEATURE: {num_ids_per_feature}, EMBEDDING_SCOPE: {num_embeddings}, COMPUTE_KERNEL:{constraints['table0'].compute_kernels}")
    rank0print("*****************************************************")

    rank0print(f"Model dist time: {dist_starttime} - {time.time()} = {time.time() - dist_starttime}")



    # for name, param in model.named_parameters():
    #     if "embedding_bag_collection" in name:  # 根据嵌入表的名称过滤
    #         print(f"Embedding table parameter: {name}, requires_grad: {param.requires_grad}")   

    loss_fn = torch.nn.BCEWithLogitsLoss()
   
    dataset = RandomDataset(SAMPLES, num_dense, num_sparse, num_embeddings, num_ids_per_feature,num_cpu_workers,batch_size=batch_size, embedding_scope=embeding_scope)
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank)
    data_loader = DataLoader(dataset, sampler = sampler, batch_size=batch_size, num_workers=num_cpu_workers, pin_memory=not args.cpu_only,collate_fn=custom_collate_fn,prefetch_factor=4)

    # 添加时间记录

    for name, param in model.named_parameters():
        if "embedding_bag_collection" in name:
            rank0print(f"Rank is {dist.get_rank()}, Parameter: {name}, requires_grad: {param.requires_grad}")

    if profilling:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA,], profile_memory=True,
            with_stack=False ) as prof: # prof 对象将收集数据
            runprocess(device, data_loader, model,batch_size)
        rank0print("\n--- Profiler Key Averages (sorted by self_cpu_time_total) ---")
        rank0print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total", row_limit=10))
        rank0print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=10))
        trace_file = f"""profilling{rank}_{datetime.now().strftime("%Y%m%d%H%M")}.json"""
        if dist.get_rank() == 0:
            prof.export_chrome_trace(trace_file)
            rank0print(f"\nTrace file saved to: {trace_file}")
    else:
        runprocess(device, data_loader, model,batch_size)

    if dist.is_initialized():
        dist.destroy_process_group()
    print(
        "\033[92m" + "Successfully ran a few epochs for DLRM. Installation looks good!"
    )

def runprocess(device, data_loader, model,batch_size):
    rank0print("==================================params=======================")
    for name, param in model.named_parameters():
        if "embedding_bag_collection" in name:  # 根据嵌入表的名称过滤
            print(f"Embedding table parameter: {name}, requires_grad: {param.requires_grad}")   
    rank0print("==================================params=======================")
    start_time = time.time()
    last_process_time = start_time
    total_samples = 0
    PRINT_ROUND = 100
    for i, (dense_features, sparse_features, labels) in enumerate(data_loader):
        dense_features = dense_features.to(device)
        sparse_features = sparse_features.to(device=device)
        output = model(sparse_features).wait()
        labels = labels.to(device)

        loss = torch.mean(output.values())
        torch.sum(loss, dim=0).backward()
        

        if i %  PRINT_ROUND == 0:
            current_time = time.time()
            current_time_span = current_time - last_process_time
            last_process_time = current_time
            elapsed_time = current_time - start_time
            total_samples = (i+1) * batch_size
            throughput = total_samples / elapsed_time  # 每秒处理的样本数
            print(
                f"Round {i+1}/{len(data_loader)} completed. "
                f"Elapsed time: {elapsed_time:.2f}s, Throughput: {throughput:.2f} samples/s, Current batch throughput: {PRINT_ROUND*batch_size/current_time_span:.2f} samples/s. Current time: {get_beijing_time()}"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
