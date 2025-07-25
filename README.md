# torchrec
torchx run -s kubernetes  dist.ddp  -j 2x2 --gpu 2 --script "/app/embedding.py" --memMB 100000 --image "${IMAGE_NAME}" -- --samples 10240000 --batch_size 4096 --num_sparse 1 --compute_kernel fused --shard_type row_wise --num_embeddings 120000000 --weighted --ids_feature 10
