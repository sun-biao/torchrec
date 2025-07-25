FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime


# 设置工作目录
WORKDIR /app


RUN pip install fbgemm-gpu==1.1.0
RUN pip install torchrec==1.1.0
RUN pip install torchx==0.7.0, pytz


# 复制训练脚本
COPY dist_test.py /app/embedding.py
# 默认命令，但 TorchX 会覆盖它来运行分布式训练
CMD ["/bin/bash"]
