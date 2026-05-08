# ==============================================================================
# 模块概述：eval.py —— VLM2Vec 项目的评估入口脚本
# ==============================================================================
# 本文件是 VLM2Vec（Vision-Language Model to Vector）项目的核心评估脚本，
# 负责对多模态嵌入模型（MMEBModel）进行检索/排序任务的评估。
#
# 主要功能：
#   1. 在 DDP（分布式数据并行）环境下加载模型和数据集
#   2. 对查询（query）和候选（candidate）分别编码为嵌入向量
#   3. 支持两种模型类型的评分：
#      - 标准稠密模型：通过余弦相似度计算得分
#      - Late-interaction 模型（如 ColPali）：通过 token 级别的交互评分
#   4. 计算排序指标（NDCG、Hit、MAP、MRR 等）并保存结果
#
# 在项目中的位置：
#   本文件是项目评估流程的入口，被直接通过 `python eval.py` 调用。
#   它依赖 src/ 目录下的模型定义、数据处理和工具函数。
# ==============================================================================

import datetime  # 用于设置 DDP 进程组的超时时间
import logging  # 用于日志记录
import json  # 用于 JSON 格式的数据读写（数据集信息、评分结果等）
import random  # 用于生成随机延迟，避免多进程同时加载模型导致 I/O 冲突
import time  # 用于在非主进程中添加随机延迟

import numpy as np  # 用于数值计算，特别是嵌入向量的矩阵运算
import os  # 用于文件路径操作和环境变量读取
import pickle  # 用于序列化保存/加载嵌入向量（.pkl 格式）
import sys  # 用于解析命令行参数
import torch  # PyTorch 深度学习框架，用于张量运算和 GPU 加速
import torch.distributed as dist  # PyTorch 分布式通信模块，用于多 GPU 间的同步和数据聚合
import torch.nn.functional as F  # PyTorch 函数式接口，用于 padding 等操作
import yaml  # 用于解析 YAML 格式的数据集配置文件

from torch.utils.data import DataLoader  # PyTorch 数据加载器，用于批量加载数据
from tqdm import tqdm  # 进度条工具，用于显示编码和评分的进度
from transformers import HfArgumentParser, AutoConfig  # HfArgumentParser: 解析命令行参数为数据类；AutoConfig: 加载 HuggingFace 模型配置
from datasets import Dataset, concatenate_datasets  # Dataset: HuggingFace 数据集对象；concatenate_datasets: 拼接多个数据集
from datasets.distributed import split_dataset_by_node  # 按分布式节点拆分数据集，确保每个 GPU 处理不同的数据子集

from src.arguments import ModelArguments, DataArguments, TrainingArguments  # 项目自定义的参数数据类，分别定义模型、数据、训练相关参数
from src.data.collator.eval_collator import MultimodalEvalDataCollator  # 评估阶段的数据整理器，将原始数据整理为模型可接受的输入格式
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, generate_cand_dataset  # AutoEvalPairDataset: 自动实例化评估数据集；generate_cand_dataset: 从语料库生成候选数据集
from src.utils.eval_utils.metrics import RankingMetrics  # 排序评估指标计算工具，支持 NDCG、Hit、MAP、MRR 等
from src.model.model import MMEBModel  # 多模态嵌入模型的主类，封装了查询和候选的编码逻辑
from src.model.processor import get_backbone_name, load_processor, COLPALI  # get_backbone_name: 根据配置获取模型骨干名称；load_processor: 加载数据处理器；COLPALI: ColPali 模型骨干标识常量
from src.utils.basic_utils import batch_to_device, print_rank, print_master  # batch_to_device: 将数据批次移至指定设备；print_rank/print_master: 仅在指定进程打印日志，避免多进程重复输出

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s')
logger = logging.getLogger(__name__)


def pad_dataset_to_divisible(dataset, world_size):
    """
    将数据集填充至可被 world_size 整除的长度，确保 DDP 分布式评估时每个进程分到相同数量的样本。

    在分布式评估中，数据集需要被均匀分配到各个 GPU 上。如果数据集大小不能被 GPU 数量整除，
    则后面的进程会少分到数据，导致 all_gather 时各进程数据量不一致而出错。
    本函数通过复制数据集前几条样本来补齐，使总样本数能被 world_size 整除。

    参数:
        dataset: HuggingFace Dataset 对象，待填充的数据集
        world_size: int，分布式训练的进程数（GPU 数量）

    返回值:
        padded_dataset: 填充后的数据集（若已可整除则返回原数据集）
        padded_size: int，填充后的数据集大小
    """
    num_samples = len(dataset)
    if num_samples % world_size == 0:
        return dataset, num_samples

    num_to_add = world_size - (num_samples % world_size)  # 需要补充的样本数量
    padded_size = num_samples + num_to_add

    padding_data = dataset.select([i % len(dataset) for i in range(num_to_add)])  # 从数据集头部循环选取样本作为填充
    padded_dataset = concatenate_datasets([dataset, padding_data])  # 将填充样本拼接到原数据集末尾
    return padded_dataset, padded_size


def encode_embeddings(
    model: MMEBModel,
    loader: DataLoader,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    full_dataset: Dataset,
    encode_side: str,
    description: str = "Encoding"
) -> tuple[np.ndarray, list]:
    """
    使用模型对给定数据集进行编码，生成嵌入向量。支持标准稠密模型和 late-interaction 模型，
    并在 DDP 环境下安全地聚合所有进程的结果。

    核心流程：
      1. 每个进程独立编码自己分到的数据子集
      2. 对于 late-interaction 模型，需要先同步全局最大序列长度并对嵌入做 padding
      3. 通过 all_gather 将所有进程的嵌入和元数据聚合到每个进程

    参数:
        model: MMEBModel，多模态嵌入模型实例
        loader: DataLoader，数据加载器，提供批量数据
        training_args: TrainingArguments，训练参数（包含设备信息等）
        model_args: ModelArguments，模型参数（包含模型骨干类型等）
        full_dataset: Dataset，完整的（未拆分的）数据集，用于确定 all_gather 后的总数据量
        encode_side: str，编码方向，"qry" 表示编码查询，"cand" 表示编码候选
        description: str，进度条显示的描述信息

    返回值:
        final_embeddings: np.ndarray，聚合后的嵌入向量数组
            - 稠密模型: 形状为 [N, H]，N 为样本数，H 为嵌入维度
            - Late-interaction 模型: 形状为 [N, L, H]，L 为序列长度，H 为嵌入维度
        all_gt_infos: list，聚合后的元数据列表
            - 编码查询时: 包含每个查询的完整信息字典
            - 编码候选时: 包含每个候选的名称（cand_name）字符串
    """
    local_rank = dist.get_rank() if dist.is_initialized() else 0  # 当前进程在分布式组中的编号
    world_size = dist.get_world_size() if dist.is_initialized() else 1  # 分布式进程总数

    is_late_interaction = (model_args.model_backbone == COLPALI)  # 判断是否为 late-interaction 模型（如 ColPali）

    local_embeds = []  # 存储当前进程编码得到的所有批次的嵌入向量
    local_gt_infos = []  # 存储当前进程编码得到的所有样本的元数据
    local_max_len = 0  # 当前进程观察到的最大序列长度（仅 late-interaction 模型使用）

    model.eval()  # 将模型切换到评估模式，关闭 dropout 和 batchnorm 的训练行为
    with torch.no_grad():  # 禁用梯度计算，减少内存占用并加速推理
        for inputs, dataset_info in tqdm(loader, desc=f"{description} (rank {local_rank})", disable=local_rank > 0):
            inputs = batch_to_device(inputs, training_args.device)  # 将输入数据移至 GPU
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):  # 启用 bfloat16 混合精度推理，加速计算
                if encode_side == "qry":  # 编码查询侧
                    output = model(qry=inputs)  # 调用模型的查询编码器
                    reps = output["qry_reps"].detach()  # 获取查询嵌入并从计算图分离
                    local_gt_infos.extend(dataset_info)  # 保留每个查询的完整信息（包含标签、候选列表等）
                else:  # 编码候选侧
                    output = model(tgt=inputs)  # 调用模型的候选编码器
                    reps = output["tgt_reps"].detach()  # 获取候选嵌入并从计算图分离
                    local_gt_infos.extend([info["cand_name"] for info in dataset_info])  # 仅保留候选名称，用于后续构建候选嵌入字典

            if is_late_interaction and reps.dim() == 3:  # late-interaction 模型的嵌入为 3D 张量 [batch, seq_len, dim]
                local_max_len = max(local_max_len, reps.shape[1])  # 更新当前进程观察到的最大序列长度

            local_embeds.append(reps)

    if not local_embeds:
        return np.array([]), []  # 如果当前进程没有数据，返回空结果

    # === DDP 同步与填充：Late-Interaction 模型的特殊处理 ===
    # late-interaction 模型（如 ColPali）的嵌入是 3D 张量 [batch, seq_len, dim]，
    # 不同样本的序列长度可能不同。为了在 all_gather 时正确拼接，需要：
    #   1. 同步所有进程的最大序列长度
    #   2. 将所有嵌入 padding 到统一长度
    if is_late_interaction:
        if dist.is_initialized():
            # 步骤1：通过 all_reduce 的 MAX 操作，获取所有进程中的最大序列长度
            local_max_len_tensor = torch.tensor(local_max_len, device=training_args.device)
            dist.all_reduce(local_max_len_tensor, op=dist.ReduceOp.MAX)  # 取所有进程的最大值
            global_max_len = local_max_len_tensor.item()
        else:
            global_max_len = local_max_len

        # 步骤2：将当前进程的所有嵌入 padding 到全局最大序列长度
        padded_embeds = []
        for reps_batch in local_embeds:
            if reps_batch.dim() == 3:  # 3D 张量需要 padding
                B, L, H = reps_batch.shape  # B: batch大小, L: 当前序列长度, H: 嵌入维度
                padding_size = global_max_len - L  # 需要填充的长度
                padded_batch = F.pad(reps_batch, (0, 0, 0, padding_size), "constant", 0)  # 在序列维度右侧填 0
                padded_embeds.append(padded_batch)
            else:  # 理论上 late-interaction 模型不应出现 2D 嵌入，此处为防御性处理
                padded_embeds.append(reps_batch)

        embeds_tensor = torch.cat(padded_embeds, dim=0).contiguous()  # 沿 batch 维度拼接所有批次
    else:  # 标准稠密模型：嵌入为 2D 张量 [batch, dim]，无需 padding
        embeds_tensor = torch.cat(local_embeds, dim=0).contiguous()


    # === 从所有进程聚合嵌入和元数据 ===
    if dist.is_initialized() and full_dataset.num_rows >= world_size:
        print_master(f"Gathering {encode_side} embeddings across all ranks...")

        # 使用 all_gather_into_tensor 高效聚合张量数据
        output_shape = list(embeds_tensor.shape)
        output_shape[0] = full_dataset.num_rows  # 设置总样本数为完整数据集大小
        embeds_tensor = embeds_tensor.to(training_args.device)
        gathered_embeds_tensor = torch.empty(output_shape, dtype=embeds_tensor.dtype, device=training_args.device)
        dist.all_gather_into_tensor(gathered_embeds_tensor, embeds_tensor)  # 将所有进程的嵌入拼接在一起
        final_embeddings = gathered_embeds_tensor.cpu().float().numpy()  # 转为 CPU 上的 float32 numpy 数组

        # 聚合元数据（非张量数据需使用 all_gather_object）
        gathered_gt_infos = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_gt_infos, local_gt_infos)  # 收集所有进程的元数据列表
        all_gt_infos = [key for rank_keys in gathered_gt_infos for key in rank_keys]  # 展平为单一列表
    else:  # 单进程模式：无需聚合
        all_gt_infos = local_gt_infos
        final_embeddings = embeds_tensor.cpu().float().numpy()

    return final_embeddings, all_gt_infos


def main():
    """
    评估主函数，执行完整的评估流程：

    1. 初始化分布式环境（如果可用）
    2. 解析命令行参数
    3. 以 DDP 安全的方式加载模型（主进程先下载，其余进程从缓存加载）
    4. 遍历配置文件中的每个数据集，依次完成：
       a. 加载并拆分数据集
       b. 编码查询嵌入
       c. 编码候选候选嵌入
       d. 计算相似度得分和排序指标
       e. 保存评分结果
    """
    # --- 初始化分布式进程组 ---
    # 检测环境变量 RANK 判断是否在分布式环境中运行
    if "RANK" in os.environ and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=60))  # 使用 NCCL 后端初始化进程组，超时 60 分钟
    local_rank = dist.get_rank() if dist.is_initialized() else 0  # 当前进程的局部编号
    world_size = dist.get_world_size() if dist.is_initialized() else 1  # 总进程数

    # 打印分布式调试信息（仅主进程打印全局信息，每个进程打印自己的 rank 信息）
    print_master("Distributed init debug info:")
    print_master(f"RANK: {os.environ.get('RANK')}")
    print_master(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    print_master(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    print_master(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print_master(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")
    if dist.is_initialized():
        print_rank(f"dist.get_rank(): {dist.get_rank()}")
        print_rank(f"dist.get_world_size(): {dist.get_world_size()}")

    # --- 解析命令行参数 ---
    # 兼容旧版 torch.distributed.launch 的 --local-rank= 格式
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()  # 将命令行参数解析为三个数据类
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    os.makedirs(data_args.encode_output_path, exist_ok=True)  # 创建嵌入输出目录

    # --- 模型加载 ---
    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)  # 加载 HuggingFace 模型配置
    # 如果未指定 model_backbone，则根据模型配置和类型自动推断
    if not getattr(model_args, "model_backbone", None):
        model_backbone = get_backbone_name(hf_config=hf_config, model_type=model_args.model_type)
        setattr(model_args, 'model_backbone', model_backbone)
        setattr(training_args, 'model_backbone', model_backbone)
    print_master(f'Model Backbone: {model_args.model_backbone}')

    # --- DDP 安全的模型加载策略 ---
    # 步骤1：仅主进程（rank 0）下载模型，避免多进程同时下载导致冲突
    if local_rank == 0:
        processor = load_processor(model_args, data_args)  # 加载数据处理器（tokenizer、图像处理器等）
        model = MMEBModel.load(model_args, is_trainable=False, processor=processor)  # 加载模型，设为不可训练
        print_master(f"[rank=0] Loading the model from Huggingface: {model_args.model_name}...")
    # 步骤2：所有进程在此等待，直到主进程完成下载
    if torch.distributed.is_initialized():
        torch.distributed.barrier()  # 屏障同步：确保 rank 0 完成下载后其他进程才继续
    # 步骤3：非主进程从本地缓存加载模型（此时模型已被主进程下载并缓存）
    if local_rank != 0:
        print_rank(f"Loading the model from cache...")
        processor = load_processor(model_args, data_args)
        time.sleep(random.randint(2 * local_rank, 3 * local_rank))  # 随机延迟，避免多进程同时读取缓存导致 I/O 竞争
        model = MMEBModel.load(model_args, is_trainable=False, processor=processor)
    model.eval()  # 切换到评估模式
    model = model.to(training_args.device, dtype=torch.bfloat16)  # 将模型移至 GPU 并转为 bfloat16 精度

    # --- 加载数据集配置文件 ---
    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_configs = yaml.safe_load(yaml_file)  # 读取 YAML 配置，获取所有待评估数据集的信息


    # --- 主评估循环：逐个数据集进行评估 ---
    for dataset_idx, (dataset_name, task_config) in enumerate(dataset_configs.items()):
        # 进程同步，确保所有进程同时开始处理当前数据集
        if dist.is_initialized():
            dist.barrier()
        print_master(f"--- Evaluating {dataset_name} ---")

        # 对容易 OOM（显存溢出）的数据集动态降低批大小
        current_batch_size = training_args.per_device_eval_batch_size
        if dataset_name in ["Charades-STA", "QVHighlight", "MomentSeeker", "YouCook2", "Video-MME"]:
            current_batch_size = 4  # 这些数据集的视频/图像较大，降低批大小以避免显存不足
            print_master(f"⚠️  Reduced batch size to {current_batch_size} for {dataset_name}")

        # 定义嵌入和元数据的保存路径
        query_embed_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_qry")  # 查询嵌入保存路径
        cand_embed_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_tgt")  # 候选嵌入保存路径
        dataset_info_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_info.jsonl")  # 数据集元信息保存路径

        # 检查是否需要重新编码查询或候选（如果已有缓存则跳过）
        do_query = not os.path.exists(query_embed_path) or not os.path.exists(dataset_info_path)
        do_cand = not os.path.exists(cand_embed_path)

        if do_query or do_cand:
            # 如果指定了数据根目录，将配置中的相对路径转为绝对路径
            if data_args.data_basedir is not None:
                for key in ["image_root", "video_root", "frame_root", "clip_root", "data_path"]:
                    if data_args.data_basedir and task_config.get(key):
                        task_config[key] = os.path.join(data_args.data_basedir, task_config[key])

            try:
                # 实例化评估数据集：full_eval_qry_dataset 为查询数据集，corpus 为候选语料库
                full_eval_qry_dataset, corpus = AutoEvalPairDataset.instantiate(model_args=model_args, data_args=data_args, **task_config)
                # 从查询数据集和语料库生成候选数据集
                full_eval_cand_dataset = generate_cand_dataset(full_eval_qry_dataset, corpus)
                eval_qry_dataset, eval_cand_dataset = full_eval_qry_dataset, full_eval_cand_dataset

                # 分布式环境下：先填充数据集使其可被 world_size 整除，再按进程拆分
                if dist.is_initialized():
                    padded_qry_dataset, _ = pad_dataset_to_divisible(full_eval_qry_dataset, world_size)  # 填充查询数据集
                    padded_cand_dataset, _ = pad_dataset_to_divisible(full_eval_cand_dataset, world_size)  # 填充候选数据集
                    eval_qry_dataset = split_dataset_by_node(padded_qry_dataset, rank=local_rank, world_size=world_size)  # 按进程拆分查询数据
                    eval_cand_dataset = split_dataset_by_node(padded_cand_dataset, rank=local_rank, world_size=world_size)  # 按进程拆分候选数据
                else:
                    padded_qry_dataset, padded_cand_dataset = full_eval_qry_dataset, full_eval_cand_dataset
            except Exception as e:
                print_master(f"Failed to load dataset {dataset_name}, skipping {dataset_name}")
                import traceback
                traceback.print_exc()
                print_master(e)
                raise e
                continue

        # --- 步骤1：编码查询嵌入 ---
        if do_query:
            print_master("Encoding queries...")
            eval_qry_collator = MultimodalEvalDataCollator(processor, model_args, data_args, "qry")  # 查询侧的数据整理器
            eval_qry_loader = DataLoader(eval_qry_dataset, batch_size=current_batch_size, collate_fn=eval_qry_collator, num_workers=training_args.dataloader_num_workers)
            query_embeds, gt_infos = encode_embeddings(model, eval_qry_loader, training_args, model_args, padded_qry_dataset, encode_side="qry", description=f"Queries for {dataset_name}")
            query_embeds = query_embeds[:len(full_eval_qry_dataset)]  # 裁剪掉因 DDP padding 而多出的数据点
            gt_infos = gt_infos[:len(full_eval_qry_dataset)]  # 同步裁剪元数据
            # 仅主进程保存嵌入和元信息到磁盘
            if local_rank == 0:
                with open(query_embed_path, 'wb') as f:
                    pickle.dump(query_embeds, f)  # 保存查询嵌入为 pkl 文件
                with open(dataset_info_path, 'w') as f:
                    for info in gt_infos:
                        f.write(json.dumps(info) + '\n')  # 每行一个 JSON 对象，保存查询的元信息
                print_master(f"Saved query embeddings to {query_embed_path}")
            if dist.is_initialized():
                dist.barrier()  # 确保所有进程在嵌入保存完成后再继续


        # --- 步骤2：编码候选嵌入 ---
        if do_cand:
            print_master("Encoding candidates...")
            eval_cand_collator = MultimodalEvalDataCollator(processor, model_args, data_args, "cand")  # 候选侧的数据整理器
            eval_cand_loader = DataLoader(eval_cand_dataset, batch_size=current_batch_size, collate_fn=eval_cand_collator, num_workers=training_args.dataloader_num_workers)

            cand_embeds, all_cand_ids = encode_embeddings(model, eval_cand_loader, training_args, model_args, padded_cand_dataset, encode_side="cand", description=f"Candidates for {dataset_name}")
            cand_embeds = cand_embeds[:len(full_eval_cand_dataset)]  # 裁剪掉因 DDP padding 而多出的数据点
            all_cand_ids = all_cand_ids[:len(full_eval_cand_dataset)]  # 同步裁剪候选 ID

            # 仅主进程保存候选嵌入（以字典形式：候选名称 -> 嵌入向量）
            if local_rank == 0:
                cand_embed_dict = {cand_id: embed for cand_id, embed in zip(all_cand_ids, cand_embeds)}  # 构建候选名称到嵌入的映射字典
                with open(cand_embed_path, 'wb') as f: pickle.dump(cand_embed_dict, f)  # 保存候选嵌入字典为 pkl 文件
                print_master(f"Saved candidate embeddings to {cand_embed_path}")

        if dist.is_initialized():
            dist.barrier()  # 确保所有进程的编码工作完成后再进入评分阶段

        # --- 步骤3：计算评分（仅在主进程执行） ---
        if local_rank == 0:
            score_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_score.json")
            # 如果评分文件已存在，尝试直接加载并跳过计算
            if os.path.exists(score_path):
                try:
                    with open(score_path, "r") as f:
                        score_dict = json.load(f)
                    print_master(f"Score of {dataset_name} (loaded from previous run): {score_path}")
                    formatted = {k: f"{v:.4f}" for k, v in score_dict.items()}
                    print_master(formatted)
                    continue  # 跳过当前数据集的评分计算
                except Exception as e:
                    print_master(f"Failed to load score for {dataset_name}, skipping {dataset_name}")

            # 加载之前保存的查询嵌入、候选嵌入和元信息
            with open(query_embed_path, 'rb') as f: qry_embeds = pickle.load(f)
            with open(cand_embed_path, 'rb') as f: cand_embed_dict = pickle.load(f)
            gt_infos = [json.loads(l) for l in open(dataset_info_path)]
            pred_dicts = []  # 存储每个查询的预测结果（排序后的候选列表和真实标签）

            # 判断评估类型：global 表示全局排序（查询与所有候选计算相似度），否则为局部排序（仅与给定候选列表排序）
            rank_against_all_candidates = task_config.get("eval_type", "global") == "global"

            if rank_against_all_candidates:
                # === 全局排序模式：每个查询与所有候选计算相似度并排序 ===
                cand_keys = list(cand_embed_dict.keys())  # 所有候选的名称列表
                cand_embeds = np.stack([cand_embed_dict[key] for key in cand_keys])  # 将候选嵌入堆叠为矩阵 [N_c, H] 或 [N_c, L, H]

                if qry_embeds.ndim == 3:  # Late-interaction 模式：查询嵌入为 3D [N_q, L_q, H]，候选嵌入为 3D [N_c, L_c, H]
                    qry_embed = torch.from_numpy(qry_embeds)  # 转为 PyTorch 张量
                    cand_embeds = [torch.from_numpy(np.array(t)) for t in cand_embeds]  # 每个候选的嵌入单独转为张量（长度可能不同）
                    scores = processor.score(qry_embed, cand_embeds, batch_size=64)  # 使用 ColPali 的评分函数计算 token 级别的交互得分
                    ranked_candids = torch.argsort(-scores, dim=1).cpu().numpy().tolist()  # 按得分降序排列候选索引
                else:  # 稠密模式：查询和候选嵌入均为 2D，直接计算余弦相似度
                    cosine_scores = np.dot(qry_embeds, cand_embeds.T)  # 矩阵乘法计算所有查询-候选的余弦相似度 [N_q, N_c]
                    ranked_candids = np.argsort(-cosine_scores, axis=1)  # 按相似度降序排列候选索引

                # 遍历每个查询，构建预测结果
                for qid, (ranked_candid, gt_info) in tqdm(enumerate(zip(ranked_candids, gt_infos)), desc=f"Calculating scores for {dataset_name}"):
                    rel_docids = gt_info["label_name"] if isinstance(gt_info["label_name"], list) else [gt_info["label_name"]]  # 真实相关的候选名称列表
                    rel_scores = gt_info["rel_scores"] if "rel_scores" in gt_info else None  # 相关性评分（用于 graded 指标如 NDCG）
                    assert rel_scores is None or len(rel_docids) == len(rel_scores)  # 确保标签和评分数量一致
                    pred_dicts.append({
                        "prediction": [cand_keys[i] for i in ranked_candid],  # 按排序顺序的候选名称列表
                        "label": rel_docids,  # 真实相关的候选名称
                        "rel_scores": rel_scores,  # 相关性评分
                    })
            else:
                # === 局部排序模式：每个查询仅与配置中指定的候选列表排序 ===
                for qid, (qry_embed, gt_info) in tqdm(enumerate(zip(qry_embeds, gt_infos)), desc=f"Calculating scores for {dataset_name}"):
                    cand_embeds = np.stack([cand_embed_dict[key] for key in gt_info["cand_names"]])  # 仅取出当前查询的候选嵌入

                    if qry_embeds.ndim == 3:  # Late-interaction 模式
                        qry_embed = torch.from_numpy(np.array(qry_embed)).unsqueeze(0)  # 添加 batch 维度 [1, L_q, H]
                        cand_embeds = [torch.from_numpy(np.array(t)) for t in cand_embeds]  # 每个候选嵌入单独转为张量
                        scores = processor.score(qry_embed, cand_embeds, batch_size=1024)  # ColPali 评分
                        ranked_candids = torch.argsort(-scores, dim=1).cpu().numpy().tolist()[0]  # 取出第一个（也是唯一一个）查询的排序结果
                    else:  # 稠密模式
                        cosine_score = np.dot(qry_embed, cand_embeds.T)  # 计算当前查询与所有候选的余弦相似度
                        ranked_candids = np.argsort(-cosine_score)  # 按相似度降序排列

                    rel_docids = gt_info["label_name"] if isinstance(gt_info["label_name"], list) else [gt_info["label_name"]]
                    rel_scores = gt_info["rel_scores"] if "rel_scores" in gt_info else None

                    assert rel_scores is None or len(rel_docids) == len(rel_scores)
                    pred_dicts.append({
                        "prediction": [gt_info["cand_names"][i] for i in ranked_candids],  # 排序后的候选名称（来自当前查询的候选列表）
                        "label": rel_docids,
                        "rel_scores": rel_scores,
                    })

            # --- 计算排序指标 ---
            score_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_score.json")
            pred_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_pred.jsonl")

            # 从配置中获取要计算的指标列表，默认包含常用排序指标
            metrics_to_report = task_config["metrics"] if task_config.get("metrics", None) is not None else ["hit", "ndcg", "precision", "recall", "f1", "map", "mrr"]
            metrics = RankingMetrics(metrics_to_report)  # 初始化指标计算器
            score_dict = metrics.evaluate(pred_dicts)  # 计算所有指标
            formatted = {k: f"{v:.4f}" for k, v in score_dict.items()}  # 格式化为 4 位小数
            score_dict["num_pred"] = len(pred_dicts)  # 记录预测数量
            score_dict["num_data"] = len(gt_infos)  # 记录数据总量
            print_master(f"Score of {dataset_name}:")
            print_master(formatted)
            print_master(f"Outputting final score to: {score_path}")

            # 保存评分结果和预测详情
            with open(score_path, "w") as f:
                json.dump(score_dict, f, indent=4)  # 保存评分指标为 JSON 文件
            with open(pred_path, "w") as f:
                for pred in pred_dicts:
                    f.write(json.dumps(pred) + '\n')  # 每行一个 JSON 对象，保存每个查询的预测结果


if __name__ == "__main__":
    main()
