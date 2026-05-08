# ==============================================================================
# 模块概述：
# 本文件是 VLM2Vec 项目的训练入口脚本（train.py）。
# VLM2Vec 是一个将视觉语言模型（VLM）转化为通用多模态嵌入模型的项目。
# 本文件负责：
#   1. 解析命令行参数（模型参数、数据参数、训练参数）
#   2. 检查并恢复训练断点（checkpoint）
#   3. 初始化 Weights & Biases 实验追踪
#   4. 构建多模态嵌入模型（MMEBModel）及其处理器（Processor）
#   5. 加载并混合多个训练数据集
#   6. 创建自定义训练器（GradCacheLateProcessTrainer）并启动训练
#   7. 保存训练完成后的模型和处理器
#
# 在项目中的位置：
#   本文件是整个训练流程的顶层入口，用户通过运行 `python train.py` 启动训练。
#   它串联了 src/ 目录下的各个子模块：
#     - src.arguments：参数定义
#     - src.model：模型构建
#     - src.data：数据加载与整理
#     - src.trainer：自定义训练器
#     - src.utils：工具函数
#
# 本代码改编自 Tevatron 项目，原始代码风格保留了 Tevatron 的训练框架设计。
# ==============================================================================

# Adapted from Tevatron code

import logging  # Python 标准日志库，用于记录训练过程中的信息、警告和错误
import os.path  # 提供文件路径操作的辅助函数，如路径拼接、判断路径是否存在等
import sys      # 提供对 Python 解释器运行环境的访问，此处主要用于处理命令行参数和标准输出

# 配置日志系统：设置日志级别为 INFO，定义日志格式（时间戳 + 级别 + 模块名:行号 + 消息），
# 并将日志输出到标准输出（stdout），方便在分布式训练中统一收集日志
logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # Ensures logs appear in stdout
)

# 创建当前模块的日志记录器，后续通过 logger.info() 等方法输出日志
logger = logging.getLogger(__name__)

import sys      # 重复导入 sys（原代码如此，保留不改动），用于命令行参数处理
import torch    # PyTorch 深度学习框架，提供张量计算、GPU 加速、分布式训练等功能
import wandb    # Weights & Biases 实验追踪工具，用于记录和可视化训练指标
import yaml     # YAML 解析库，用于读取数据集配置文件（.yaml 格式）

# HfArgumentParser：HuggingFace 提供的参数解析器，
# 可以将命令行参数自动解析为 dataclass 对象，比 argparse 更简洁
from transformers import HfArgumentParser

# 从项目自定义的参数模块中导入三类参数定义：
#   ModelArguments：模型相关参数（如模型名称、LoRA 配置、池化方式等）
#   DataArguments：数据相关参数（如数据集配置路径、最大序列长度、图像分辨率等）
#   TrainingArguments：训练相关参数（如学习率、批次大小、梯度缓存配置等，继承自 HuggingFace TrainingArguments）
from src.arguments import ModelArguments, DataArguments, TrainingArguments

# MultimodalDataCollator：多模态数据整理器，
# 负责将原始数据样本整理成模型可输入的批量张量格式（包括文本 tokenization、图像预处理等）
from src.data.collator.train_collator import MultimodalDataCollator

# init_mixed_dataset：初始化混合数据集的函数，
# 支持按权重从多个数据集中交错采样，实现多任务联合训练
from src.data.loader.mixed_dataset import init_mixed_dataset

# MMEBModel：多模态嵌入模型（Multimodal Embedding Model），
# 基于视觉语言模型构建，通过对比学习训练，使模型能生成高质量的文本-图像联合嵌入表示
from src.model.model import MMEBModel

# GradCacheLateProcessTrainer：自定义训练器，
# 继承自 HuggingFace Trainer，支持梯度缓存（GradCache）技术，
# 可在有限显存下使用更大的全局批次大小进行对比学习训练
from src.trainer import GradCacheLateProcessTrainer

# 工具函数：
#   print_rank：打印当前分布式进程的 rank 信息
#   print_master：仅在主进程（rank 0）上打印信息，避免多进程重复输出
#   find_latest_checkpoint：扫描输出目录，找到最新的训练断点路径
from src.utils.basic_utils import print_rank, print_master, find_latest_checkpoint

# 模型处理器相关函数：
#   load_processor：根据模型参数加载对应的处理器（tokenizer + image processor）
#   get_backbone_name：根据 HuggingFace 模型配置获取骨干网络名称（如 qwen2_vl、llava_next 等）
from src.model.processor import load_processor, get_backbone_name


def main():
    """
    主训练函数，执行完整的模型训练流程。

    流程概述：
        1. 处理分布式训练的命令行参数兼容性问题
        2. 解析三类训练参数（模型、数据、训练）
        3. 打印分布式训练调试信息
        4. 检查并恢复训练断点
        5. 初始化 Weights & Biases 实验追踪
        6. 构建模型、加载处理器
        7. 加载训练数据集和数据整理器
        8. 创建训练器并启动训练
        9. 保存训练结果

    参数：无（所有参数通过命令行传入）

    返回值：无
    """

    # ---- 第一步：处理分布式训练启动参数的兼容性问题 ----
    # 这是一个针对 torch.distributed.launch 的 hack 修复：
    # 旧版 PyTorch 使用 --local-rank=0 格式，而 HuggingFace 的参数解析器
    # 不识别这种格式，需要将其转换为 --local_rank 0 的标准格式
    # 参考：https://github.com/huggingface/transformers/issues/22171
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]       # 提取等号后面的 rank 值
            sys.argv.remove(arg)           # 移除旧格式的参数
            sys.argv.append('--local_rank') # 添加标准格式的参数名
            sys.argv.append(rank)           # 添加对应的参数值

    # ---- 第二步：解析命令行参数 ----
    # 使用 HfArgumentParser 将命令行参数解析为三个 dataclass 对象
    # 每个 dataclass 对应一组参数定义，包含类型检查和默认值
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    # parse_args_into_dataclasses() 将命令行参数自动映射到对应的 dataclass 字段
    # 返回三个参数对象：模型参数、数据参数、训练参数
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 添加类型注解，方便 IDE 进行类型提示和检查
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments


    # ---- 第三步：打印分布式训练调试信息 ----
    # 输出当前分布式训练环境的关键信息，便于排查分布式启动问题
    # 包括：全局 rank、本地 rank、总进程数、主节点地址和端口
    print("Distributed init debug info:")
    print(f"RANK: {os.environ.get('RANK')}")           # 全局进程编号
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}") # 当前节点上的进程编号
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}") # 总进程数
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}") # 主节点地址
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}") # 主节点端口

    # 检查 PyTorch 分布式是否已初始化，并打印当前进程的 rank 和 world_size
    if torch.distributed.is_available():
        print(f"torch.distributed.is_initialized: {torch.distributed.is_initialized()}")
        if torch.distributed.is_initialized():
            print(f"torch.distributed.get_rank(): {torch.distributed.get_rank()}")
            print(f"torch.distributed.get_world_size(): {torch.distributed.get_world_size()}")


    # ---- 第四步：检查并恢复训练断点（Checkpoint） ----
    # resume_from 参数支持三种模式：
    #   "auto"：自动检测输出目录中最新的断点并恢复
    #   数字字符串（如 "1000"）：恢复指定步数的断点
    #   其他值：不恢复断点，从头开始训练
    resume_checkpoint_dir = None  # 初始化断点路径为 None

    if training_args.resume_from == 'auto':
        # 自动模式：扫描输出目录，找到编号最大的 checkpoint 文件夹
        resume_checkpoint_dir = find_latest_checkpoint(training_args.output_dir)
        if resume_checkpoint_dir:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint_dir}")
    elif training_args.resume_from.isdigit():
        # 指定步数模式：根据步数拼接断点路径，如 output_dir/checkpoint-1000
        resume_checkpoint_dir = os.path.join(training_args.output_dir, f'checkpoint-{training_args.resume_from}')
        if os.path.exists(resume_checkpoint_dir):
            logger.info(f"Resuming from checkpoint: {resume_checkpoint_dir}")
    else:
        # 不恢复断点，从头开始训练
        resume_checkpoint_dir = None
        logger.info("No checkpoint found. Starting fresh training.")

    # ---- 第五步：初始化 Weights & Biases 实验追踪 ----
    # 仅在训练参数的 report_to 列表中包含 "wandb" 时启用
    # 且仅在主进程（rank 0）上初始化，避免多进程重复创建 wandb 运行
    if 'wandb' in training_args.report_to:
        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or (not torch.distributed.is_initialized()):
            print_rank('init wandb')
            # 初始化 wandb 运行，设置项目名和运行名
            wandb.init(project=training_args.project_name, name=training_args.run_name, mode="online")
            # 将所有参数记录到 wandb，方便后续分析不同参数配置对训练效果的影响
            wandb.config.update(model_args)
            wandb.config.update(data_args)
            wandb.config.update(training_args)

    # ---- 第六步：构建模型和加载处理器 ----
    # MMEBModel.build() 是工厂方法，根据 model_args 中的配置
    # （模型名称、是否使用 LoRA、checkpoint 路径等）构建多模态嵌入模型
    model = MMEBModel.build(model_args)

    # 获取模型的骨干网络名称（如 qwen2_vl、llava_next、phi3_v 等）
    # 这个名称决定了后续使用哪种处理器和数据处理逻辑
    model_backbone = get_backbone_name(hf_config=model.config)

    # 将骨干网络名称动态添加到 model_args 和 training_args 中，
    # 以便后续的数据整理器和训练器能根据骨干类型选择正确的处理逻辑
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_backbone}')

    # 根据模型参数和数据参数加载对应的处理器（Processor），
    # 处理器包含 tokenizer（文本分词器）和 image_processor（图像预处理器）
    processor = load_processor(model_args, data_args)

    # 将处理器绑定到模型对象上，方便训练器在需要时访问
    setattr(model, 'processor', processor)

    # ---- 第七步：加载训练数据集 ----
    # 从 YAML 配置文件中读取数据集配置，配置文件定义了：
    #   - 各数据集的路径、权重、解析器类型等
    #   - 图像目录路径等
    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_config = yaml.safe_load(yaml_file)  # 解析 YAML 文件为 Python 字典

        # 如果设置了 data_basedir（数据根目录），则将相对路径转换为绝对路径
        # 这样配置文件中可以使用相对路径，运行时自动补全为绝对路径
        if data_args.data_basedir:
            for _, task_config in dataset_config.items():
                image_dir = task_config.get('image_dir')
                if image_dir and not os.path.isabs(image_dir):
                    task_config['image_dir'] = os.path.join(data_args.data_basedir, image_dir)

        # 根据配置初始化混合数据集：
        #   - 按权重从多个数据集中交错采样
        #   - 支持分布式训练时按节点划分数据
        train_dataset = init_mixed_dataset(dataset_config, model_args, data_args, training_args)

    # 创建多模态数据整理器（Data Collator），
    # 负责将数据集中的原始样本整理成模型可接受的批量输入格式，
    # 包括文本 tokenization、图像预处理、填充（padding）等
    train_collator = MultimodalDataCollator(processor, model_args, data_args, training_args)

    # ---- 第八步：创建训练器并启动训练 ----
    # 使用 GradCacheLateProcessTrainer 训练器，
    # 它支持梯度缓存（GradCache）技术，可以在显存有限的情况下
    # 通过分块计算梯度来模拟更大的全局批次大小，对对比学习尤为重要
    trainer_cls = GradCacheLateProcessTrainer
    trainer = trainer_cls(
        model=model,                          # 多模态嵌入模型
        processing_class=processor,           # 数据处理器（tokenizer + image processor）
        args=training_args,                   # 训练参数
        model_args=model_args,                # 模型参数（自定义扩展）
        train_dataset=train_dataset,          # 训练数据集
        data_collator=train_collator,         # 数据整理器
        max_length=data_args.max_len,         # 最大输入序列长度，防止过长的输入导致显存溢出
    )

    # 将训练器引用绑定到数据集上，
    # 这是为了支持训练过程中动态调整数据采样策略（如根据训练步数切换数据集）
    train_dataset.trainer = trainer

    # 启动训练：
    #   - 如果 resume_checkpoint_dir 不为 None，则从断点恢复训练
    #   - 否则从头开始训练
    trainer.train(resume_from_checkpoint=resume_checkpoint_dir)

    # 训练完成后，将模型保存到输出目录
    trainer.save_model(training_args.output_dir)

    # 仅在主进程（world_process_zero）上保存处理器配置，
    # 避免多进程同时写入导致文件冲突
    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)


# ---- 脚本入口 ----
# 当直接运行 `python train.py` 时，调用 main() 函数启动训练
if __name__ == "__main__":
    main()
