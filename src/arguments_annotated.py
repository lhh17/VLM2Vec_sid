"""
模块概述：
本模块定义了 VLM2Vec 项目中使用的所有命令行参数和数据配置。
通过 Python dataclass 机制，将参数分为四大类：
  1. ModelArguments  —— 模型相关参数（模型路径、LoRA 配置、池化方式等）
  2. DataArguments    —— 数据相关参数（数据集配置、图像分辨率、编码输出路径等）
  3. TrainingArguments —— 训练相关参数（继承自 HuggingFace TrainingArguments，扩展了梯度缓存等）
  4. MTEBArguments    —— MTEB 评测相关参数（设备、批次大小、评测任务等）

这些 dataclass 可与 HuggingFace 的 HfArgumentParser 配合使用，
将命令行参数自动解析为对应的 Python 对象，方便在训练和评测脚本中使用。
"""

from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import List


# ============================================================================
# ModelArguments：模型相关参数
# 用途：定义模型的加载方式、结构配置、LoRA 微调参数以及 UI 图相关的 token 选择策略。
# ============================================================================
@dataclass
class ModelArguments:
    # HuggingFace 模型名称或本地路径，例如 "google/siglip-so400m-patch14-384" 或 "/path/to/local/model"
    model_name: str = field(metadata={"help": "huggingface model name or path"})

    # 模型类型，通常可从模型配置文件中自动读取，但某些情况下需要手动指定
    # 例如："siglip"、"qwen2vl" 等
    model_type: str = field(default=None, metadata={"help": "model type, typically includes in config file, but sometimes needs mannually add"})

    # 处理器（processor）名称或路径，用于加载图像/文本预处理器
    # 若不指定，默认使用与 model_name 相同的处理器
    processor_name: str = field(default=None, metadata={"help": "processor_name, huggingface model name or path"})

    # 模型骨干网络类型，用于区分不同架构的视觉语言模型
    # 例如："siglip"、"clip" 等
    model_backbone: str = field(default=None, metadata={"help": "HF model type"})

    # 本地模型检查点路径，可以是完整模型路径，也可以是 LoRA 适配器路径
    # 用于从特定检查点恢复训练或推理
    checkpoint_path: str = field(default=None, metadata={"help": "a local model path, could be a LoRA version"})

    # 编码器的池化方法，决定如何将 token 级别的隐藏状态聚合为句子/图像级别的嵌入向量
    # 可选值："last"（取最后一个 token）、"mean"（取平均）、"cls" 等
    # 默认值 'last' 表示使用最后一个 token 的表示作为整体嵌入
    pooling: str = field(default='last', metadata={"help": "pooling method for encoder"})

    # 是否对查询和文档的表示向量进行 L2 归一化
    # 归一化后，相似度计算等价于余弦相似度；默认 False 表示不归一化
    normalize: bool = field(default=False, metadata={"help": "normalize query and passage representations"})

    # 对比学习损失函数中 softmax 的温度系数
    # 取值范围：正浮点数；值越小，模型对正样本的区分越锐利
    # 默认值 0.02 是对比学习中的常用值
    temperature: float = field(default=0.02, metadata={"help": "temperature for softmax"})

    # 是否使用 LoRA（Low-Rank Adaptation）进行参数高效微调
    # True 表示只训练低秩适配器参数，冻结原始模型权重
    lora: bool = field(default=False, metadata={"help": "do parameter-efficient fine-tuning with lora"})

    # LoRA 的秩（rank），控制适配器中低秩矩阵的维度
    # 取值范围：正整数；值越大，可训练参数越多，表达能力越强，但显存开销也越大
    # 默认值 16 是常用的平衡点
    lora_r: int = field(default=16, metadata={"help": "lora r"})

    # LoRA 的缩放因子 alpha，实际缩放比例为 lora_alpha / lora_r
    # 值越大，LoRA 更新的幅度越大；默认值 64 对应缩放比例 64/16=4
    lora_alpha: int = field(default=64, metadata={"help": "lora alpha"})

    # LoRA 的 dropout 概率，用于正则化防止过拟合
    # 取值范围：[0.0, 1.0)；0.0 表示无 dropout
    # 默认值 0.1 表示 10% 的神经元随机失活
    lora_dropout: float = field(default=0.1, metadata={"help": "lora dropout"})

    # LoRA 需要应用的目标模块名称，以逗号分隔
    # 包括注意力层的 q/k/v/o 投影以及 FFN 层的 gate/up/down 投影
    # 默认值覆盖了常见 Transformer 模型中的主要线性层
    lora_target_modules: str = field(default="qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj,gate_proj,up_proj", metadata={"help": "lora target modules"})

    # 图像编码器使用的裁剪数量，用于将高分辨率图像切分为多个子图分别编码
    # 取值范围：正整数；值越大，图像细节保留越多，但计算开销也越大
    # 默认值 16 是常见的设置
    num_crops: int = field(default=16, metadata={"help": "number of crops used in image encoder"})

    # 是否启用 UI 图（UI Graph）进行 token 选择
    # UI 图基于图像的视觉结构来选择重要的 token，减少冗余计算
    uigraph_use: bool = field(default=False, metadata={"help": "Enable ui graph for token selection"})

    # 构建 UI 图时使用的像素差异阈值
    # 取值范围：正整数；值越大，对差异的敏感度越低，图越稀疏
    # 默认值 1 表示对最微小的像素差异也敏感
    uigraph_diff: int = field(default=1, metadata={"help": "Pixel difference used for constructing ui graph for token selection"})

    # 是否使用随机图构建替代基于像素差异的 UI 图构建
    # True 表示随机选择连接关系，用于消融实验或对比
    uigraph_rand: bool = field(default=False, metadata={"help": "Enable random graph construction for token selection"})

    # 在 UI 图 token 选择中，每个连通分量跳过的 patch token 比例
    # 取值范围：(0.0, 1.0]；0.5 表示跳过一半的 token，保留另一半
    # 值越大，保留的 token 越少，计算量越低但可能丢失信息
    uimask_ratio: float = field(default=0.5, metadata={"help": "Specify the percentage of patch tokens to skip per component for token selection"})

    # 是否使用随机 token 选择替代均匀选择
    # True 表示随机选择保留的 token，而非按均匀间隔选择
    uimask_rand: bool = field(default=False, metadata={"help": "Enable random token selection instead of uniform selection"})

    # 语言模型中需要跳过的层配置，格式为 JSON 列表字符串
    # 格式说明：[起始层, 结束层, 步长]，用于 token 选择时跳过指定层
    # 默认值 '[1,28,0]' 表示从第1层到第28层
    lm_skip_layer: str = field(default='[1,28,0]', metadata={"help": "Specify the layers of the language model to skip for token selection"})

    # 视觉模型中需要跳过的层配置，格式同上
    # 默认值 '[1,32,0]' 表示从第1层到第32层
    vis_skip_layer: str = field(default='[1,32,0]', metadata={"help": "Specify the layers of the vision model to skip for token selection"})


# ============================================================================
# DataArguments：数据相关参数
# 用途：定义训练/评测数据的来源、格式、图像处理方式以及编码输出配置。
# ============================================================================
@dataclass
class DataArguments:
    # 数据集配置文件的 YAML 路径，YAML 文件中定义了数据集的详细配置
    # 包括数据路径、采样策略、格式等
    dataset_config: str = field(default=None, metadata={"help": "yaml file with dataset configuration"})

    # 所有数据集的基础目录路径（绝对路径）
    # 若设置，将自动添加到每个数据集路径的前面，方便管理多个数据集
    data_basedir: str = field(default=None, metadata={"help": "Expect an absolute path to the base directory of all datasets. If set, it will be prepended to each dataset path"})

    # HuggingFace 数据集名称，例如 "HuggingFaceM4/COCO"
    dataset_name: str = field(default=None, metadata={"help": "huggingface dataset name"})

    # 数据集的子集名称列表，用于包含多个子集的数据集
    # 例如：["subset_a", "subset_b"]
    subset_name: List[str] = field(default=None, metadata={"help": "Useful for datasets with subsets"})

    # 使用的数据集划分，默认使用训练集 "train"
    # 其他可选值："validation"、"test" 等
    dataset_split: str = field(default='train', metadata={"help": "dataset split"})

    # 每个子集采样的训练样本数量
    # None 表示使用该子集的全部数据；设置后可控制每个子集的数据量，平衡各子集
    num_sample_per_subset: int = field(default=None, metadata={"help": "number of training samples per subset"})

    # 图像文件的目录路径，用于加载本地图像数据
    image_dir: str = field(default=None, metadata={"help": "Image directory path"})

    # 编码输出的保存路径，用于将模型编码结果写入文件
    encode_output_path: str = field(default=None, metadata={"help": "encode output path"})

    # 分词后的最大输入序列长度（包含图像 token）
    # 注意：由于图像 token 占用较多长度，设置过小可能会截断文本提示
    # None 表示不限制长度；建议根据模型和显存合理设置
    max_len: int = field(default=None, metadata={"help": "The maximum total input sequence length after tokenization. Use with caution, since it may truncate text prompts due to large image lengths."},)

    # 嵌入类型，用于指定生成嵌入向量的方式
    # 空字符串表示使用默认方式
    embedding_type: str = field(default="", metadata={"help": "embedding type"})

    # 图像分辨率设置，用于 LLaVA-next、Qwen 等需要预调整图像分辨率的模型
    # None 表示使用原始图像分辨率
    # 仅在 resize_use_processor=False 时生效
    image_resolution: str = field(default=None, metadata={"help": "for models i.e. LLaVA-next and Qwen, resize images first, none means using original image resolution. This is only works when `--resize_use_processor false`."})

    # 是否使用处理器（如 Qwen2VLImageProcessor）内部逻辑来调整图像大小
    # True 表示由处理器自动处理图像缩放；False 表示使用自定义代码处理
    resize_use_processor: bool = field(default=True, metadata={"help": "Resize visual inputs insides processor, e.g. Qwen2VLImageProcessor, instead of by our code."})

    # 图像缩放的最小像素数，基于 28×28 的 patch 大小计算
    # 默认值 28*28*4=3136 像素，确保图像不会缩得太小而丢失信息
    # 仅在 resize_use_processor=True 时生效
    resize_min_pixels: int = field(default=28*28*4, metadata={"help": "The min pixels of the image to resize the image. This is only works when `--resize_use_processor true`."})

    # 图像缩放的最大像素数，基于 28×28 的 patch 大小计算
    # 默认值 28*28*1280=1003520 像素，限制图像不会过大导致显存溢出
    # 仅在 resize_use_processor=True 时生效
    resize_max_pixels: int = field(default=28*28*1280, metadata={"help": "The max pixels of the image to resize the image. This is only works when `--resize_use_processor true`."})

    # 时序图像的衰减因子，用于逐步缩小后续帧的图像分辨率
    # None 表示不使用衰减；值越小，后续帧分辨率越低
    image_decay_factor: float = field(default=None, metadata={"help": "The image decay factor for resizing temporal images"})

    # 每个样本中硬负样本的数量
    # 0 表示不使用硬负样本；值越大，对比学习中的负样本越多，训练越困难但可能效果更好
    num_hardneg: int = field(default=0, metadata={"help": "hard negative number"})


# ============================================================================
# TrainingArguments：训练相关参数
# 用途：继承自 HuggingFace 的 TrainingArguments，扩展了图像编码器冻结、
#       梯度缓存、交错采样等 VLM2Vec 特有的训练配置。
# ============================================================================
@dataclass
class TrainingArguments(TrainingArguments):
    # 是否冻结图像编码器的参数
    # True 表示训练时不更新图像编码器的权重，仅训练其他部分
    # 适用于图像编码器已经预训练好的场景，可加速训练并减少显存占用
    image_encoder_freeze: bool = field(default=False, metadata={"help": "huggingface model name"})

    # 模型输出目录，用于保存训练后的模型检查点和相关文件
    output_dir: str = field(default=None, metadata={"help": "directory for saving trained models"})

    # 训练恢复策略：
    # "none" 表示从头开始训练
    # "auto" 表示自动检测是否有之前的检查点需要恢复
    # 也可以指定具体的检查点步数来恢复
    resume_from: str = field(default="none", metadata={"help": "`auto` will detect if any previous checkpoints should be resumed. or specify specific step of the checkpoint."})

    # 项目名称，用于在日志和实验管理工具中标识本次训练
    project_name: str = field(default=None, metadata={"help": "project name"})

    # 日志记录步数间隔，每隔多少步记录一次训练指标
    # 默认值 1 表示每步都记录，适合调试；正式训练可适当增大以减少日志量
    logging_steps: int = field(default=1, metadata={"help": "logging steps"})

    # 训练的总轮数（epoch 数），即遍历整个训练集的次数
    # 默认值 1 表示只训练一轮
    num_train_epochs: int = field(default=1, metadata={"help": "number of training epochs"})

    # 是否使用梯度缓存（Gradient Cache）策略进行训练
    # 适用于批次大小超出显存限制的场景，通过分块计算梯度来模拟大批次训练
    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})

    # 梯度缓存中查询（query）侧的分块大小
    # 取值范围：正整数；值越大，单次计算的查询越多，显存占用越高
    gc_q_chunk_size: int = field(default=2, metadata={"help": "query side subset size"})

    # 梯度缓存中目标（target/passage）侧的分块大小
    # 取值范围：正整数；值越大，单次计算的目标越多，显存占用越高
    gc_p_chunk_size: int = field(default=2, metadata={"help": "target side subset size"})

    # 交错采样策略中数据集耗尽时的处理方式：
    # "all_exhausted" —— 所有数据集都用完后才停止
    # "first_exhausted" —— 任意一个数据集用完即停止
    interleave_stopping_strategy: str = field(default="all_exhausted", metadata={"help": "all_exhausted or first_exhausted"})

    # 每个设备上来自同一数据集的连续样本数
    # 0 或 None 表示随机混合不同数据集的样本
    # 大于 0 时，会连续采样同一数据集的指定数量样本，有助于稳定训练
    homogeneous_batch_size_per_device: float = field(default=0, metadata={"help": "Specify number of consecutive samples from the same dataset PER DEVICE. 0/None means random mixing."})

    # [已弃用] 请使用 homogeneous_batch_size_per_device 替代
    interleave_batch_size: float = field(default=0, metadata={"help": "[DEPRECATED] Use `homogeneous_batch_size_per_device`."})


# ============================================================================
# MTEBArguments：MTEB 评测相关参数
# 用途：定义 MTEB（Massive Text Embedding Benchmark）评测的配置，
#       包括推理设备、批次大小、最大序列长度、评测任务等。
# ============================================================================
@dataclass
class MTEBArguments:
    # 推理使用的设备："cuda" 表示使用 GPU，"cpu" 表示使用 CPU
    # 若有多个 GPU 可用，会自动使用数据并行（DP）加速推理
    device: str = field(default="cuda", metadata={"help": "use cuda for single GPU inference, if multiple GPUs are available it will use DP automatically"})

    # 每个设备上的推理批次大小
    # 取值范围：正整数；值越大，推理速度越快，但显存占用越高
    # 默认值 16 适合大多数 GPU
    batch_size_per_device: int = field(default=16, metadata={"help": ""})

    # 推理时的最大序列长度
    # 超过此长度的文本将被截断；默认值 512 是常见的设置
    max_length: int = field(default=512, metadata={"help": ""})

    # 评测结果的输出目录，用于保存评测日志和指标
    eval_output_dir: str = field(default=None, metadata={"help": "directory for saving trained models"})

    # 要评测的任务类型列表，用于筛选特定类型的 MTEB 任务
    # 例如：["Classification", "Retrieval"] 等
    # None 表示不按类型筛选
    task_types: List[str] = field(default=None, metadata={"help": ""})

    # 要评测的具体任务名称列表
    # 例如：["MNLI", "SciDocs"] 等
    # None 表示评测所有可用任务
    tasks: List[str] = field(default=None, metadata={"help": ""})

    # 提示词族列表，用于指定评测时使用的提示词模板
    # 不同的提示词可能影响评测结果，可用于对比不同提示策略的效果
    # None 表示使用默认提示词
    prompt_family: List[str] = field(default=None, metadata={"help": ""})
