"""
处理器模块 (processor_annotated.py)
====================================

本模块是 VLM2Vec 项目中多模态数据处理的核心模块，负责为不同的视觉语言模型 (VLM) 骨干网络
加载对应的处理器 (Processor)，并将原始的文本和图像/视频输入转换为模型可接受的张量格式。

主要功能：
1. 定义并管理多种 VLM 骨干网络的常量标识、图像/视频特殊 token 映射以及模型类映射
2. 根据模型配置加载对应的 Processor（支持 Phi-3-Vision、LLaVA-NeXT、Qwen2-VL、
   Qwen2.5-VL、InternVideo2、ColPali、GME、LamRA、E5-V 等模型）
3. 为每种骨干网络提供专门的数据处理函数，将文本和视觉输入处理为模型输入字典
4. 提供统一的输入文本构造接口，根据不同模型的需求插入特殊 token

在项目中的位置：
- 被训练脚本 (train.py) 和评估脚本 (eval.py) 调用
- 位于 src/model/ 目录下，是模型数据预处理的关键环节
- 与各骨干网络的 Processor 实现配合使用
"""

import logging  # 日志记录模块

import PIL  # Python 图像处理库，用于判断图像类型
from transformers.image_utils import ChannelDimension  # 图像通道维度枚举，用于指定输入数据格式

from src.model.baseline_backbone.colpali import ColPaliProcessor  # ColPali 模型的自定义处理器

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器

import torch  # PyTorch 深度学习框架
import numpy as np  # NumPy 数值计算库
from src.utils.basic_utils import print_master  # 仅在主进程打印信息的工具函数

from src.model.baseline_backbone.llava_next import LlavaNextForConditionalGeneration  # LLaVA-NeXT 模型类
from src.model.baseline_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM  # Phi-3-Vision 模型类
from src.model.vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLProcessor  # Qwen2-VL 模型类和处理器
from src.model.vlm_backbone.qwen2_vl_tokenselection import \
    Qwen2VLForConditionalGeneration as Qwen2VLTokenSelectionForConditionalGeneration, \
    Qwen2VLProcessor as Qwen2VLTokenSelectionProcessor  # Qwen2-VL TokenSelection 变体模型类和处理器
from src.model.baseline_backbone.internvideo2.modeling_internvideo2 import InternVideo2_Stage2  # InternVideo2 第二阶段模型类
from src.model.vlm_backbone.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration  # Qwen2.5-VL 模型类
from src.model.vlm_backbone.qwen2_5_vl_tokenselection import \
    Qwen2_5_VLForConditionalGeneration as Qwen2_5_VL_TokenSelectionForConditionalGeneration  # Qwen2.5-VL TokenSelection 变体模型类


# Phi-3-Vision 中图像 token 对应的最大 input_id 值，用于标识图像占位符
PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)
# LLaVA-NeXT 中图像 token 的 ID 编号
LLAVA_IMAGE_TOKEN_ID = 32000

# ========== 骨干网络名称常量 ==========
# 以下常量定义了项目中支持的所有 VLM 骨干网络的标识名称

PHI3V = 'phi3_v'  # Microsoft Phi-3-Vision 模型
LLAVA_NEXT = 'llava_next'  # LLaVA-NeXT 模型
QWEN2_VL = 'qwen2_vl'  # Qwen2 视觉语言模型
QWEN2_VL_TOKENSELECTION = 'qwen2_vl'  # Qwen2-VL TokenSelection 变体（骨干名与 QWEN2_VL 相同）
QWEN2_5_VL = 'qwen2_5_vl'  # Qwen2.5 视觉语言模型
QWEN2_VL_TOKENSELECTION = 'qwen2_vl_tokenselection'  # Qwen2-VL TokenSelection 变体（独立标识）
QWEN2_5_VL_TOKENSELECTION = 'qwen2_5_vl_tokenselection'  # Qwen2.5-VL TokenSelection 变体
INTERNVIDEO2 = 'internvideo2'  # InternVideo2 视频理解模型
GME = 'gme'  # GME 模型，基于 QWEN2-VL 骨干
LamRA = 'lamra'  # LamRA 模型，基于 QWEN2-VL 骨干
LamRA_QWEN2_5 = 'lamra_qwen25'  # LamRA 模型的 QWEN2.5-VL 变体
COLPALI = 'colpali'  # ColPali 模型，基于 PaliGemma-3B 骨干
E5_V = 'e5_v'  # E5-V 模型，基于 LLaVA-NeXT 骨干

# 模型类型到骨干网络名称的映射字典
# 键来自 HuggingFace 配置的 model_type 字段，或手动添加的自定义标识
MODEL2BACKBONE = {
    'phi3_v': PHI3V,
    'llava_next': LLAVA_NEXT,
    'qwen2_vl': QWEN2_VL,
    'qwen2_vl_tokenselection': QWEN2_VL,  # TokenSelection 变体映射到 QWEN2_VL 骨干
    'qwen2_5_vl': QWEN2_5_VL,
    'qwen2_vl_tokenselection': QWEN2_VL_TOKENSELECTION,
    'qwen2_5_vl_tokenselection': QWEN2_5_VL_TOKENSELECTION,
    'internvideo2': INTERNVIDEO2,
    'gme': GME,
    'lamra': LamRA,
    'lamra_qwen25': LamRA,  # LamRA 的 QWEN2.5 变体映射到 LamRA 骨干
    'colpali': COLPALI,
    'e5_v': E5_V,
}

# 支持的模型类型集合，用于验证模型类型是否受支持
SUPPORTED_MODELS = set(MODEL2BACKBONE.keys())

# 各骨干网络对应的图像特殊 token 映射
# 在构造输入文本时，需要插入对应模型的图像占位符 token
VLM_IMAGE_TOKENS = {
    PHI3V: "<|image_1|>",  # Phi-3-Vision 使用编号图像 token
    LLAVA_NEXT: "<image>",  # LLaVA-NeXT 使用 <image> token
    QWEN2_VL: "<|image_pad|>",  # Qwen2-VL 使用 image_pad token
    QWEN2_5_VL: "<|image_pad|>",  # Qwen2.5-VL 使用 image_pad token
    QWEN2_VL_TOKENSELECTION: "<|image_pad|>",
    QWEN2_5_VL_TOKENSELECTION: "<|image_pad|>",
    GME: "<|image_pad|>",
    LamRA: "<|image_pad|>",
    LamRA_QWEN2_5: "<|image_pad|>",
    INTERNVIDEO2: "",  # InternVideo2 不使用图像 token（视频模型）
    COLPALI: "",  # ColPali 不使用图像 token（独立处理流程）
    E5_V: "<image>",  # E5-V 使用与 LLaVA-NeXT 相同的 <image> token
}

# 各骨干网络对应的视频特殊 token 映射
# 在构造输入文本时，需要插入对应模型的视频占位符 token
VLM_VIDEO_TOKENS = {
    LLAVA_NEXT: "<image>",  # LLaVA-NeXT 视频帧也使用 <image> token
    QWEN2_VL: "<|video_pad|>",  # Qwen2-VL 使用 video_pad token
    QWEN2_5_VL: "<|video_pad|>",  # Qwen2.5-VL 使用 video_pad token
    QWEN2_VL_TOKENSELECTION: "<|video_pad|>",
    QWEN2_5_VL_TOKENSELECTION: "<|video_pad|>",
    GME: "<|video_pad|>",
    LamRA: "<|video_pad|>",
    LamRA_QWEN2_5: "<|video_pad|>",
    INTERNVIDEO2: "",  # InternVideo2 不使用视频 token
    COLPALI: "",  # ColPali 不使用视频 token
    E5_V: "<image>",  # E5-V 视频帧也使用 <image> token
}

# 骨干网络名称到模型类的映射字典
# 用于根据骨干网络名称实例化对应的模型
backbone2model = {
    PHI3V: Phi3VForCausalLM,
    LLAVA_NEXT: LlavaNextForConditionalGeneration,
    QWEN2_VL: Qwen2VLForConditionalGeneration,
    QWEN2_5_VL: Qwen2_5_VLForConditionalGeneration,
    QWEN2_VL_TOKENSELECTION: Qwen2VLTokenSelectionForConditionalGeneration,
    QWEN2_5_VL_TOKENSELECTION: Qwen2_5_VL_TokenSelectionForConditionalGeneration,
    INTERNVIDEO2: InternVideo2_Stage2,
    E5_V: LlavaNextForConditionalGeneration,  # E5-V 复用 LLaVA-NeXT 模型类
}


def load_processor(model_args, data_args=None):
    """
    根据模型骨干网络类型加载对应的 Processor。

    不同的视觉语言模型需要使用不同的 Processor 来处理输入数据。本函数根据 model_args 中
    指定的骨干网络类型，加载对应的 Processor 实例，并根据需要配置图像处理器和分词器。

    注意：由于 transformers 库的变更（参见 GitHub commit 9215cc62），部分 Processor 的
    加载方式需要特殊处理。

    Args:
        model_args: 模型参数对象，包含以下关键属性：
            - checkpoint_path: 模型检查点路径（优先使用）
            - model_name: 模型名称或路径（checkpoint_path 为空时使用）
            - model_backbone: 骨干网络类型标识
            - num_crops: Phi-3-Vision 的裁剪数量参数
            - processor_name: 自定义处理器名称
            - uigraph_use/uigraph_diff/uigraph_rand: UIGraph 相关参数
            - uimask_ratio/uimask_rand: UIMask 相关参数
        data_args: 数据参数对象，可选，包含以下关键属性：
            - resize_min_pixels: 图像最小像素数
            - resize_max_pixels: 图像最大像素数
            - resize_use_processor: 是否使用处理器内置的 resize 逻辑

    Returns:
        processor: 加载好的 Processor 实例；InternVideo2 返回 None
    """
    # 优先使用检查点路径，否则使用模型名称
    model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
    print_master(f'Loading processor from: {model_name_or_path}')

    # ========== Phi-3-Vision 处理器加载 ==========
    if model_args.model_backbone == PHI3V:
        from src.model.baseline_backbone.phi3_v.processing_phi3_v import Phi3VProcessor
        processor = Phi3VProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            num_crops=model_args.num_crops  # 指定图像裁剪数量
        )
        processor.tokenizer.padding_side = "right"  # 设置分词器右填充（训练时常用）

    # ========== LLaVA-NeXT 处理器加载 ==========
    elif model_args.model_backbone == LLAVA_NEXT:
        from src.model.baseline_backbone.llava_next import LlavaNextProcessor
        processor = LlavaNextProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )

    # ========== Qwen2-VL / GME / LamRA 处理器加载 ==========
    # 这三种模型都基于 Qwen2-VL 骨干，共享处理器
    elif model_args.model_backbone in [QWEN2_VL, GME, LamRA]:
        from src.model.vlm_backbone.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
        from src.model.vlm_backbone.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
        # 从数据参数中获取图像尺寸限制
        min_pixels, max_pixels = None, None
        if data_args is not None:
            min_pixels, max_pixels = data_args.resize_min_pixels, data_args.resize_max_pixels
        # 构造尺寸配置字典
        size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        # 分别加载图像处理器和分词器，再组合为 Processor
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_name_or_path, size=size)
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2VLProcessor.from_pretrained(
            model_name_or_path,
            image_processor=image_processor, tokenizer=tokenizer, size=size
        )

    # ========== Qwen2-VL TokenSelection 处理器加载 ==========
    # TokenSelection 变体支持 UIGraph 和 UIMask 等高级功能
    elif model_args.model_backbone == QWEN2_VL_TOKENSELECTION:
        from src.model.vlm_backbone.qwen2_vl_tokenselection.processing_qwen2_vl import Qwen2VLProcessor
        from src.model.vlm_backbone.qwen2_vl_tokenselection.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl_tokenselection.tokenization_qwen2_fast import Qwen2TokenizerFast
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_name_or_path)
        # 根据数据参数配置图像处理器的 resize 行为
        if data_args is not None:
            image_processor.do_resize = data_args.resize_use_processor
            image_processor.min_pixels = data_args.resize_min_pixels
            image_processor.max_pixels = data_args.resize_max_pixels
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2VLProcessor.from_pretrained(
            model_name_or_path,
            image_processor=image_processor, tokenizer=tokenizer,
            uigraph_use=model_args.uigraph_use,  # 是否使用 UIGraph
            uigraph_diff=model_args.uigraph_diff,  uigraph_rand=model_args.uigraph_rand,  # UIGraph 差分和随机参数
            uimask_ratio=model_args.uimask_ratio, uimask_rand=model_args.uimask_rand  # UIMask 比例和随机参数
        )

    # ========== Qwen2.5-VL / LamRA-QWEN2.5 处理器加载 ==========
    elif model_args.model_backbone in [QWEN2_5_VL, LamRA_QWEN2_5]:
        from src.model.vlm_backbone.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        from src.model.vlm_backbone.qwen2_5_vl.image_processing_qwen2_5_vl import Qwen2_5_VLImageProcessor
        # Qwen2.5-VL 复用 Qwen2-VL 的分词器
        from src.model.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
        min_pixels, max_pixels = None, None
        if data_args is not None:
            min_pixels, max_pixels = data_args.resize_min_pixels, data_args.resize_max_pixels
        # Qwen2.5-VL 的尺寸配置额外包含 min_pixels 和 max_pixels 字段
        size = {"shortest_edge": min_pixels, "longest_edge": max_pixels, "min_pixels": min_pixels, "max_pixels": max_pixels}
        image_processor = Qwen2_5_VLImageProcessor.from_pretrained(model_name_or_path, size=size)
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2_5_VLProcessor.from_pretrained(model_name_or_path, image_processor=image_processor, tokenizer=tokenizer)

    # ========== Qwen2.5-VL TokenSelection 处理器加载 ==========
    elif model_args.model_backbone == QWEN2_5_VL_TOKENSELECTION:
        # TODO: qwen2.5 token selection not working yet  # Qwen2.5 TokenSelection 尚未完成
        from src.model.vlm_backbone.qwen2_5_vl_tokenselection.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        from src.model.vlm_backbone.qwen2_5_vl_tokenselection.image_processing_qwen2_5_vl import Qwen2_5_VLImageProcessor
        # Qwen2.5-VL TokenSelection 复用 Qwen2-VL TokenSelection 的分词器
        from src.model.vlm_backbone.qwen2_vl_tokenselection.tokenization_qwen2_fast import Qwen2TokenizerFast
        min_pixels, max_pixels = None, None
        if data_args is not None:
            min_pixels, max_pixels = data_args.resize_min_pixels, data_args.resize_max_pixels
        size = {"shortest_edge": min_pixels, "longest_edge": max_pixels, "min_pixels": min_pixels, "max_pixels": max_pixels}
        image_processor = Qwen2_5_VLImageProcessor.from_pretrained(model_name_or_path, size=size)
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2_5_VLProcessor.from_pretrained(
            model_name_or_path,
            image_processor=image_processor, tokenizer=tokenizer,
            uigraph_use=model_args.uigraph_use,  # 是否使用 UIGraph
            uigraph_diff=model_args.uigraph_diff,  uigraph_rand=model_args.uigraph_rand,  # UIGraph 差分和随机参数
            uimask_ratio=model_args.uimask_ratio, uimask_rand=model_args.uimask_rand  # UIMask 比例和随机参数
        )

    # ========== InternVideo2 处理器 ==========
    # InternVideo2 使用独立的处理流程，不需要标准的 Processor
    elif model_args.model_backbone == INTERNVIDEO2:
        return None

    # ========== ColPali 处理器加载 ==========
    elif model_args.model_backbone == COLPALI:
        from transformers import AutoProcessor
        # ColPali 使用自定义的 Processor，从原始模型名称加载
        processor = ColPaliProcessor.from_pretrained(model_args.model_name)

    # ========== 其他模型：使用 HuggingFace AutoProcessor 自动加载 ==========
    else:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,  # 优先使用自定义处理器名称
            trust_remote_code=True,
        )
    return processor


def get_backbone_name(hf_config, model_type=None):
    """
    根据 HuggingFace 配置获取骨干网络名称。

    从模型的 HuggingFace 配置对象中读取 model_type 字段，并通过 MODEL2BACKBONE 映射
    获取对应的骨干网络标识名称。如果提供了 model_type 参数，则会覆盖配置中的 model_type。

    Args:
        hf_config: HuggingFace 模型配置对象，需包含 model_type 属性
        model_type: 可选，手动指定的模型类型，会覆盖 hf_config 中的 model_type

    Returns:
        str: 骨干网络名称（如 'phi3_v', 'qwen2_vl' 等）

    Raises:
        AssertionError: 当 model_type 不在 SUPPORTED_MODELS 集合中时抛出
    """
    if model_type is not None:
        setattr(hf_config, 'model_type', model_type)  # 手动覆盖配置中的 model_type
    # 验证模型类型是否受支持
    assert hf_config.model_type in SUPPORTED_MODELS, f"Unknown backbone name {hf_config.model_type}.Supported models are {SUPPORTED_MODELS}"
    return MODEL2BACKBONE[hf_config.model_type]


def Llava_NEXT_process_fn(model_inputs: dict, processor, max_length=None):
    """
    LLaVA-NeXT 模型的输入处理函数。

    将原始的文本和图像数据处理为 LLaVA-NeXT 模型可接受的输入格式。由于 LLaVA-NeXT 的
    Processor 不支持批量处理，需要逐条处理后再进行 padding。

    注意：此函数尚未完成（NOT FINISHED YET）。

    Args:
        model_inputs: 模型输入字典，包含：
            - 'text': 文本列表，每个元素为一个样本的文本
            - 'images': 图像列表，每个元素为一个样本的图像（PIL Image 或 None）
        processor: LLaVA-NeXT 的 Processor 实例
        max_length: 可选，输入序列的最大长度

    Returns:
        dict: 处理后的模型输入字典，包含：
            - 'input_ids': token ID 张量 (LongTensor)
            - 'attention_mask': 注意力掩码张量
            - 'pixel_values': 像素值张量（有图像时）或零张量（无图像时）
            - 'image_sizes': 图像尺寸张量（有图像时）或全一张量（无图像时）
    """
    # TODO: NOT FINISHED YET!  # 此函数尚未完成
    input_ids, pixel_values, image_sizes = [], [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False  # 标记批次中是否存在图像

    # 第一步：逐条处理每个样本（因为 Processor 不支持批量处理）
    for text, images in zip(texts, visual_inputs):
        # 理论上每个批次项应包含帧列表，但仍需检查异常情况
        # 如果没有图像输入（在 MMEB 评估场景中不太可能出现）
        if images is None or (type(images)==list and any(i is None for i in images)):
            inputs = processor(images=None, text=text, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # 空字符串情况下，只包含 BOS token，squeeze 后变为标量
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
        else:
            image_exists = True
            # 有效的图像应为帧列表
            assert isinstance(images, list), f"images should be a list, but got {type(images)}"
            inputs = processor(images=images, text=text, return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            image_sizes.append(inputs['image_sizes'])

    # 第二步：对 input_ids 进行 padding
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask,
        # 'texts': texts,  # 保留注释：原始文本（调试用）
        # 'images': visual_inputs,  # 保留注释：原始图像（调试用）
    }
    # 检查批次中是否有任何样本包含图像
    image_exists = any([p is not None for p in pixel_values])
    if image_exists:
        # 将 pixel_values 从 numpy 数组转为张量，并展平批次和帧维度
        pixel_values = torch.from_numpy(np.array(pixel_values)).float()
        pixel_values_shape = pixel_values.shape
        pixel_values = pixel_values.reshape(pixel_values_shape[0] * pixel_values_shape[1], *pixel_values_shape[2:])
        # 将 image_sizes 转为张量，同样展平批次和帧维度
        image_sizes = torch.tensor(np.array(image_sizes)).long()
        image_sizes_shape = image_sizes.shape
        image_sizes = image_sizes.reshape(image_sizes_shape[0] * image_sizes_shape[1], *image_sizes_shape[2:])
        inputs['pixel_values'] = torch.from_numpy(np.array(pixel_values)).float()
        inputs['image_sizes'] = torch.tensor(np.array(image_sizes)).long()
    else:
        # 无图像时，填充占位零张量以保持输入格式一致
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs


def Phi3V_process_fn(model_inputs: dict, processor, max_length=None):
    """
    Phi-3-Vision 模型的输入处理函数。

    将原始的文本和图像数据处理为 Phi-3-Vision 模型可接受的输入格式。逐条处理每个样本，
    然后进行 padding，并处理混合批次（同时包含有图像和无图像的样本）的情况。

    Args:
        model_inputs: 模型输入字典，包含：
            - 'text': 文本列表
            - 'images': 图像列表（PIL Image 或 None）
        processor: Phi-3-Vision 的 Processor 实例
        max_length: 可选，输入序列的最大长度

    Returns:
        dict: 处理后的模型输入字典，包含：
            - 'input_ids': token ID 张量
            - 'attention_mask': 注意力掩码张量
            - 'texts': 原始文本列表
            - 'images': 原始图像列表
            - 'pixel_values': 像素值（有图像时）或零张量（无图像时）
            - 'image_sizes': 图像尺寸（有图像时）或全一张量（无图像时）
    """
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['images']
    image_exists = False  # 标记批次中是否存在图像

    # 第一步：逐条处理每个样本（Processor 不支持批量处理）
    for text, image in zip(texts, images):
        if image is None:
            # 无图像样本：仅处理文本
            inputs = processor(text, None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # 空字符串情况下，只包含 BOS token
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            image_exists = True
            # 有图像样本：同时处理文本和图像
            inputs = processor(text=text, images=[image], return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            # 可选字段：部分 Processor 可能不返回这些字段
            if 'image_sizes' in inputs:
                image_sizes.append(inputs['image_sizes'])
            if 'image_grid_thw' in inputs:
                image_grid_thw.append(inputs['image_grid_thw'])

    # 第二步：对 input_ids 进行 padding
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'texts': texts,  # 保留原始文本用于后续处理
        'images': images,  # 保留原始图像用于后续处理
    }
    # 第三步：处理混合批次（批次中同时包含有图像和无图像的样本）
    if image_exists:
        # 有图像时，直接传递列表（后续在模型中处理）
        inputs['pixel_values'] = pixel_values
        inputs['image_sizes'] = image_sizes
    else:
        # 无图像时，填充占位零张量以保持输入格式一致
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs


def Qwen2_VL_process_fn(model_inputs: dict, processor: Qwen2VLProcessor, max_length=None):
    """
    Qwen2-VL 模型的输入处理函数。

    将原始的文本和图像/视频数据处理为 Qwen2-VL 模型可接受的输入格式。支持图像和视频
    两种视觉输入，根据文本中的特殊 token 判断输入类型。由于 Processor 不支持混合批次
    处理（同时包含有视觉输入和无视觉输入的数据），需要逐条处理。

    Args:
        model_inputs: 模型输入字典，包含：
            - 'text': 文本列表，文本中包含图像/视频特殊 token
            - 'images': 视觉输入列表，可以是 PIL Image、图像列表或视频帧列表
        processor: Qwen2-VL 的 Processor 实例
        max_length: 可选，输入序列的最大长度（目前仅应用于纯文本数据）

    Returns:
        dict: 处理后的模型输入字典，包含：
            - 'input_ids': token ID 张量 (LongTensor)
            - 'attention_mask': 注意力掩码张量 (LongTensor)
            - 'texts': 原始文本列表
            - 'images': 原始视觉输入列表
            - 'pixel_values': 图像像素值列表（有图像时为 numpy 数组，无图像时为 None）
            - 'image_grid_thw': 图像网格时间-高度-宽度信息列表
            - 'pixel_values_videos': 视频像素值列表（有视频时为 numpy 数组，无视频时为 None）
            - 'video_grid_thw': 视频网格时间-高度-宽度信息列表
    """
    # TODO: set separate max_len for text/visual inputs, currently max_length is only applied to text-only data
    # TODO: 为文本/视觉输入设置独立的最大长度，目前 max_length 仅应用于纯文本数据
    input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = [], [], [], [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    # 获取 Qwen2-VL 的图像和视频特殊 token
    vlm_image_token, vlm_video_token = VLM_IMAGE_TOKENS[QWEN2_VL], VLM_VIDEO_TOKENS[QWEN2_VL]

    # 第一步：逐条处理每个样本，因为 Processor 不支持混合批次处理
    for text, visual_input in zip(texts, visual_inputs):
        # 检查是否为纯文本输入（所有图像必须有效）
        if not visual_input or (type(visual_input)==list and any(i is None for i in visual_input)):
            # 纯文本输入处理
            inputs = processor(text=[text], images=None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # 空字符串情况下，只包含 BOS token
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_grid_thw.append(None)
            pixel_values_videos.append(None)
            video_grid_thw.append(None)
        else:
            # 包含视觉输入的处理
            try:
                if vlm_image_token in text:
                    # 图像输入处理
                    if isinstance(visual_input, PIL.Image.Image):
                        # 单张图像：转为列表以统一处理
                        visual_input = [visual_input]
                    for iid, image in enumerate(visual_input):
                        # MMEB 评估中的罕见情况：如果宽或高小于 28 像素，则缩放到 56x56
                        if image.size[0] < 28 or image.size[1] < 28:
                            image = image.resize((56, 56))
                            visual_input[iid] = image
                    inputs = processor(text=[text], images=visual_input, return_tensors="np", max_length=max_length, truncation=(max_length is not None), input_data_format=ChannelDimension.LAST)
                elif vlm_video_token in text:
                    # TODO: check text/video data validity  # TODO: 检查文本/视频数据有效性
                    # 视频输入处理
                    inputs = processor(text=[text], videos=[visual_input], return_tensors="np", max_length=max_length, truncation=(max_length is not None), input_data_format=ChannelDimension.LAST)
                else:
                    # 文本中未找到视觉 token，无法确定输入类型
                    raise NotImplementedError(f"No visual token found ({vlm_image_token} or {vlm_video_token}) in the text: {text}")
            except Exception as e:
                # 打印出错的视觉输入文件名以便调试
                for i in visual_input:
                    print(i.filename)
                raise e
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            # 根据输入类型分别存储图像或视频的像素值和网格信息
            if 'pixel_values' in inputs:
                # 图像输入：存储像素值和图像网格信息
                pixel_values.append(inputs['pixel_values'])
                image_grid_thw.append(inputs['image_grid_thw'])
                pixel_values_videos.append(None)
                video_grid_thw.append(None)
            else:
                # 视频输入：存储视频像素值和视频网格信息
                pixel_values.append(None)
                image_grid_thw.append(None)
                pixel_values_videos.append(inputs['pixel_values_videos'])
                video_grid_thw.append(inputs['video_grid_thw'])

    # 第二步：对 input_ids 进行 padding
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    # 手动强制转换为 long 类型，原因如下：
    # (1) RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int;
    #     but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
    # (2) IndexError: tensors used as indices must be long, int, byte or bool tensors
    #     在 _pooling 函数中使用 input_ids 作为索引时需要 long 类型
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long(),
        'texts': texts,  # 保留原始文本
        'images': visual_inputs,  # 保留原始视觉输入
    }
    # 存储图像和视频的像素值及网格信息（列表形式，包含 None 值）
    inputs['pixel_values'] = pixel_values
    inputs['image_grid_thw'] = image_grid_thw
    inputs['pixel_values_videos'] = pixel_values_videos
    inputs['video_grid_thw'] = video_grid_thw

    return inputs


def Gme_process_fn(model_inputs: dict, processor: Qwen2VLProcessor, max_length=None):
    """
    GME 模型的输入处理函数。

    GME 模型不需要通过 Processor 进行数据预处理，而是直接传递原始文本和图像，
    在模型内部自行处理。此函数仅提取并返回原始输入。

    LamRA 和 LamRA_QWEN2_5 模型也复用此处理函数。

    Args:
        model_inputs: 模型输入字典，包含：
            - 'text': 文本列表
            - 'images': 图像列表
        processor: 未使用（GME 模型内部自行处理）
        max_length: 未使用

    Returns:
        dict: 仅包含原始文本和图像的字典：
            - 'texts': 原始文本列表
            - 'images': 原始图像列表
    """
    inputs = {
        'texts': model_inputs['text'],
        'images': model_inputs['images'],
    }
    return inputs


def Qwen2_VL_TokenSelection_process_fn(model_inputs: dict, processor: Qwen2VLTokenSelectionProcessor, max_length=None):
    """
    Qwen2-VL TokenSelection 变体的输入处理函数。

    与标准 Qwen2-VL 处理函数类似，但额外支持 TokenSelection 相关的 patch_pos 和
    select_mask 字段。TokenSelection 机制允许模型选择性地关注特定的图像 patch，
    从而提高计算效率。同时支持 UIGraph 和 UIMask 等高级功能。

    Args:
        model_inputs: 模型输入字典，包含：
            - 'text': 文本列表，文本中包含图像/视频特殊 token
            - 'images': 视觉输入列表
        processor: Qwen2-VL TokenSelection 的 Processor 实例
        max_length: 可选，输入序列的最大长度（目前仅应用于纯文本数据）

    Returns:
        dict: 处理后的模型输入字典，包含：
            - 'input_ids': token ID 张量 (LongTensor)
            - 'attention_mask': 注意力掩码张量 (LongTensor)
            - 'pixel_values': 图像像素值列表
            - 'image_grid_thw': 图像网格信息列表
            - 'pixel_values_videos': 视频像素值列表
            - 'video_grid_thw': 视频网格信息列表
            - 'patch_pos': patch 位置信息张量（用于 TokenSelection）
            - 'select_mask': 选择掩码张量（用于 TokenSelection）
    """
    # TODO: set separate max_len for text/visual inputs, currently max_length is only applied to text-only data
    # TODO: 为文本/视觉输入设置独立的最大长度，目前 max_length 仅应用于纯文本数据
    input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = [], [], [], [], []
    patch_pos, select_mask = [], []  # TokenSelection 专用字段
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False  # 标记批次中是否存在图像

    # 第一步：逐条处理每个样本（Processor 不支持批量处理）
    for text, images in zip(texts, visual_inputs):
        if images is None or (type(images)==list and any(i is None for i in images)):
            # 纯文本输入：所有图像必须有效
            inputs = processor(text=[text], images=None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # 空字符串情况下，只包含 BOS token
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_grid_thw.append(None)
            patch_pos.append(None)  # 无图像时 patch_pos 为 None
            select_mask.append(None)  # 无图像时 select_mask 为 None
            pixel_values_videos.append(None)
            video_grid_thw.append(None)
        else:
            image_exists = True
            # TODO only  # 仅 TODO 标记
            # 处理来自视频的多图像数据，无法处理混合图像+视频数据
            if VLM_IMAGE_TOKENS[QWEN2_VL] in text:
                # 图像输入处理
                inputs = processor(text=[text], images=[images], return_tensors="np", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
            elif VLM_VIDEO_TOKENS[QWEN2_VL] in text:
                # 视频输入处理：必须包含多于 1 帧
                assert len(images) > 1, f"Video data must have more than 1 frame, got {len(images)}"
                inputs = processor(text=[text], videos=[images], return_tensors="np", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
            else:
                raise NotImplementedError(f"Unsupported visual token in text: {text}")
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            # 根据输入类型分别存储图像或视频的像素值和网格信息
            if 'pixel_values' in inputs:
                pixel_values.append(inputs['pixel_values'])
                image_grid_thw.append(inputs['image_grid_thw'])
                pixel_values_videos.append(None)
                video_grid_thw.append(None)
                # 提取 TokenSelection 专用字段
                if 'patch_pos' in inputs:
                    patch_pos.append(inputs['patch_pos'])
                if 'select_mask' in inputs:
                    select_mask.append(inputs['select_mask'])
            else:
                pixel_values.append(None)
                image_grid_thw.append(None)
                patch_pos.append(None)
                select_mask.append(None)
                pixel_values_videos.append(inputs['pixel_values_videos'])
                video_grid_thw.append(inputs['video_grid_thw'])

    # 第二步：对 input_ids 进行 padding
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']

    # 对 patch_pos 和 select_mask 进行 padding 对齐
    if image_exists:
        if patch_pos:
            # 获取非 None 的 patch_pos 的形状，用于创建占位张量
            patch_pos_shape_for_padding = list(v.shape for v in patch_pos if v is not None)[0]
            # 将 None 替换为 -1 填充的占位张量（-1 表示无效位置）
            key_tmp = [torch.from_numpy(v) if v is not None else (torch.zeros(patch_pos_shape_for_padding) - 1) for v in patch_pos]
            max_length = input_ids.size(1)  # 获取 padding 后的最大序列长度
            # 对每个样本的 patch_pos 进行右填充，填充值为 -1
            padded_key = [torch.nn.functional.pad(pos, (0, max_length - pos.size(1)), value=-1) for pos in key_tmp]
            patch_pos = torch.cat(padded_key, dim=0)  # 拼接为批次张量
        if select_mask:
            # 获取非 None 的 select_mask 的形状，用于创建占位张量
            select_mask_shape_for_padding = list(v.shape for v in select_mask if v is not None)[0]
            # 将 None 替换为全 True 的占位张量（True 表示不进行 token 选择，保留所有 token）
            key_tmp = [torch.from_numpy(v) if v is not None else torch.ones(select_mask_shape_for_padding).bool() for v in select_mask]
            max_length = input_ids.size(1)
            # 对每个样本的 select_mask 进行右填充，填充值为 True
            padded_key = [torch.nn.functional.pad(pos, (0, max_length - pos.size(1)), value=True) for pos in key_tmp]
            select_mask = torch.cat(padded_key, dim=0)  # 拼接为批次张量

    # 手动强制转换为 long 类型，原因同 Qwen2_VL_process_fn
    # (1) RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int;
    #     but got torch.cuda.FloatTensor instead
    # (2) IndexError: tensors used as indices must be long, int, byte or bool tensors
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long()
    }
    # 存储所有视觉相关字段（列表形式，包含 None 值）
    inputs['pixel_values'] = pixel_values
    inputs['image_grid_thw'] = image_grid_thw
    inputs['pixel_values_videos'] = pixel_values_videos
    inputs['video_grid_thw'] = video_grid_thw
    inputs['patch_pos'] = patch_pos  # TokenSelection 的 patch 位置信息
    inputs['select_mask'] = select_mask  # TokenSelection 的选择掩码

    return inputs


def InternVL_process_fn(model_inputs: dict, processor, max_length=None):
    """
    InternVL 模型的输入处理函数。

    将原始的文本和图像数据处理为 InternVL 模型可接受的输入格式。结构与 Phi3V_process_fn
    类似，逐条处理后进行 padding。

    注意：此函数尚未完成（not working yet）。

    Args:
        model_inputs: 模型输入字典，包含：
            - 'text': 文本列表
            - 'images': 图像列表（PIL Image 或 None）
        processor: InternVL 的 Processor 实例
        max_length: 可选，输入序列的最大长度

    Returns:
        dict: 处理后的模型输入字典，包含：
            - 'input_ids': token ID 张量
            - 'attention_mask': 注意力掩码张量
            - 'texts': 原始文本列表
            - 'images': 原始图像列表
            - 'pixel_values': 像素值（有图像时）或零张量（无图像时）
            - 'image_sizes': 图像尺寸（有图像时）或全一张量（无图像时）
    """
    # TODO not working yet  # 此函数尚未完成
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['images']
    image_exists = False  # 标记批次中是否存在图像

    # 第一步：逐条处理每个样本（Processor 不支持批量处理）
    for text, image in zip(texts, images):
        if image is None:
            # 无图像样本：仅处理文本
            inputs = processor(text, None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # 空字符串情况下，只包含 BOS token
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            image_exists = True
            # 有图像样本：同时处理文本和图像
            inputs = processor(text=text, images=[image], return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            # 可选字段：部分 Processor 可能不返回这些字段
            if 'image_sizes' in inputs:
                image_sizes.append(inputs['image_sizes'])
            if 'image_grid_thw' in inputs:
                image_grid_thw.append(inputs['image_grid_thw'])

    # 第二步：对 input_ids 进行 padding
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'texts': texts,  # 保留原始文本
        'images': images,  # 保留原始图像
    }
    # 第三步：处理混合批次（批次中同时包含有图像和无图像的样本）
    if image_exists:
        # 有图像时，直接传递列表
        inputs['pixel_values'] = pixel_values
        inputs['image_sizes'] = image_sizes
    else:
        # 无图像时，填充占位零张量以保持输入格式一致
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs


def ColPali_process_fn(model_inputs: dict, processor, max_length=None):
    """
    ColPali 模型的输入处理函数。

    ColPali 使用特殊的处理流程：图像和查询（文本）分别通过不同的方法处理。
    图像通过 process_images 方法处理，查询通过 process_queries 方法处理。
    处理后需要对 input_ids 和 attention_mask 进行 padding，并对 pixel_values
    进行特殊处理以应对混合批次。

    Args:
        model_inputs: 模型输入字典，包含：
            - 'text': 文本列表
            - 'images': 图像列表（PIL Image 或 None）
        processor: ColPali 的 Processor 实例
        max_length: 未使用

    Returns:
        dict: 处理后的模型输入字典，包含：
            - 'input_ids': padding 后的 token ID 张量
            - 'attention_mask': padding 后的注意力掩码张量
            - 'pixel_values': 像素值张量（有图像时为实际值，无图像时为零张量占位）
    """
    texts, images = model_inputs['text'], model_inputs['images']

    input_ids_batch = []  # 存储每个样本的 input_ids
    attention_mask_batch = []  # 存储每个样本的 attention_mask
    pixel_values_batch = []  # 存储每个样本的 pixel_values

    for text, image in zip(texts, images):
        if image is not None:
            # 有图像：使用 process_images 处理图像输入
            inputs = processor.process_images([image])
            pixel_values_batch.append(inputs['pixel_values'])
        else:
            # 无图像：使用 process_queries 处理文本查询
            inputs = processor.process_queries([text])
            pixel_values_batch.append(None)

        input_ids_batch.append(inputs['input_ids'].squeeze().tolist())
        attention_mask_batch.append(inputs['attention_mask'].squeeze().tolist())

    # 对 input_ids 和 attention_mask 进行 padding
    padded_text_inputs = processor.tokenizer.pad(
        {'input_ids': input_ids_batch, 'attention_mask': attention_mask_batch},
        return_tensors="pt"
    )

    final_input_ids = padded_text_inputs['input_ids']
    final_attention_mask = padded_text_inputs['attention_mask']

    # 处理 pixel_values：需要处理混合批次中部分样本无图像的情况
    if any(pv is not None for pv in pixel_values_batch):
        # 找到一个有效的 pixel_values 形状作为参考
        representative_pv_shape = None
        for pv in pixel_values_batch:
            if pv is not None:
                representative_pv_shape = pv.shape
                break

        processed_pixel_values = []
        for pv in pixel_values_batch:
            if pv is None:
                # 无图像样本：创建与参考形状相同的零张量作为占位
                processed_pixel_values.append(torch.zeros(representative_pv_shape))
            else:
                processed_pixel_values.append(pv)
        # 将所有 pixel_values 在批次维度上拼接
        final_pixel_values = torch.cat(processed_pixel_values)
    else:
        # 批次中完全没有图像：创建默认形状的零张量
        batch_size = len(texts)
        # SigLIP 期望 3 通道 (RGB) 和 448x448 的正方形图像
        # 1024 patches = 32x32 patches，patch_size=14，32*14=448
        default_channels = 3
        default_height = 448
        default_width = 448
        final_pixel_values = torch.zeros(batch_size, default_channels, default_height, default_width)

    return {
        'input_ids': final_input_ids,
        'attention_mask': final_attention_mask,
        'pixel_values': final_pixel_values,
    }


def InternVideo2_process_fn(model_inputs: dict, processor, max_length=None):
    """
    InternVideo2 模型的输入处理函数。

    InternVideo2 使用完全独立的处理流程，不依赖标准的 Processor。文本侧使用
    BERT 分词器处理，视频侧使用 torchvision 的 transforms 进行图像预处理。
    视频输入需要固定为 4 帧，不足时重复最后一帧，超出时均匀采样。

    Args:
        model_inputs: 模型输入字典，包含：
            - 'text': 文本列表（文本侧输入）
            - 'images': 图像/视频帧列表（视频侧输入，None 表示纯文本）
        processor: 未使用（InternVideo2 使用独立的处理流程）
        max_length: 未使用

    Returns:
        dict: 处理后的模型输入字典：
            - 文本侧：包含 'input_ids' 和 'attention_mask'
            - 视频侧：包含 'pixel_values'，形状为 (B, num_frames, C, H, W)
    """
    if all(x is None for x in model_inputs["images"]):
        # 文本侧处理：使用 BERT 分词器
        from src.model.baseline_backbone.internvideo2.modeling_internvideo2 import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        inputs = tokenizer(
            model_inputs["text"],
            padding="max_length",  # 填充到最大长度
            truncation=True,  # 超长截断
            max_length=40,  # 最大长度为 40 个 token
            return_tensors="pt")
    else:
        # 视频侧处理：使用 torchvision transforms 进行图像预处理
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),  # 确保图像为 RGB 模式
            transforms.Resize((224, 224)),  # 缩放到 224x224
            transforms.ToTensor(),  # 将 PIL 图像转为张量 (C, H, W)
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 均值归一化
                                 std=[0.229, 0.224, 0.225])  # ImageNet 标准差归一化
        ])
        frame_list = model_inputs["images"]
        # 确保图像输入恰好为 4 帧
        # 情况 1：frame_list 是扁平结构（非列表的列表），例如 [PIL, PIL, ...]
        # 将每张图像复制 4 份作为 4 帧
        if type(frame_list[0]) is not list:
            frame_list = [[img.copy() for _ in range(4)] for img in frame_list]
        # 情况 2：frame_list 已经是列表的列表，确保每个子列表恰好有 4 帧
        elif type(frame_list[0]) is list and len(frame_list[0]) != 4:
            new_list = []
            for frames in frame_list:
                if len(frames) < 4:
                    # 帧数不足：重复最后一帧补齐到 4 帧
                    frames = frames + [frames[-1].copy() for _ in range(4 - len(frames))]
                elif len(frames) > 4:
                    # 帧数超出：在序列中均匀采样 4 个索引
                    indices = np.linspace(0, len(frames) - 1, num=4, dtype=int)
                    frames = [frames[i] for i in indices]
                new_list.append(frames)
            frame_list = new_list
        # 对每帧图像进行预处理，然后堆叠为 (num_frames, C, H, W)
        pixel_values = [
            torch.stack([preprocess(img) for img in frames], dim=0)
            for frames in frame_list
        ]

        # 在批次维度上堆叠，形状为 (B, num_frames, C, H, W)
        pixel_values = torch.stack(pixel_values, dim=0)
        inputs = {'pixel_values': pixel_values}

    return inputs


def e5_v_prompt_template(text, add_video_token, add_image_token):
    """
    E5-V 模型的提示模板函数。

    根据 E5-V 模型的要求，使用 LLaMA-3 的对话模板格式构造输入提示。
    根据输入类型（纯文本、纯视频、纯图像、视频+文本、图像+文本）生成不同的提示。

    Args:
        text: 文本内容，可为 None（纯视觉输入时）
        add_video_token: 是否添加视频 token
        add_image_token: 是否添加图像 token

    Returns:
        str: 构造好的提示字符串，遵循 LLaMA-3 对话模板格式
    """
    # LLaMA-3 的用户-助手对话模板
    llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
    if text is not None and add_video_token is False and add_image_token is False:  # 仅文本输入
        prompt = llama3_template.format('{}\nSummary above sentence in one word: '.format(text))
    if text is None and add_video_token:  # 仅视频输入
        prompt = llama3_template.format('<image>\nSummary above video in one word: ')
    if text is None and add_image_token:  # 仅图像输入
        prompt = llama3_template.format('<image>\nSummary above image in one word: ')
    if text is not None and add_video_token:  # 视频+文本输入
        prompt = llama3_template.format('<image>\n{}\nSummary above video and text in one word: '.format(text))
    if text is not None and add_image_token:  # 图像+文本输入
        prompt = llama3_template.format('<image>\n{}\nSummary above image and text in one word: '.format(text))

    return prompt


# 提示模板字典：模型名称到提示模板函数的映射
PROMPT_TEMPLATE_DICT = {
    "e5_v": e5_v_prompt_template,  # E5-V 模型使用专用的 LLaMA-3 模板
}


def process_input_text(instruction, model_backbone, text=None, add_video_token=False, add_image_token=False):
    """
    根据模型骨干网络类型构造输入文本。

    不同的模型对输入文本的格式要求不同：有些需要插入特殊的图像/视频 token，
    有些使用特定的提示模板，有些则直接拼接指令和文本。本函数根据模型类型
    自动选择合适的文本构造方式。

    Args:
        instruction: 指令文本，通常是任务描述（如 "Represent the image"）
        model_backbone: 骨干网络名称（如 'qwen2_vl', 'internvideo2' 等）
        text: 可选，附加的文本内容（如查询文本）
        add_video_token: 是否在文本中添加视频特殊 token
        add_image_token: 是否在文本中添加图像特殊 token

    Returns:
        str: 构造好的输入文本字符串
    """
    # 根据模型类型选择不同的文本构造方式
    # TBD: Reorganize the hard-code part for baselines such as internvideo2
    # 待办：重新组织 InternVideo2 等基线模型的硬编码部分
    if model_backbone == "internvideo2":
        # InternVideo2 直接返回原始文本，不添加任何特殊 token
        return text
    elif model_backbone in [GME, LamRA, LamRA_QWEN2_5]:
        # GME 和 LamRA 不需要特殊 token，直接拼接指令和文本
        if text:
            return instruction + " " + text
        else:
            return instruction + " "
    elif model_backbone == E5_V:
        # E5-V 使用专用的提示模板
        return PROMPT_TEMPLATE_DICT[model_backbone](text, add_video_token, add_image_token)

    # 通用处理流程：拼接指令、文本和特殊 token
    prompt = instruction
    if text:
        prompt = prompt + " " + text  # 拼接指令和文本
    if add_video_token:
        video_token = VLM_VIDEO_TOKENS[model_backbone]  # 获取对应模型的视频 token
        prompt = video_token + " " + prompt  # 在文本前添加视频 token
    if add_image_token:
        image_token = VLM_IMAGE_TOKENS[model_backbone]  # 获取对应模型的图像 token
        prompt = image_token + " " + prompt  # 在文本前添加图像 token

    return prompt


# 骨干网络到数据处理函数的映射字典
# 根据骨干网络名称选择对应的输入处理函数
process_vlm_inputs_fns = {
    PHI3V: Phi3V_process_fn,  # Phi-3-Vision 处理函数
    LLAVA_NEXT: Llava_NEXT_process_fn,  # LLaVA-NeXT 处理函数
    QWEN2_VL: Qwen2_VL_process_fn,  # Qwen2-VL 处理函数
    QWEN2_5_VL: Qwen2_VL_process_fn,  # Qwen2.5-VL 复用 Qwen2-VL 的处理函数
    QWEN2_VL_TOKENSELECTION: Qwen2_VL_TokenSelection_process_fn,  # Qwen2-VL TokenSelection 处理函数
    QWEN2_5_VL_TOKENSELECTION: Qwen2_VL_TokenSelection_process_fn,  # Qwen2.5-VL TokenSelection 复用 Qwen2-VL TokenSelection 的处理函数
    INTERNVIDEO2: InternVideo2_process_fn,  # InternVideo2 处理函数
    GME: Gme_process_fn,  # GME 处理函数（直接传递原始输入）
    LamRA: Gme_process_fn,  # LamRA 复用 GME 处理函数
    LamRA_QWEN2_5: Gme_process_fn,  # LamRA-QWEN2.5 复用 GME 处理函数
    COLPALI: ColPali_process_fn,  # ColPali 处理函数
    E5_V: Llava_NEXT_process_fn,  # E5-V 复用 LLaVA-NeXT 处理函数
}
