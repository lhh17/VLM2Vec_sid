# ==============================================================================
# 模块概述：model.py —— VLM2Vec 项目的多模态嵌入模型定义
# ==============================================================================
# 本文件定义了 MMEBModel（Multimodal Embedding Model），是 VLM2Vec 项目的核心模型类。
# MMEBModel 基于视觉语言模型（VLM）构建，通过对比学习训练，使模型能够生成
# 高质量的文本-图像/视频联合嵌入表示，用于多模态检索和排序任务。
#
# 主要功能：
#   1. 支持多种视觉语言模型骨干（Qwen2-VL、LLaVA-Next、Phi-3V、InternVideo2 等）
#   2. 提供统一的编码接口（encode_input），对不同骨干的输入/输出进行适配
#   3. 实现对比学习训练的 forward 逻辑（InfoNCE 损失），支持 DDP 分布式训练
#   4. 支持 LoRA 参数高效微调，以及 ColPali、GME、LamRA 等基线模型
#   5. 提供模型构建（build）和加载（load）的工厂方法，根据配置自动选择骨干和加载方式
#
# 在项目中的位置：
#   本文件位于 src/model/ 目录下，是模型层的核心定义。
#   它被训练脚本 train.py 和评估脚本 eval.py 直接引用，
#   依赖 src/model/processor.py 进行骨干识别和处理器加载，
#   依赖 src/model/baseline_backbone/ 下的子模块支持各种基线模型。
# ==============================================================================

from typing import Dict  # 类型提示：字典类型，用于 forward 方法中 qry/tgt 参数的类型标注
import torch  # PyTorch 深度学习框架，提供张量运算、GPU 加速和自动求导
import torch.distributed as dist  # PyTorch 分布式通信模块，用于多 GPU 间的数据同步和聚合
from torch import nn, Tensor  # nn: 神经网络模块基类；Tensor: 张量类型，用于类型标注
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig  # PreTrainedModel: HuggingFace 预训练模型基类；AutoModelForCausalLM: 自动加载因果语言模型；AutoConfig: 自动加载模型配置
from peft import LoraConfig, get_peft_model, PeftModel  # LoraConfig: LoRA 配置类；get_peft_model: 将基础模型包装为 LoRA 模型；PeftModel: LoRA 模型类，用于加载已训练的 LoRA 权重
from src.model.processor import QWEN2_5_VL_TOKENSELECTION  # Qwen2.5-VL TokenSelection 骨干标识常量（支持层跳过的 Qwen2.5-VL 变体）
from src.arguments import ModelArguments, TrainingArguments  # ModelArguments: 模型相关参数数据类；TrainingArguments: 训练相关参数数据类
from src.model.processor import LLAVA_NEXT, QWEN2_VL, PHI3V, get_backbone_name, print_master, QWEN2_5_VL, \
    backbone2model, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION, E5_V  # 各骨干标识常量、骨干名称获取函数、主进程打印函数、骨干到模型类的映射字典

from src.arguments import ModelArguments  # 重复导入 ModelArguments（原代码如此，保留不改动）
from src.model.processor import LLAVA_NEXT, QWEN2_VL, PHI3V, get_backbone_name, print_master, QWEN2_5_VL, INTERNVIDEO2, \
    QWEN2_VL_TOKENSELECTION, backbone2model, GME, VLM_IMAGE_TOKENS, LamRA, LamRA_QWEN2_5, COLPALI  # INTERNVIDEO2/GME/LamRA/COLPALI: 额外的骨干标识常量；VLM_IMAGE_TOKENS: 视觉语言模型图像占位符映射
from src.model.baseline_backbone.colpali import ColPali  # ColPali 基线模型：基于 late-interaction 机制的多模态嵌入模型
from src.model.baseline_backbone.gme.gme_inference import GmeQwen2VL  # GME 基线模型：基于 Qwen2-VL 的通用多模态嵌入模型
from src.model.baseline_backbone.lamra.lamra_inference import LamRAQwen2VL  # LamRA 基线模型：基于 Qwen2-VL 的多模态检索模型
from src.model.baseline_backbone.lamra.lamra_qwen25_inference import LamRAQwen25VL  # LamRA-Qwen2.5 基线模型：基于 Qwen2.5-VL 的多模态检索模型
from src.model.baseline_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM  # Phi-3-Vision 因果语言模型：微软的视觉语言模型
from src.model.baseline_backbone.llava_next import LlavaNextForConditionalGeneration  # LLaVA-Next 条件生成模型：改进的视觉语言模型

from transformers import modeling_utils  # HuggingFace 建模工具模块，包含模型并行等配置
# 修复 transformers 的并行样式配置：如果 ALL_PARALLEL_STYLES 不存在或为 None，
# 则设置默认值，避免模型加载时因并行配置缺失而报错
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", 'rowwise']


class MMEBModel(nn.Module):
    """
    多模态嵌入模型（Multimodal Embedding Model）。

    基于视觉语言模型（VLM）构建的嵌入模型，通过对比学习训练，使模型能够将
    文本和图像/视频编码到同一嵌入空间中，用于多模态检索和排序任务。

    核心属性：
        encoder: 编码器模型（基础 VLM 或 LoRA 包装后的 VLM），负责将输入编码为隐藏状态
        pooling: 池化策略，决定如何从隐藏状态中提取句级嵌入（默认 'last'：取最后一个 token 的表示）
        normalize: 是否对嵌入进行 L2 归一化
        temperature: 对比学习中的温度参数，控制相似度得分的缩放
        cross_entropy: 交叉熵损失函数，用于计算 InfoNCE 对比损失
        is_ddp: 是否处于分布式数据并行（DDP）环境中
        process_rank: 当前进程在分布式组中的编号（仅 DDP 时有效）
        world_size: 分布式训练的总进程数（仅 DDP 时有效）
        model_backbone: 模型骨干名称，标识当前使用的 VLM 类型（在 load 方法中动态设置）

    类属性：
        TRANSFORMER_CLS: 默认的 Transformer 模型类，用于加载未特殊处理的骨干模型
    """
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'last',
                 normalize: bool = False,
                 temperature: float = 0.02,
                 ):
        """
        初始化多模态嵌入模型。

        参数：
            encoder: 预训练的编码器模型（基础 VLM 或 LoRA 包装后的模型）
            pooling: 池化策略，'last' 或 'eos' 表示取最后一个有效 token 的表示
            normalize: 是否对池化后的嵌入进行 L2 归一化
            temperature: 对比学习温度参数，值越小正负样本区分度越高
        """
        super().__init__()
        self.config = encoder.config  # 保存编码器的配置对象，包含模型结构信息
        self.encoder = encoder  # 编码器模型，负责将输入编码为隐藏状态
        self.pooling = pooling  # 池化策略：'last'/'eos' 取最后有效 token
        self.normalize = normalize  # 是否对嵌入进行 L2 归一化
        self.temperature = temperature  # 对比学习温度参数
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')  # 交叉熵损失，用于 InfoNCE 对比损失计算
        self.is_ddp = dist.is_initialized()  # 检测是否处于 DDP 分布式训练环境中
        if self.is_ddp:
            self.process_rank = dist.get_rank()  # 当前进程的编号（0, 1, 2, ...）
            self.world_size = dist.get_world_size()  # 分布式训练的总进程数

    @property
    def device(self):
        """
        获取模型参数所在的设备。

        通过遍历模型参数获取设备信息。如果模型没有参数（如空模型），
        则返回 CPU 设备。

        返回值：
            torch.device: 模型参数所在的设备
        """
        try:
            return next(self.parameters()).device  # 从第一个参数获取设备信息
        except StopIteration:
            return torch.device("cpu")  # 模型无参数时默认返回 CPU

    def encode_input(self, input):
        """
        将输入编码为嵌入向量。根据不同的模型骨干类型，采用不同的编码逻辑。

        支持的骨干类型及编码方式：
            - INTERNVIDEO2: 分别调用文本编码器和视觉编码器，通过投影层映射到统一空间
            - GME/LamRA/LamRA_QWEN2_5: 调用模型的 get_fused_embeddings 方法，融合文本和图像嵌入
            - COLPALI: 直接前向传播，返回完整的隐藏状态（用于 late-interaction 评分）
            - LLAVA_NEXT: 前向传播后取最后一层隐藏状态，再通过池化得到句级嵌入
            - 其他（Qwen2-VL 等）: 前向传播后取最后一层隐藏状态，再通过池化得到句级嵌入

        参数：
            input: dict，模型输入字典，包含 input_ids、attention_mask、pixel_values 等键，
                   具体内容因骨干类型而异

        返回值：
            Tensor 或对象：编码后的嵌入表示
                - 稠密模型: 形状为 [batch_size, hidden_dim] 的张量
                - Late-interaction 模型（ColPali）: 包含完整隐藏状态的对象
        """
        if getattr(self, "model_backbone", None) == INTERNVIDEO2:
            # InternVideo2 骨干：文本和视觉使用独立的编码器和投影层
            if "input_ids" in input.keys():
                # 文本侧编码：使用文本编码器提取文本特征
                text_output = self.encoder.get_text_encoder()(
                    input["input_ids"],
                    attention_mask=input["attention_mask"],
                    return_dict=True,
                    mode="text",
                )
                text_embeds = text_output.last_hidden_state  # 获取最后一层隐藏状态
                pooled_text_embeds = text_embeds[:, 0]  # 取 [CLS] token 的表示作为池化结果
                pooled_output = self.encoder.text_proj(pooled_text_embeds)  # 通过文本投影层映射到统一嵌入空间
                pooled_output /= pooled_output.norm(dim=-1, keepdim=True)  # L2 归一化
                return pooled_output
            else:
                # 视觉侧编码：使用视觉编码器提取视觉特征
                _, vfeat = self.encoder.encode_vision(input["pixel_values"], test=True)  # 编码视觉输入，test=True 表示推理模式
                vfeat = self.encoder.vision_proj(vfeat)  # 通过视觉投影层映射到统一嵌入空间
                vfeat /= vfeat.norm(dim=-1, keepdim=True)  # L2 归一化
                return vfeat
        elif getattr(self, "model_backbone", None) in [GME, LamRA, LamRA_QWEN2_5]:
            # GME/LamRA 骨干：使用模型自带的融合嵌入方法，直接获取文本-图像融合嵌入
            texts = [text.replace(VLM_IMAGE_TOKENS[QWEN2_VL] + '\n', '') for text in input["texts"]]  # 移除文本中的图像占位符
            images = []
            for imgs in input['images']:
                # 如果给定多张图像（如视频帧），仅选取中间帧
                if isinstance(imgs, list):
                    imgs = imgs[len(imgs) // 2]  # 取中间帧
                    assert not isinstance(imgs, list)  # 确保已提取为单张图像
                    images.append(imgs)
                else:
                    images.append(imgs)
            pooled_output = self.encoder.get_fused_embeddings(texts=texts, images=images)  # 获取文本-图像融合嵌入
            return pooled_output
        elif getattr(self, "model_backbone", None) == COLPALI:
            # ColPali 骨干：直接前向传播，返回完整隐藏状态用于 late-interaction 评分
            pooled_output = self.encoder(**input, return_dict=True, output_hidden_states=True)
            return pooled_output
        elif getattr(self, "model_backbone", None) == LLAVA_NEXT:
            # LLaVA-Next 骨干：需要先去除多余的 batch 维度，再前向传播和池化
            input['pixel_values'] = input['pixel_values'].squeeze(dim=1)  # 去除 pixel_values 的第二维（batch 内的多图维度）
            input['image_sizes'] = input['image_sizes'].squeeze(dim=1)  # 去除 image_sizes 的第二维
            hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)  # 前向传播
            hidden_states = hidden_states.hidden_states[-1]  # 取最后一层隐藏状态
            pooled_output = self._pooling(hidden_states, input['attention_mask'])  # 池化得到句级嵌入
            return pooled_output
        else:
            # 默认编码逻辑（适用于 Qwen2-VL、Qwen2.5-VL 等）：
            # 前向传播 → 取最后一层隐藏状态 → 池化
            hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)  # 前向传播
            hidden_states = hidden_states.hidden_states[-1]  # 取最后一层隐藏状态
            pooled_output = self._pooling(hidden_states, input['attention_mask'])  # 池化得到句级嵌入
            return pooled_output

    def _pooling(self, last_hidden_state, attention_mask):
        """
        对最后一层隐藏状态进行池化，提取句级嵌入表示。

        支持的池化策略：
            - 'last'/'eos': 取每个样本最后一个有效 token（即 EOS token）的隐藏状态作为句级表示

        对于左填充（left padding）的输入，最后一个位置即为 EOS token；
        对于右填充（right padding）的输入，需要根据 attention_mask 计算每个样本的 EOS 位置。

        参数：
            last_hidden_state: Tensor，形状为 [batch_size, seq_len, hidden_dim]，
                               最后一层隐藏状态
            attention_mask: Tensor，形状为 [batch_size, seq_len]，
                            注意力掩码，1 表示有效 token，0 表示填充

        返回值：
            Tensor，形状为 [batch_size, hidden_dim]，池化后的句级嵌入表示
        """
        if self.pooling == 'last' or self.pooling == 'eos':
            # 判断是否为左填充：如果所有样本的最后一个位置都是有效的，则为左填充
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            batch_size = last_hidden_state.shape[0]
            if left_padding:
                # 左填充：最后一个位置就是 EOS token，直接取最后一列
                reps = last_hidden_state[torch.arange(batch_size), -1, :]
            else:
                # 右填充：需要计算每个样本中最后一个有效 token 的位置
                eos_indices = attention_mask.sum(dim=1) - 1  # 每个样本的有效 token 数 - 1 = EOS 索引
                # 根据每个样本的 EOS 索引提取对应的隐藏状态
                reps = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device), eos_indices]
        else:
            raise NotImplementedError  # 目前仅支持 'last'/'eos' 池化策略
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)  # L2 归一化
        return reps

    @classmethod
    def build(cls, model_args: ModelArguments, **kwargs):
        """
        根据模型参数构建新的多模态嵌入模型（从预训练权重初始化）。

        本方法是工厂方法，根据 model_args 中的配置自动选择骨干模型类型，
        加载对应的预训练权重，并可选地应用 LoRA 适配器。

        构建流程：
            1. 加载模型配置，识别骨干类型
            2. 根据骨干类型加载对应的预训练模型（设置注意力实现、填充方向等）
            3. 如果启用 LoRA，则创建 LoRA 配置并包装基础模型
            4. 返回 MMEBModel 实例

        参数：
            model_args: ModelArguments，模型参数配置，包含模型名称、LoRA 配置等
            **kwargs: 其他传递给模型加载函数的关键字参数

        返回值：
            MMEBModel: 构建好的多模态嵌入模型实例
        """
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)  # 加载 HuggingFace 模型配置
        model_backbone = get_backbone_name(hf_config=config)  # 根据配置识别骨干类型
        print_master(f'Loading backbone [{model_backbone}] from {model_args.model_name}')  # 仅主进程打印加载信息
        # 根据骨干类型加载对应的预训练模型
        if model_backbone == PHI3V:
            # Phi-3-Vision：使用 eager 注意力实现，右填充，禁用 KV 缓存
            config._attn_implementation = "eager"
            config.padding_side = "right"
            config.use_cache = False
            base_model = Phi3VForCausalLM.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,  # 使用 bfloat16 精度以节省显存
                low_cpu_mem_usage=True,  # 低 CPU 内存模式，避免一次性加载全部权重到 CPU
            )
        elif model_backbone == LLAVA_NEXT:
            # LLaVA-Next：禁用 KV 缓存，左填充
            config.use_cache = False
            config.padding_side = "left"
            base_model = LlavaNextForConditionalGeneration.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone in [QWEN2_VL, QWEN2_5_VL]:
            # Qwen2-VL / Qwen2.5-VL：使用 Flash Attention 2，左填充，禁用 KV 缓存
            config._attn_implementation = "flash_attention_2"
            config.padding_side = "left"
            config.use_cache = False
            base_model = backbone2model[model_backbone].from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone in [QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION]:
            # Qwen2-VL / Qwen2.5-VL TokenSelection 变体：
            # 支持层跳过（layer skipping）的轻量化版本，可跳过部分语言模型层和视觉编码器层
            config._attn_implementation = "flash_attention_2"
            config.padding_side = "left"
            config.use_cache = False

            from .utils import parse_layer_type  # 导入层类型解析工具函数
            lm_qwen_layer = 28  # Qwen2-VL 语言模型的默认层数
            vis_qwen_layer = 32  # Qwen2-VL 视觉编码器的默认层数
            lm_skip_layer = parse_layer_type(model_args.lm_skip_layer, lm_qwen_layer)  # 解析要跳过的语言模型层
            vis_skip_layer = parse_layer_type(model_args.vis_skip_layer, vis_qwen_layer)  # 解析要跳过的视觉编码器层

            base_model = backbone2model[model_backbone].from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                lm_skip_layer=lm_skip_layer,  # 传入语言模型层跳过配置
                vis_skip_layer=vis_skip_layer,  # 传入视觉编码器层跳过配置
            )
        else:
            # 其他骨干：使用 AutoModelForCausalLM 自动加载，Flash Attention 2，禁用 KV 缓存
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_args.model_name, **kwargs, config=config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)

        # 如果启用 LoRA，创建 LoRA 配置并包装基础模型
        if model_args.lora:
            print_master(f'Loading lora adapter from {base_model}')
            lora_config = LoraConfig(
                r=model_args.lora_r,  # LoRA 秩（低秩矩阵的维度）
                lora_alpha=model_args.lora_alpha,  # LoRA 缩放因子
                target_modules=model_args.lora_target_modules.split(','),  # 要应用 LoRA 的目标模块列表
                lora_dropout=model_args.lora_dropout,  # LoRA Dropout 概率
                init_lora_weights="gaussian",  # LoRA 权重初始化方式：高斯分布
                use_dora=True,  # 启用 DoRA（Weight-Decomposed Low-Rank Adaptation），提升 LoRA 性能
                inference_mode=False  # 训练模式，允许 LoRA 权重更新
            )
            lora_model = get_peft_model(base_model, lora_config)  # 将基础模型包装为 LoRA 模型
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        else:
            # 不使用 LoRA，直接用基础模型作为编码器
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        return model

    @classmethod
    def load(cls, model_args: ModelArguments, is_trainable=True, **kwargs):
        """
        加载已训练的多模态嵌入模型（从 checkpoint 或预训练权重加载）。

        与 build 方法不同，load 方法用于加载已经训练过或微调过的模型，
        支持从 checkpoint 路径加载 LoRA 权重，并在推理时自动合并 LoRA 权重。

        加载流程：
            1. 确定模型加载路径（checkpoint 优先，否则使用模型名称）
            2. 识别骨干类型并加载对应的基础模型
            3. 如果启用 LoRA，加载 LoRA 适配器权重
            4. 将骨干名称绑定到模型实例上

        参数：
            model_args: ModelArguments，模型参数配置，包含模型名称、checkpoint 路径等
            is_trainable: bool，是否将模型设为可训练模式。
                          如果为 False 且使用 LoRA，则会合并 LoRA 权重到基础模型中
            **kwargs: 其他传递给模型加载函数的关键字参数（如 processor）

        返回值：
            MMEBModel: 加载好的多模态嵌入模型实例
        """
        # 优先使用 checkpoint 路径，否则使用模型名称
        model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)  # 加载模型配置
        # 如果 model_args 中没有 model_backbone 属性，则根据配置自动推断并设置
        if not hasattr(model_args, "model_backbone") or not model_args.model_backbone:
            model_backbone = get_backbone_name(hf_config=config, model_type=model_args.model_type)
            setattr(model_args, 'model_backbone', model_backbone)
        print_master(f'Loading backbone [{model_args.model_backbone}] from {model_name_or_path}')

        # 根据骨干类型加载对应的基础模型
        if model_args.model_backbone in {LLAVA_NEXT, QWEN2_VL, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION, E5_V}:
            # LLaVA-Next / Qwen2-VL / Qwen2.5-VL / TokenSelection 变体 / E5-V：
            # 使用 Flash Attention 2（包括视觉编码器），从原始模型名称加载权重
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config._attn_implementation = "flash_attention_2"
            config.vision_config._attn_implementation = "flash_attention_2"  # 视觉编码器也使用 Flash Attention 2
            base_model = backbone2model[model_args.model_backbone].from_pretrained(
                model_args.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                config=config
            )
        elif model_args.model_backbone == PHI3V:
            # Phi-3-Vision：禁用 KV 缓存，右填充
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            config.padding_side = "right"
            base_model = Phi3VForCausalLM.from_pretrained(model_args.model_name, **kwargs, config=config,
                                                          torch_dtype=torch.bfloat16, trust_remote_code=True)
            base_model.padding_side = "right"  # 额外设置模型对象的填充方向
        elif model_args.model_backbone == INTERNVIDEO2:
            # InternVideo2：从本地路径加载，不使用 HuggingFace Hub
            print_master(f'Loading backbone [{model_args.model_backbone}] from {"src/model/vlm_backbone/internvideo2/"}')
            config = AutoConfig.from_pretrained("src/model/vlm_backbone/internvideo2/",
                                                trust_remote_code=True)
            base_model = backbone2model[model_args.model_backbone].from_pretrained("src/model/vlm_backbone/internvideo2/", config=config,
                                                                                   trust_remote_code=True)
        elif model_args.model_backbone == GME:
            # GME 基线模型：使用自定义推理类，需要传入 processor
            base_model = GmeQwen2VL(model_args.model_name, processor=kwargs['processor'])
            setattr(base_model, 'config', config)  # 手动设置 config 属性
        elif model_args.model_backbone == LamRA:
            # LamRA 基线模型：使用自定义推理类
            base_model = LamRAQwen2VL(model_args.model_name)
            setattr(base_model, 'config', config)
        elif model_args.model_backbone == LamRA_QWEN2_5:
            # LamRA-Qwen2.5 基线模型：使用自定义推理类
            base_model = LamRAQwen25VL(model_args.model_name)
            setattr(base_model, 'config', config)
        elif model_args.model_backbone == COLPALI:
            # ColPali 基线模型：使用 from_pretrained 加载
            base_model = ColPali.from_pretrained(model_args.model_name)
            setattr(base_model, 'config', config)
        else:
            # 其他骨干：使用 AutoModelForCausalLM 自动加载
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_name_or_path, **kwargs, config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)

        # 如果启用 LoRA，加载已保存的 LoRA 适配器权重
        if model_args.lora:
            print_master(f'Loading LoRA from {model_name_or_path}')
            lora_config = LoraConfig.from_pretrained(model_name_or_path)  # 从 checkpoint 加载 LoRA 配置
            lora_model = PeftModel.from_pretrained(base_model, model_name_or_path, config=lora_config, is_trainable=is_trainable)  # 加载 LoRA 权重
            lora_model.load_adapter(model_name_or_path, lora_model.active_adapter, is_trainable=is_trainable)  # 加载活跃的 LoRA 适配器
            if not is_trainable:
                # 推理模式：将 LoRA 权重合并到基础模型中，消除推理时的额外计算开销
                lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        else:
            # 不使用 LoRA，直接用基础模型作为编码器
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )

        model.model_backbone = model_args.model_backbone  # 将骨干名称绑定到模型实例，供 encode_input 方法使用
        return model

    def save(self, output_dir: str):
        """
        保存模型权重到指定目录。

        参数：
            output_dir: str，模型保存路径，编码器的权重和配置将保存到此目录
        """
        self.encoder.save_pretrained(output_dir)

    def forward(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None, *args, **kwargs):
        """
        前向传播，计算对比学习损失（InfoNCE Loss）。

        核心逻辑：
            1. 分别编码查询（qry）和目标（tgt）得到嵌入表示
            2. 在 DDP 环境下，通过 all_gather 收集所有进程的嵌入
            3. 计算查询与所有目标之间的相似度矩阵
            4. 以对角线为正样本对，构建 InfoNCE 损失

        参数：
            qry: dict，查询侧输入，包含 input_ids、attention_mask、pixel_values 等
            tgt: dict，目标侧输入，格式与 qry 相同
            *args, **kwargs: 其他未使用参数

        返回值：
            如果 qry 或 tgt 为 None：返回 {"qry_reps": qry_reps, "tgt_reps": tgt_reps}，
            用于推理时仅编码一侧的输入
            否则：返回对比学习损失（标量张量）
        """
        qry_reps = self.encode_input(qry) if qry else None  # 编码查询，形状 [batch_size, hidden_dim]
        tgt_reps = self.encode_input(tgt) if tgt else None  # 编码目标，形状 [batch_size, hidden_dim]

        # 如果缺少查询或目标，直接返回编码结果（用于推理阶段）
        if qry_reps is None or tgt_reps is None:
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}

        # DDP 环境下：收集所有进程的嵌入，以获得更大的全局批次
        if self.is_ddp:
            all_qry_reps = self._dist_gather_tensor(qry_reps)  # 聚合所有进程的查询嵌入
            all_tgt_reps = self._dist_gather_tensor(tgt_reps)  # 聚合所有进程的目标嵌入
        else:
            all_qry_reps = qry_reps
            all_tgt_reps = tgt_reps

        # 计算相似度矩阵：[num_qry, num_tgt]
        scores = self.compute_similarity(all_qry_reps, all_tgt_reps)
        scores = scores.view(all_qry_reps.size(0), -1)  # 展平为一维行

        # 构建标签：正样本在对角线上（每个查询与对应的目标配对）
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        # 处理查询数量多于目标数量的情况（如多个查询对应同一个目标）
        target = target * (all_qry_reps.size(0) // all_tgt_reps.size(0))

        # 计算 InfoNCE 损失：相似度除以温度参数后计算交叉熵
        loss = self.cross_entropy(scores / self.temperature, target)
        if self.is_ddp:
            # DDP 下乘以 world_size，因为 CrossEntropyLoss 的 reduction='mean'
            # 会对所有进程的损失取平均，而我们需要的是全局平均
            loss = loss * self.world_size

        return loss

    def _dist_gather_tensor(self, t: Tensor):
        """
        在分布式训练中，收集所有进程上的张量并拼接。

        使用 all_gather 通信原语，将所有进程的同形状张量收集到每个进程上，
        然后沿第 0 维拼接为一个更大的张量。这在对比学习中用于构建更大的
        全局批次，使每个查询能与更多负样本计算相似度。

        参数：
            t: Tensor，当前进程上的张量，形状为 [local_batch_size, ...]

        返回值：
            Tensor，所有进程张量拼接后的结果，形状为 [world_size * local_batch_size, ...]
        """
        t = t.contiguous()  # 确保张量在内存中连续存储，all_gather 要求连续张量
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]  # 预分配接收缓冲区
        dist.all_gather(all_tensors, t)  # 收集所有进程的张量
        all_tensors[self.process_rank] = t  # 用当前进程的原始张量替换接收缓冲区中的副本，确保梯度正确
        all_tensors = torch.cat(all_tensors, dim=0)  # 沿 batch 维度拼接
        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        """
        计算查询嵌入与目标嵌入之间的相似度矩阵。

        使用矩阵乘法计算余弦相似度（如果嵌入已归一化）或点积相似度。

        参数：
            q_reps: Tensor，查询嵌入，形状为 [num_qry, hidden_dim]
            p_reps: Tensor，目标嵌入，形状为 [num_tgt, hidden_dim]

        返回值：
            Tensor，相似度矩阵，形状为 [num_qry, num_tgt]
        """
        return torch.matmul(q_reps, p_reps.transpose(0, 1))  # 矩阵乘法计算相似度

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        启用梯度检查点（Gradient Checkpointing），以节省训练显存。

        梯度检查点通过在前向传播时不保存中间激活值，而是在反向传播时
        重新计算这些值来减少显存占用，代价是增加约 30% 的计算时间。
        同时调用 enable_input_require_grads 确保输入嵌入的梯度可以传播，
        这对于 LoRA 训练是必要的。

        参数：
            gradient_checkpointing_kwargs: dict，梯度检查点的额外配置参数
        """
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)  # 启用编码器的梯度检查点
        if hasattr(self.encoder, "enable_input_require_grads"):
            self.encoder.enable_input_require_grads()  # 确保输入嵌入需要梯度，LoRA 训练必需
