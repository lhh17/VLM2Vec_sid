"""
MMEB 训练器模块 (trainer.py 中文注释版)

本模块是 MMEB (Multimodal Embedding Benchmark) 项目的核心训练器实现，
基于 HuggingFace Transformers 的 Trainer 类进行扩展和定制。

主要功能：
1. MMEBTrainer: 基础训练器类，继承自 Transformers Trainer，
   重写了数据加载、损失计算、模型保存、检查点加载等关键方法，
   以适配多模态嵌入模型的训练需求。
2. GradCacheLateProcessTrainer: 支持梯度缓存(GradCache)的训练器，
   继承自 MMEBTrainer，用于处理大 batch 对比学习中的显存不足问题，
   通过分块计算和梯度累积实现高效的对比学习训练。

在项目中的位置：
- 本模块位于 src/trainer.py，是训练流程的核心入口
- 被 src/train.py 调用，负责实际的模型训练循环
- 依赖 src/model/model.py 中的 MMEBModel 进行模型前向传播
- 依赖 src/loss.py 中的对比学习损失函数
- 依赖 src/grad_cache/ 实现梯度缓存机制
"""

import collections  # 提供命名元组、计数器等特殊容器数据类型
import contextlib  # 提供上下文管理器工具，用于 with 语句
import functools  # 提供偏函数等高阶函数工具
import shutil  # 提供文件和目录操作，如删除目录树
import sys  # 提供系统相关功能，如最大整数 sys.maxsize
import time  # 提供时间相关功能，用于训练计时
from datetime import timedelta  # 时间差类型，用于训练时间统计

from packaging import version  # 版本号解析和比较工具
from accelerate import skip_first_batches, DistributedType, InitProcessGroupKwargs  # accelerate 库：跳过前N个batch、分布式类型枚举、进程组初始化参数
from transformers import PretrainedConfig  # 预训练模型配置基类
from transformers.trainer import Trainer, TRAINING_ARGS_NAME, TRAINER_STATE_NAME  # 核心训练器类、训练参数文件名常量、训练状态文件名常量
import torch.distributed as dist  # PyTorch 分布式通信模块
from typing import Optional  # 可选类型注解
import os  # 操作系统接口，文件和目录操作
import torch  # PyTorch 核心库
import math  # 数学运算，如 ceil、max 等

from src.data.collator.train_collator import split_vlm_inputs, get_dense_rep, split_and_process_vlm_inputs  # 数据整理函数：拆分VLM输入、获取密集表示、拆分并处理VLM输入
from src.model.model import MMEBModel  # MMEB 多模态嵌入模型
from src.loss import SimpleContrastiveLoss, DistributedContrastiveLoss  # 对比学习损失函数：单卡版本和分布式版本
from src.grad_cache.grad_cache import GradCache  # 梯度缓存实现，用于大batch对比学习的显存优化
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler  # PyTorch 数据加载工具：数据加载器、数据集、可迭代数据集、随机/顺序采样器

from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments  # 优化器名称枚举、并行模式枚举、训练参数类
from transformers.trainer_callback import (  # 训练器回调相关
    ExportableState,  # 可导出状态接口
    TrainerState,  # 训练器状态类
)
from transformers.trainer_utils import (  # 训练器工具函数
    TrainOutput,  # 训练输出结果类
    has_length,  # 判断数据集是否有长度信息
    speed_metrics, seed_worker,  # 速度指标计算、随机种子工作函数
)
from transformers.trainer_pt_utils import (  # 训练器 PyTorch 工具
    get_model_param_count,  # 获取模型参数数量
)
from transformers.trainer import FSDP_MODEL_NAME  # FSDP 模型文件名常量
from transformers.utils import (  # Transformers 工具函数和常量
    XLA_FSDPV2_MIN_VERSION,  # XLA FSDP v2 最低版本要求
    is_accelerate_available,  # 检查 accelerate 库是否可用
    is_apex_available,  # 检查 NVIDIA Apex 库是否可用
    is_torch_xla_available,  # 检查 PyTorch XLA 是否可用
    logging, is_sagemaker_mp_enabled,  # 日志工具、检查是否启用 SageMaker 模型并行
    CONFIG_NAME, WEIGHTS_NAME, SAFE_WEIGHTS_NAME,  # 模型配置/权重文件名常量
    ADAPTER_WEIGHTS_NAME, ADAPTER_SAFE_WEIGHTS_NAME  # 适配器权重文件名常量
)

from src.utils.basic_utils import batch_to_device  # 将 batch 数据移动到指定设备
from src.utils.basic_utils import print_master, print_rank  # 仅主进程打印、按进程号打印

if is_apex_available():  # 如果 NVIDIA Apex 库可用
    from apex import amp  # 导入 Apex 混合精度训练模块

if is_torch_xla_available():  # 如果 PyTorch XLA (TPU) 可用
    import torch_xla.core.xla_model as xm  # XLA 模型核心模块
    from torch_xla import __version__ as XLA_VERSION  # XLA 版本号

    # 判断 XLA FSDP v2 版本是否 >= 2.2
    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        pass
else:
    IS_XLA_FSDPV2_POST_2_2 = False  # XLA 不可用时设为 False

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


class MMEBTrainer(Trainer):
    """
    MMEB 基础训练器类

    继承自 HuggingFace Transformers 的 Trainer 类，针对多模态嵌入模型训练进行了定制化修改。
    主要重写了数据加载、损失计算、模型保存和检查点加载等方法。

    核心属性：
        is_ddp (bool): 是否启用了分布式数据并行(DDP)
        processor: 数据处理器，等同于 processing_class（tokenizer/processor）
        _dist_loss_scale_factor (int): 分布式损失缩放因子，DDP 时为 world_size，否则为 1
    """

    def __init__(self, *args, **kwargs):
        """初始化 MMEBTrainer

        参数：
            *args: 传递给父类 Trainer 的位置参数
            **kwargs: 传递给父类 Trainer 的关键字参数
        """
        super(MMEBTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()  # 检查分布式进程组是否已初始化
        self.processor = self.processing_class  # 将 processing_class 别名为 processor，方便访问
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1  # DDP 时损失需除以进程数以保持梯度一致性

    def get_batch_samples(self, epoch_iterator, num_batches):
        """从 epoch 迭代器中获取指定数量的 batch 样本

        在梯度累积场景下，需要一次获取多个 batch 的数据。

        参数：
            epoch_iterator: 当前 epoch 的数据迭代器
            num_batches (int): 需要获取的 batch 数量

        返回：
            tuple: (batch_samples, num_items_in_batch)
                - batch_samples (list): 获取到的 batch 列表
                - num_items_in_batch: batch 中有效标签的数量（非 -100 的标签数），
                  用于计算 token 级别的损失归一化
        """
        batch_samples = []
        num_items_in_batch = None
        for _ in range(num_batches):  # 从迭代器中获取 num_batches 个 batch
            try:
                batch_samples += [next(epoch_iterator)]
            except StopIteration:  # 迭代器耗尽时停止
                break
        if len(batch_samples) > 0 and "labels" in batch_samples[0]:
            # 目前不支持目标检测任务
            try:
                # 统计所有 batch 中有效标签的数量（标签值不等于 -100 的数量）
                num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
            except (TypeError, AttributeError):
                pass
        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            # 如果需要在多设备间平均 token 数量，则收集所有设备上的 token 数并求和
            num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum().item()
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.item()  # 将张量转换为 Python 标量
        return batch_samples, num_items_in_batch

    def compute_loss(self, model, inputs, *args, **kwargs):
        """计算训练损失

        重写父类方法，将输入拆分为查询(qry)和目标(tgt)两部分，
        直接调用模型前向传播计算对比学习损失。

        参数：
            model: 训练模型
            inputs: 输入数据，包含 (qry_inputs, tgt_inputs) 两个元素的元组
            *args: 额外位置参数
            **kwargs: 额外关键字参数

        返回：
            torch.Tensor: 模型计算得到的损失值
        """
        qry_inputs, tgt_inputs = inputs  # 拆分为查询输入和目标输入
        return model(qry=qry_inputs, tgt=tgt_inputs)  # 调用模型前向传播，返回损失

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """保存模型到指定目录

        重写父类方法，只保存编码器(encoder)部分，去除 state_dict 中的 'encoder.' 前缀。
        同时保存 tokenizer 和训练参数。

        参数：
            output_dir (Optional[str]): 输出目录路径
            state_dict: 模型状态字典，如果为 None 则从模型获取
        """
        os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

        if state_dict is None:
            state_dict = self.model.state_dict()  # 获取模型状态字典
        prefix = 'encoder.'  # 编码器参数的前缀
        # 断言所有参数键都以 'encoder.' 开头，确保只保存编码器部分
        assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
        # 去除 'encoder.' 前缀，使保存的参数名与编码器模型一致
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
        self.model.encoder.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors  # 使用 safetensors 格式保存（如果配置启用）
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)  # 保存 tokenizer

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))  # 保存训练参数


    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """获取训练数据采样器

        重写父类方法，覆盖原始 Trainer 的采样器逻辑。
        当训练数据集不存在或没有长度信息时返回 None（适用于 IterableDataset）。

        返回：
            Optional[torch.utils.data.Sampler]: 训练采样器，或 None
        """
        # 覆盖原始 trainer 的方法
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None  # 对于 IterableDataset 或无长度信息的数据集，不使用采样器
            return RandomSampler(self.train_dataset)  # 注意：此行不可达，可能是遗留代码

    def get_train_dataloader(self) -> DataLoader:
        """获取训练数据加载器

        重写父类方法，禁用 self.accelerator.prepare，因为它会包装 DataLoaderDispatcher，
        导致以下问题：
        (1) RuntimeError: 不能在 dispatch_batches=True 或使用 IterableDataset 时使用不同大小的 batch
        (2) 数据加载器的所有输出必须是张量

        返回：
            DataLoader: 配置好的训练数据加载器
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        train_dataset = self._remove_unused_columns(train_dataset, description="training")  # 移除模型不需要的列
        dataloader_params = {
            "batch_size": self._train_batch_size,  # 训练 batch 大小
            "collate_fn": data_collator,  # 数据整理函数
            "num_workers": self.args.dataloader_num_workers,  # 数据加载工作进程数
            "pin_memory": self.args.dataloader_pin_memory,  # 是否将数据固定在内存中（加速 GPU 传输）
            "persistent_workers": self.args.dataloader_persistent_workers,  # 是否保持工作进程持久化
        }
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            # 对于普通 MapDataset，使用采样器和完整的数据加载参数
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last  # 是否丢弃最后不完整的 batch
            dataloader_params["worker_init_fn"] = seed_worker  # 工作进程随机种子初始化
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor  # 预取因子
        else:
            # 对于 IterableDataset，不使用采样器，不进行随机打乱
            dataloader_params["sampler"] = None
            dataloader_params["shuffle"] = False
            dataloader_params["drop_last"] = True
            # 同时调整 prefetch_factor 和 persistent_workers 会导致第2个 epoch 挂起
            dataloader_params["prefetch_factor"] = None
        return DataLoader(train_dataset, **dataloader_params)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """从检查点加载模型

        重写父类方法，使用 MMEBModel.load() 加载模型，
        而非标准的 Transformers 检查点加载方式。

        参数：
            resume_from_checkpoint: 检查点路径
            model: 未使用的参数（保持接口兼容）
        """
        self.model_args.checkpoint_path = resume_from_checkpoint  # 设置模型参数中的检查点路径
        logger.info(f"Loading checkpoint from {resume_from_checkpoint}")
        self.model = MMEBModel.load(self.model_args)  # 使用 MMEBModel 的 load 方法加载模型
        self.model_wrapped = self.model  # 更新包装模型引用

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """内部训练循环

        重写父类方法，实现完整的训练循环逻辑。这是训练过程的核心方法，
        包含了数据加载、前向传播、反向传播、梯度累积、优化器更新、
        学习率调度、检查点保存、日志记录等所有训练步骤。

        参数：
            batch_size: 训练 batch 大小，None 时使用默认值
            args: 训练参数，None 时使用 self.args
            resume_from_checkpoint: 从检查点恢复训练的路径
            trial: 超参数搜索试验对象
            ignore_keys_for_eval: 评估时忽略的键列表

        返回：
            无（训练结果存储在 self.state 中）
        """
        self.accelerator.free_memory()  # 释放加速器中的内存
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:  # 自动寻找合适的 batch 大小
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory  # 导入内存释放工具

                (self.model_wrapped,) = release_memory(self.model_wrapped)  # 释放包装模型的内存
                self.model_wrapped = self.model  # 重新设置为原始模型

                # 在初始 pass 之后检查 DeepSpeed 并修改配置
                if self.is_deepspeed_enabled:
                    # 临时取消 self.args.train_batch_size 的设置
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)  # 将新的 batch 大小传播到 DeepSpeed 配置
                    self.args.per_device_train_batch_size = original_bs  # 恢复原始 batch 大小
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # 数据加载器和训练步数
        train_dataloader = self.get_train_dataloader()

        # 设置训练控制变量：
        # 训练轮数: num_train_epochs
        # 每轮训练步数: num_update_steps_per_epoch
        # 总训练步数: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size  # 总 batch 大小 = 单卡 batch × 梯度累积步数 × 进程数

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):  # 如果数据加载器有长度信息（MapDataset）
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps  # 每轮更新步数 = batch数 / 梯度累积步数
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)  # 至少为 1
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:  # 如果指定了最大训练步数
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )  # 计算需要的训练轮数（向上取整）
                # 如果最后一个 batch 较小，可能略有偏差，但这是最好的估算
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:  # 未指定 max_steps，按 num_train_epochs 计算
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # 数据加载器无长度信息时，依赖 max_steps
            max_steps = args.max_steps
            # 设置非常大的轮数，以便迭代器可以尽可能多地遍历
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )  # 数据加载器无长度信息时必须设置 max_steps

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled  # 是否延迟创建优化器

        # 需要重置学习率调度器，因为后续调用的参数可能不同
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)  # 创建优化器和学习率调度器

        self.state = TrainerState(  # 初始化训练器状态
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None  # 是否正在进行超参数搜索
        self.state.train_batch_size = self._train_batch_size

        # 如果 logging_steps/eval_steps/save_steps 给定为比例值（<1），则计算绝对步数
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)  # 比例转绝对步数
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # 如果需要，激活梯度检查点以节省显存
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)  # 包装模型（DDP/FSDP/DeepSpeed 等）

        # 由于模型已被包装，不使用 accelerator.prepare
        # 这是为了处理 FSDP-XLA、SageMaker MP/DP、DataParallel、IPEX 等特殊情况
        use_accelerator_prepare = True if model is self.model else False  # 如果模型未被包装，则需要 accelerator.prepare

        if delay_optimizer_creation:  # 延迟创建优化器的情况
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()  # FSDP QLoRA 插件更新
                self.model = self.accelerator.prepare(self.model)  # 使用 accelerator 准备模型
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)  # 创建优化器和调度器

        # 使用 accelerator.prepare 准备模型和优化器
        if use_accelerator_prepare:
            self.model.train()  # 设置为训练模式
            if hasattr(self.lr_scheduler, "step"):  # 如果学习率调度器有 step 方法
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)  # Apex 模式下只准备模型
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)  # 准备模型和优化器
            else:
                # 处理传入 "DummyScheduler" 的情况，例如 DeepSpeed 配置中指定时
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # DDP + LOMO 的情况
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:  # FSDP 模式下更新模型引用
            self.model = self.model_wrapped = model

        # 在此函数的剩余部分，model 是外部模型（无论是否被包装）
        if model is not self.model:
            self.model_wrapped = model

        # 向后兼容：DeepSpeed 模式下设置 deepspeed 属性
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # 检查是否存在已保存的优化器或调度器状态
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # 重要提示：此时
        # self.model         是 Transformers 模型
        # self.model_wrapped 是 DDP(Transformers 模型)、Deepspeed(Transformers 模型)、
        # FSDP(Transformers 模型)、Dynamo 优化模块(Transformers 模型) 等

        # 开始训练！
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()  # 记录训练开始时间
        epochs_trained = 0  # 已训练的轮数
        steps_trained_in_current_epoch = 0  # 当前 epoch 中已训练的步数
        steps_trained_progress_bar = None

        # @ruimeng 使用 steps_trained_in_current_epoch 跳过 batch 以查找有问题的数据
        # steps_trained_in_current_epoch = 42

        # 检查是否从检查点继续训练
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))  # 从 JSON 加载训练状态
            self.compare_trainer_and_checkpoint_args(self.args, self.state)  # 比较当前参数与检查点参数
            self._load_callback_state()  # 加载回调状态
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)  # 计算已完成的轮数
            if not args.ignore_data_skip:  # 如果不跳过数据
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)  # 当前 epoch 中已训练的更新步数
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps  # 转换为微步数（考虑梯度累积）
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # 更新回调处理器中的引用
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        # 如果状态已保存，这些值应该相同，但为了安全起见在加载后重新设置
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss 使用张量以避免 TPU 通过 .item() 进行同步
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar 在每次调用 tr_loss.item() 时更新，存储所有损失的总和
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step  # 上次记录日志时的全局步数
        model.zero_grad()  # 清零梯度
        grad_norm: Optional[float] = None  # 梯度范数
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)  # 触发训练开始回调

        if args.eval_on_start:  # 如果配置了训练前先评估
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        total_batched_samples = 0  # 总已处理样本计数
        for epoch in range(epochs_trained, num_train_epochs):  # 遍历每个训练轮次
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader.dataset, "set_epoch"):
                # 设置当前 epoch 编号，确保分布式训练中数据打乱一致
                epoch_dataloader.dataset.set_epoch(epoch)

            # 如果需要，在每个 epoch 开始时重置过去的 mems 状态
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )  # 当前 epoch 中的步数
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)  # 触发 epoch 开始回调

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)  # 恢复随机数生成器状态

            rng_to_sync = False  # 是否需要同步随机数生成器
            steps_skipped = 0  # 跳过的步数
            if steps_trained_in_current_epoch > 0:  # 如果当前 epoch 有已训练的步数需要跳过
                epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)  # 跳过已训练的 batch
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True  # 标记需要同步随机数状态

            step = -1  # 当前 epoch 内的步数计数器
            epoch_iterator = iter(epoch_dataloader)  # 创建 epoch 数据迭代器
            # 将 epoch 迭代器按梯度累积步数分块
            remainder = num_examples % args.gradient_accumulation_steps  # 最后一个更新步的 batch 数量（可能不足一个完整的梯度累积步数）
            num_items_in_batch = None
            if remainder == 0:
                remainder = args.gradient_accumulation_steps  # 如果整除，则最后一个更新步也是完整的梯度累积步数
            update_step = -1  # 更新步计数器
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1  # 总更新步数
            for _ in range(total_updates):  # 遍历每个更新步
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder  # 当前更新步的 batch 数量
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches)  # 预取 batch 样本
                for i, inputs in enumerate(batch_samples):  # 遍历当前更新步中的每个 batch
                    step += 1
                    total_batched_samples += 1

                    # 统计当前 batch 中的数据集分布
                    dataset_stat = collections.Counter(inputs[0]['global_dataset_name'])
                    if step < 5:  # 只在前5步打印数据集信息
                        print_rank(f"dataset name: {str(set(inputs[0]['global_dataset_name']))}")
                        for dname, count in sorted(dataset_stat.items(), key=lambda t:t[1], reverse=True):
                            print_rank(f"\t\tdataset_name={dname}, count={count}")

                    # 判断是否是同步梯度的步骤
                    is_last_step_and_steps_less_than_grad_acc = (
                        steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                    )  # 当 epoch 步数少于梯度累积步数时的特殊处理
                    do_sync_step = is_last_step_and_steps_less_than_grad_acc or (
                        total_batched_samples % args.gradient_accumulation_steps == 0
                    )  # 是否在当前步同步梯度
                    # 由于使用了预取，需要手动设置 sync_gradients
                    if not do_sync_step:
                        self.accelerator.gradient_state._set_sync_gradients(False)  # 非同步步：禁用梯度同步
                    else:
                        self.accelerator.gradient_state._set_sync_gradients(True)  # 同步步：启用梯度同步

                    if self.args.include_num_input_tokens_seen:  # 如果需要统计已看到的输入 token 数
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                            self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).cpu().item()  # 收集所有设备的 token 数并累加
                    if rng_to_sync:  # 如果需要同步随机数状态
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # 跳过已训练的步（恢复训练时）
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)  # 恢复随机数状态
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:  # 在梯度累积步的边界触发 step_begin 回调
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # 显式避免使用 accelerator.accumulate 进行生成式训练
                    # 在梯度累积中，除最后一个 batch 外都使用 no_sync 避免梯度同步
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1  # 非最后一个 batch：禁用梯度同步
                        else contextlib.nullcontext  # 最后一个 batch：允许梯度同步
                    )
                    with context():
                        tr_loss_step = self.training_step(model, inputs, num_items_in_batch)  # 执行单步训练，计算损失

                    if (
                        args.logging_nan_inf_filter  # 如果启用了 NaN/Inf 过滤
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # 如果损失为 NaN 或 Inf，使用之前记录的平均损失代替
                        tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step  # 累加损失

                    self.current_flos += float(self.floating_point_ops(inputs))  # 累加浮点运算数

                    if do_sync_step:  # 在同步步执行优化器更新
                        # 由于使用了预取，需要手动将 sync_gradients 设为 True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # 梯度裁剪
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            # DeepSpeed 有自己的梯度裁剪逻辑

                            if self.use_apex:
                                # Apex 模式下使用 Apex 的梯度裁剪
                                _grad_norm = torch.nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                # 标准模式：使用 accelerator 的梯度裁剪
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()  # DeepSpeed 模式下获取全局梯度范数
                                # 某些情况下梯度范数可能不是 float 类型
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm  # 非DeepSpeed模式直接使用裁剪后的梯度范数

                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)  # 优化器步骤前回调

                        self.optimizer.step()  # 执行优化器更新

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)  # 优化器步骤后回调

                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped  # 检查优化器是否实际执行了更新
                        if optimizer_was_run:
                            # 延迟优化器调度直到指标生成
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()  # 更新学习率（非 ReduceLROnPlateau 调度器）

                        model.zero_grad()  # 清零梯度
                        self.state.global_step += 1  # 更新全局步数
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch  # 更新当前 epoch 进度
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)  # 步结束回调
                        self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, time.time())  # 可能执行日志记录、保存或评估
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)  # 子步结束回调

                    # PyTorch/XLA 依赖数据加载器插入 mark_step
                    # 由于我们提前跳出循环，需要手动插入 mark_step
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                # 也需要跳出嵌套循环
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:  # 如果 epoch 中没有任何样本
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)  # epoch 结束回调
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, time.time())  # epoch 结束时可能记录日志、保存或评估

            if self.control.should_training_stop:  # 如果训练被请求停止
                break

        if args.past_index and hasattr(self, "_past"):
            # 训练结束时清理状态
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:  # 如果配置了训练结束后加载最佳模型
            # 等待所有进程到达此处，确保模型已被进程0保存
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()  # 分布式屏障同步

            self._load_best_model()  # 加载最佳模型

        # 添加剩余的 tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # 避免除零错误
        train_loss = self._total_loss_scalar / effective_global_step  # 计算平均训练损失

        metrics = speed_metrics(  # 计算速度指标
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()  # 存储浮点运算数
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False  # 标记训练结束

        self._memory_tracker.stop_and_update_metrics(metrics)  # 停止内存追踪并更新指标

        self.log(metrics)  # 记录指标

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)  # 获取排序后的检查点列表

        # 当 save_total_limit=1 时，删除最后一个检查点（如果与最佳检查点不同且进程允许保存）
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)  # 删除旧检查点

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)  # 训练结束回调

        # 等待检查点上传完成
        self._finish_current_push()

        # 训练结束后，移除 NEFTune 噪声的前向钩子，恢复嵌入层的原始前向传播
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)  # 返回训练输出


class GradCacheLateProcessTrainer(MMEBTrainer):
    """
    支持梯度缓存(GradCache)的训练器

    继承自 MMEBTrainer，实现了梯度缓存机制以解决大 batch 对比学习中的显存不足问题。
    核心思想：将大 batch 拆分为多个小 chunk，分别计算前向传播获取表示，
    然后使用缓存的表示计算对比损失并反向传播梯度。

    适配自 gradcache 仓库。

    核心属性：
        max_length (int): 输入序列的最大长度，默认 512
        model_args: 模型参数配置
        gc (GradCache): 梯度缓存实例，负责分块前向传播和梯度缓存
        is_ddp (bool): 是否启用了分布式数据并行
        _dist_loss_scale_factor (int): 分布式损失缩放因子
    """
    def __init__(self, *args, **kwargs):
        """初始化 GradCacheLateProcessTrainer

        从 kwargs 中提取 max_length 和 model_args 参数，
        然后初始化梯度缓存(GradCache)实例。

        参数：
            *args: 传递给父类的位置参数
            **kwargs: 关键字参数，可包含:
                - max_length (int): 输入序列最大长度
                - model_args: 模型参数配置
        """
        self.max_length = kwargs.get("max_length", 512)  # 获取最大序列长度，默认 512
        if "max_length" in kwargs:
            del kwargs["max_length"]  # 从 kwargs 中移除，避免传递给父类
        self.model_args = kwargs.get("model_args", None)  # 获取模型参数配置
        if "model_args" in kwargs:
            del kwargs["model_args"]  # 从 kwargs 中移除
        super(GradCacheLateProcessTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()  # 检查是否启用 DDP
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1  # 分布式损失缩放因子
        # 根据 DDP 模式选择对应的对比学习损失函数
        loss_fn_cls = DistributedContrastiveLoss if self.is_ddp else SimpleContrastiveLoss
        loss_fn = loss_fn_cls(temperature=self.model.temperature)  # 使用模型的温度参数创建损失函数

        # 初始化梯度缓存实例
        self.gc = GradCache(
            models=[self.model, self.model],  # 查询和目标使用同一个模型
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],  # 查询和目标的分块大小
            loss_fn=loss_fn,  # 对比学习损失函数
            split_input_fn=split_and_process_vlm_inputs,  # 输入拆分和处理函数
            # process_fn=process_fn,  # 数据处理函数（已注释）
            get_rep_fn=get_dense_rep,  # 从模型输出中获取密集表示的函数
            fp16=self.args.fp16,  # 是否使用 FP16 混合精度
            scaler=self.scaler if self.args.fp16 else None  # FP16 梯度缩放器
        )

    def training_step(self, model, inputs, *args, **kwargs) -> torch.Tensor:
        """执行单步训练

        重写父类方法，使用梯度缓存机制进行训练。
        在分布式模式下，通过 GradCache 分块计算前向传播和对比损失；
        在单卡模式下，直接调用模型前向传播。

        参数：
            model: 训练模型
            inputs: 输入数据，包含 (queries, targets) 两个元素的元组
            *args: 额外位置参数
            **kwargs: 额外关键字参数

        返回：
            torch.Tensor: 归一化后的损失值（除以分布式缩放因子）
        """
        model.train()  # 确保模型处于训练模式
        queries, targets = inputs  # 拆分查询和目标输入

        # 获取模型所在设备
        if hasattr(model, "module"):
            device = model.module.device  # DDP 包装的模型
        elif hasattr(model, "device"):
            device = model.device  # 普通模型
        else:
            device = next(model.parameters()).device  # 从参数推断设备

        # 将输入数据移动到模型所在设备
        queries = batch_to_device(queries, device)
        targets = batch_to_device(targets, device)

        _distributed = dist.is_initialized() and dist.get_world_size() > 1  # 是否为分布式训练
        if _distributed:
            # 分布式模式：使用梯度缓存进行分块前向传播和对比损失计算
            gc_queries, gc_targets = {'qry': queries}, {'tgt': targets}
            self.gc.models = [model, model]  # 更新梯度缓存中的模型引用
            loss = self.gc(gc_queries, gc_targets, no_sync_except_last=True)  # 除最后一步外不同步梯度
        else:
            # 单卡模式：直接调用模型前向传播
            loss = model(queries, targets)
        return loss / self._dist_loss_scale_factor  # 损失除以缩放因子，确保分布式训练中梯度一致


    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """保存模型到指定目录

        重写父类方法，与 MMEBTrainer._save 类似，
        但额外保存了编码器的配置文件，并使用 print_master 输出保存信息。

        参数：
            output_dir (Optional[str]): 输出目录路径
            state_dict: 模型状态字典，如果为 None 则从模型获取
        """
        print_master(f"Saving model to {output_dir}")  # 仅主进程打印保存信息
        os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

        if state_dict is None:
            state_dict = self.model.state_dict()  # 获取模型状态字典
        prefix = 'encoder.'  # 编码器参数的前缀
        # 断言所有参数键都以 'encoder.' 开头
        assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
        # 去除 'encoder.' 前缀
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
        self.model.encoder.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors  # 保存编码器权重
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)  # 保存 tokenizer

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))  # 保存训练参数
        self.model.encoder.config.to_json_file(os.path.join(output_dir, 'config.json'))  # 额外保存编码器配置文件
