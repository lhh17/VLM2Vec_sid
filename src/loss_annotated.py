"""
损失函数模块 (Loss Module)
=========================

本模块定义了检索模型训练过程中使用的对比学习损失函数，是模型训练的核心组件之一。

主要功能：
    - SimpleContrastiveLoss: 基础对比学习损失，通过计算查询与文档之间的相似度矩阵，
      使用交叉熵进行优化，使正样本对的相似度高于负样本对。
    - DistributedContrastiveLoss: 分布式版本的对比学习损失，在多 GPU 训练时
      跨进程收集张量以构建全局相似度矩阵，从而利用更多负样本提升训练效果。
    - InExampleContrastiveLoss: 样本内对比学习损失，用于分类任务场景，
      在每个样本内部从多个标签中选择正确的目标标签。

在项目中的位置：
    本模块位于 src/loss.py，被训练脚本 train.py 调用，用于计算模型前向传播后的梯度更新信号。
"""

from torch import Tensor  # PyTorch 张量类型，用于类型注解
import torch.distributed as dist  # PyTorch 分布式通信模块，用于多 GPU 训练时的进程间通信
import torch  # PyTorch 核心库，提供张量运算和自动求导
import torch.nn.functional as F  # PyTorch 函数式接口，提供交叉熵等常用损失函数


class SimpleContrastiveLoss:
    """
    简单对比学习损失函数

    通过计算查询向量 (x) 与文档向量 (y) 之间的余弦相似度矩阵，
    并使用带温度系数的交叉熵损失进行优化。

    核心思想：对于每个查询，使其对应正文档的相似度得分远高于其他负文档，
    从而学习到区分正负样本的表示。

    核心属性：
        temperature (float): 温度系数，控制 softmax 分布的平滑程度。
            较小的温度值使分布更尖锐，模型对困难负样本更敏感；
            较大的温度值使分布更平滑，训练更稳定。默认值为 0.02。
    """

    def __init__(self, temperature: float = 0.02):
        """
        初始化简单对比学习损失函数

        参数：
            temperature (float): 温度系数，默认 0.02。
                该值会除到 logits 上，起到缩放相似度得分的作用。
        """
        self.temperature = temperature

    def __call__(self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = 'mean') -> Tensor:
        """
        计算对比学习损失

        参数：
            x (Tensor): 查询向量，形状为 [num_queries, hidden_dim]
            y (Tensor): 文档向量，形状为 [num_docs, hidden_dim]
            target (Tensor, optional): 目标标签，指定每个查询对应的正文档索引。
                若为 None，则自动按顺序生成默认标签（假设每个查询对应相同数量的文档）。
            reduction (str): 损失归约方式，'mean' 为取均值，'sum' 为求和，'none' 为不归约。默认 'mean'。

        返回：
            Tensor: 计算得到的对比学习损失值
        """
        if target is None:
            # 当未提供目标标签时，自动构建默认标签
            # target_per_qry 计算每个查询对应的文档数量（假设均匀分配）
            target_per_qry = y.size(0) // x.size(0)
            # 生成目标索引：[0, target_per_qry, 2*target_per_qry, ...]
            # 即第 i 个查询的正文档索引为 i * target_per_qry
            target = torch.arange(
                0, x.size(0) * target_per_qry, target_per_qry, device=x.device, dtype=torch.long)

        # 计算查询与所有文档的相似度矩阵：[num_queries, num_docs]
        # x @ y^T 即为批量点积，结果矩阵中 logits[i][j] 表示第 i 个查询与第 j 个文档的相似度
        logits = torch.matmul(x, y.transpose(0, 1))

        # 使用带温度系数的交叉熵损失
        # logits / temperature 对相似度进行缩放，温度越小分布越尖锐
        # 交叉熵会自动对 logits 做 softmax，然后与 target 计算负对数似然
        loss = F.cross_entropy(logits / self.temperature, target, reduction=reduction)
        return loss


class DistributedContrastiveLoss(SimpleContrastiveLoss):
    """
    分布式对比学习损失函数

    继承自 SimpleContrastiveLoss，在多 GPU 分布式训练场景下，
    通过 all_gather 操作收集所有进程上的查询和文档向量，
    构建全局相似度矩阵以利用更多负样本，从而提升对比学习的效果。

    核心属性：
        word_size (int): 分布式训练的进程数（即 GPU 数量）
        rank (int): 当前进程的编号
        scale_loss (bool): 是否按进程数缩放损失。默认为 True，
            因为 all_gather 后样本数增加了 word_size 倍，
            缩放可以保持与单卡训练时损失量级一致
        temperature (float): 温度系数，默认 0.02
    """

    def __init__(self, n_target: int = 0, scale_loss: bool = True, temperature: float = 0.02):
        """
        初始化分布式对比学习损失函数

        参数：
            n_target (int): 目标数量参数（当前未使用，保留接口兼容性）。默认 0。
            scale_loss (bool): 是否按进程数缩放损失值。默认 True。
            temperature (float): 温度系数。默认 0.02。
        """
        # 确保分布式环境已正确初始化，否则无法进行跨进程通信
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__()
        self.word_size = dist.get_world_size()  # 获取分布式训练的总进程数
        self.rank = dist.get_rank()  # 获取当前进程的全局编号
        self.scale_loss = scale_loss
        self.temperature = temperature

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        """
        计算分布式对比学习损失

        先通过 all_gather 收集所有进程上的查询和文档向量，
        然后调用父类的 __call__ 方法在全局向量上计算对比损失。

        参数：
            x (Tensor): 当前进程上的查询向量，形状为 [local_num_queries, hidden_dim]
            y (Tensor): 当前进程上的文档向量，形状为 [local_num_docs, hidden_dim]
            **kwargs: 传递给父类的额外参数（如 target、reduction 等）

        返回：
            Tensor: 分布式对比学习损失值（可能经过进程数缩放）
        """
        # 收集所有进程上的查询向量，拼接为全局查询向量
        dist_x = self.gather_tensor(x)
        # 收集所有进程上的文档向量，拼接为全局文档向量
        dist_y = self.gather_tensor(y)
        # 在全局向量上计算对比损失
        loss = super().__call__(dist_x, dist_y, **kwargs)
        if self.scale_loss:
            # 按进程数缩放损失，保持与单卡训练时损失量级一致
            # 因为 all_gather 后样本数增加了 word_size 倍，
            # mean 归约后的损失会相应缩小，乘以 word_size 进行补偿
            loss = loss * self.word_size
        return loss

    def gather_tensor(self, t):
        """
        从所有进程收集张量并拼接

        使用 all_gather 通信原语，将所有进程上的同名张量收集到每个进程上，
        然后按第 0 维拼接为一个大张量。

        参数：
            t (Tensor): 当前进程上的局部张量

        返回：
            Tensor: 拼接后的全局张量，形状为 [world_size * local_size, ...]
        """
        # 预先分配与输入张量形状相同的空张量列表，用于接收各进程的数据
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        # 执行 all_gather 操作，每个进程将自己的 t 广播给所有其他进程
        dist.all_gather(gathered, t)
        # 用当前进程的原始数据替换 gathered 中对应位置，确保数据一致性
        gathered[self.rank] = t
        # 按第 0 维拼接所有进程的张量，形成全局张量
        return torch.cat(gathered, dim=0)


class InExampleContrastiveLoss:
    """
    样本内对比学习损失函数

    用于分类/标注场景的对比学习损失。与 SimpleContrastiveLoss 不同，
    本类在每个样本内部进行对比：给定一个查询和多个候选标签，
    从中选择正确的目标标签。

    典型应用场景：文档分类、实体链接等需要从固定候选集中选择正确答案的任务。

    核心属性：
        target_per_qry (int): 每个查询对应的标签数量（包含 1 个正标签 + n_hard_negatives 个硬负标签）
        temperature (float): 温度系数，用于缩放 logits。默认 1.0（即不缩放）
        ndim (int, optional): 可选的向量维度截断值。若指定，则只使用向量的前 ndim 维进行计算
    """

    def __init__(self, n_hard_negatives: int = 0, temperature: float = 1.0, ndim: int = None, *args, **kwargs):
        """
        初始化样本内对比学习损失函数

        参数：
            n_hard_negatives (int): 硬负样本数量。默认 0（即只有正标签，无硬负标签）。
            temperature (float): 温度系数。默认 1.0。
            ndim (int, optional): 向量维度截断值。若指定，只使用前 ndim 维。默认 None（使用全部维度）。
            *args, **kwargs: 保留的额外参数，用于接口兼容性。
        """
        # 每个查询对应的标签总数 = 1（正标签）+ n_hard_negatives（硬负标签）
        self.target_per_qry = n_hard_negatives + 1
        self.temperature = temperature
        self.ndim = ndim

    def __call__(self, x: Tensor, y: Tensor, reduction: str = 'mean'):
        """
        计算样本内对比学习损失

        在每个样本内部，计算查询向量与所有候选标签向量的相似度，
        然后使用交叉熵损失优化，使查询与正确标签的相似度最高。

        参数：
            x (Tensor): 查询向量，形状为 [batch_size, hidden_dim]
            y (Tensor): 标签向量，形状为 [batch_size, num_labels, hidden_dim]
            reduction (str): 损失归约方式。默认 'mean'。

        返回：
            tuple: (loss, loss_detail)
                - loss (Tensor): 对比学习损失值
                - loss_detail (dict): 包含详细信息的字典：
                    - 'logits': 相似度得分矩阵，形状为 [batch_size, num_labels]
                    - 'labels': 目标标签（全 0，即每个查询的第 0 个标签为正标签）
                    - 'preds': 模型预测的标签索引
        """
        # print("gather InExampleContrastiveLoss")
        # 如果处于分布式训练环境，则跨进程收集查询和标签向量
        if torch.distributed.is_initialized():
            x = dist_utils.dist_gather(x)
            y = dist_utils.dist_gather(y)

        bsz, ndim = x.size(0), x.size(1)  # bsz 为批次大小，ndim 为向量维度

        # 目标标签全为 0，即每个查询的第 0 个候选标签为正标签
        target = torch.zeros(bsz, dtype=torch.long, device=x.device)

        # 如果指定了维度截断，则只使用向量的前 ndim 维
        if self.ndim:
            ndim = self.ndim
            x = x[:, :ndim]
            y = y[:, :ndim]

        # 使用 einsum 计算每个查询与所有候选标签的相似度
        # x.view(bsz, 1, ndim) 形状为 [bsz, 1, ndim]，表示每个查询
        # y.view(bsz, -1, ndim) 形状为 [bsz, num_labels, ndim]，表示每个查询对应的候选标签集
        # 'bod,bsd->bs' 表示对 o 和 s 维度做点积，结果为 [bsz, num_labels] 的相似度矩阵
        # 乘以温度系数进行缩放
        logits = torch.einsum('bod,bsd->bs', x.view(bsz, 1, ndim), y.view(bsz, -1, ndim)) * self.temperature

        # 获取模型预测的标签索引（相似度最高的候选标签）
        preds = torch.argmax(logits, dim=-1)

        # 计算交叉熵损失：目标标签为 0（正标签始终在第 0 个位置）
        loss = F.cross_entropy(logits, target, reduction=reduction)

        # 构建详细信息字典，用于日志记录和分析
        loss_detail = {"logits": logits, "labels": target, "preds": preds}
        return loss, loss_detail
