import numpy as np
import torch
from typing import Callable, List, Tuple
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.base_cam import BaseCAM

class DifferenceCAM(BaseCAM):
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True) -> None:
        """
        初始化 DifferenceCAM。

        Args:
            model (torch.nn.Module): 需要解释的模型。
            target_layers (List[torch.nn.Module]): 目标层列表。
            use_cuda (bool, optional): 是否使用 CUDA。默认值为 False。
            reshape_transform (Callable, optional): 重塑转换函数。默认值为 None。
            compute_input_gradient (bool, optional): 是否计算输入梯度。默认值为 False。
            uses_gradients (bool, optional): 是否使用梯度。默认值为 True。
        """
        super(DifferenceCAM, self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform,
            compute_input_gradient,
            uses_gradients
        )

    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        target_layer: torch.nn.Module,
                        targets: List[torch.nn.Module],
                        activations: np.ndarray,
                        grads: np.ndarray) -> np.ndarray:
        """
        计算差值梯度，并通过全局平均池化得到权重。

        Args:
            input_tensor (torch.Tensor): 输入张量。
            target_layer (torch.nn.Module): 目标层。
            targets (List[torch.nn.Module]): 目标类别列表，包含两个类别。
            activations (np.ndarray): 激活图。
            grads (np.ndarray): 差值梯度。

        Returns:
            np.ndarray: 每个通道的权重。
        """
        return np.mean(grads, axis=(2, 3))

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                target_size,
                K: int = 1,  # 选择第 K 个最高得分的类别，默认为 1 表示第二高得分的类别
                alpha: float = 1.0,  # 损失函数的 alpha 参数
                eigen_smooth: bool = False) -> np.ndarray:
        """
        重写 BaseCAM 的 forward 方法，以确保自动选择指定得分的类别作为目标，并计算它们的差值。

        Args:
            input_tensor (torch.Tensor): 输入张量。
            targets (List[torch.nn.Module], optional): 目标类别。如果为 None，将自动选择。
            target_size: 目标尺寸。
            K: 选择第 K 个最高的类别（默认是第 2 高的类别）。
            alpha: 调整损失计算中的权重系数。
            eigen_smooth (bool, optional): 是否使用特征值平滑。默认值为 False。

        Returns:
            np.ndarray: 生成的 CAM 图。
        """
        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        W, H = self.get_target_width_height(input_tensor)
        outputs = self.activations_and_grads(input_tensor, H, W)

        if targets is None:
            # 自动选择得分最高和第 K 高的类别
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # 取第一个元素
            output_np = outputs.cpu().data.numpy()
            top_indices = np.argsort(output_np, axis=-1)[..., -K-1:]  # 获取分数最高的 K 个类别
            targets = [ClassifierOutputTarget(category) for category in [top_indices[0][-1], top_indices[0][-K-1]]]
            

        if self.uses_gradients:
            self.model.zero_grad()

            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)

            # 计算自定义损失 w = (w_1 - w_k) + α * w_k
            w1 = outputs[:, targets[0].category]
            wk = outputs[:, targets[1].category]
            loss = (w1 - wk) + alpha * wk
            loss.backward(retain_graph=True)

        # 计算每个目标层的 CAM
        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, target_size, eigen_smooth)

        # 提取前两类的分数
        top1_score = outputs[:, targets[0].category].cpu().detach().numpy()
        top2_score = outputs[:, targets[1].category].cpu().detach().numpy()

        if isinstance(input_tensor, list):
            return self.aggregate_multi_layers(cam_per_layer), top1_score, top2_score
        else:
            return self.aggregate_multi_layers(cam_per_layer), top1_score, top2_score