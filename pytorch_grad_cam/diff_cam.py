import numpy as np
import torch
from pytorch_grad_cam.base_cam import BaseCAM
from typing import List

class DiffCategoryTarget:
    def __init__(self, class1_idx: int, class2_idx: int):
        self.class1_idx = class1_idx  
        self.class2_idx = class2_idx  

    def __call__(self, model_output):
        return model_output[..., self.class1_idx] - model_output[..., self.class2_idx]

class DiffCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None, compute_input_gradient=False):
        super(DiffCAM, self).__init__(model, target_layers, use_cuda,
                                      reshape_transform, compute_input_gradient)

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[DiffCategoryTarget] = None,
                target_size = None,
                eigen_smooth: bool = False,
                ) -> np.ndarray:

        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)


        W,H = self.get_target_width_height(input_tensor)
        outputs = self.activations_and_grads(input_tensor,H,W)
        if targets is None:
            if isinstance(outputs, (list, tuple)):
                output_data = outputs[0].detach().cpu().numpy()
            else:
                output_data = outputs.detach().cpu().numpy()

            top_categories = np.argsort(output_data, axis=-1)[:, -2:]

            targets = []
            for i in range(top_categories.shape[0]):
                class1_idx = int(top_categories[i, 1])  
                class2_idx = int(top_categories[i, 0])  
                target = DiffCategoryTarget(class1_idx, class2_idx)
                targets.append(target)
                print("target is ")
                print(class1_idx,class2_idx)

        if self.uses_gradients:
            self.model.zero_grad()
            if isinstance(outputs, (list, tuple)):
                loss = sum([target(output[0]) for target, output in zip(targets, outputs)])
            else:
                loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   target_size,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)
    
    def get_cam_weights(self,
                    input_tensor,
                    target_layer,
                    target_category,
                    activations,
                    grads):

        return np.mean(grads, axis=(2, 3))
