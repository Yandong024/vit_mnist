import torch
from vit import VisionTransformer
from functools import partial
from torch import nn


def print_model_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:  # 只计算需要梯度的参数
            param_count = parameter.numel()  # 参数数量
            total_params += param_count
            print(f"{name} has {param_count} parameters")
    print(f"Total trainable parameters: {total_params}")


def test_vit_forward(x):
    # vit model
    model = VisionTransformer(img_size=[28],
                              patch_size=7,
                              in_chans=1,  # Linear CNN input dim
                              num_classes=10,
                              embed_dim=8,  # Linear CNN output feature
                              depth=2,  # layer number
                              num_heads=2,  # multi head
                              mlp_ratio=2,  # hidden dim
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6))
    out = model(x)
    print(out)
    # print model parameters
    # print_model_parameters(model)


if __name__ == '__main__':
    # mnist img 1x28x28, change into vit format
    input_data = torch.rand([1, 1, 28, 28])  # batch, channel, width, height
    test_vit_forward(input_data)


