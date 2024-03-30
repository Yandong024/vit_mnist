import torch
from vit import VisionTransformer
from functools import partial
from torch import nn
from mnist_dataset import mnist_dataset
import os


def print_model_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:  # 只计算需要梯度的参数
            param_count = parameter.numel()  # 参数数量
            total_params += param_count
            print(f"{name} has {param_count} parameters")
    print(f"Total trainable parameters: {total_params}")


def vit_inference_mnist(model_path):
    # mnist data
    _, test_dataset = mnist_dataset(is_shuffle=False)
    examples = enumerate(test_dataset)
    # example_data：64x1x28x28, example_targets: 64 label
    batch_idx, (input_imgs, ground_truth) = next(examples)

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

    # load weights
    model.load_state_dict(torch.load(os.path.join(model_path, 'ViT_model.pt')))
    model.eval()
    inference_results = model(input_imgs)
    predicted = torch.max(inference_results, 1)[1].tolist()
    # print the results
    ground_truth = ground_truth.tolist()

    for idx in range(len(predicted)):
        is_acc = False
        if predicted[idx] == ground_truth[idx]:
            is_acc = True
        print("Idx:%d GT:%d Pred:%d isAcc:%r"%(idx, ground_truth[idx], predicted[idx], is_acc))
        print("---"*10)




if __name__ == '__main__':
    model_path = './models'
    vit_inference_mnist(model_path)


