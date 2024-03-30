# Vit Mnist

**Process**

(1) Visualization dataset; mnist_dataset.py

(2) Forward vit network; vit_forward.py

(3) Training; train_vit.py

(4) Inference; test_vit.py

- python Env install

pip install -r requirements.txt



## Visualization

### 1. Dataloader

[what is dataloader ?](https://machinelearningmastery.com/training-a-pytorch-model-with-dataloader-and-dataset/)

> ## What is `DataLoader`?
>
> To train a deep learning model, you need data. Usually data is available as a dataset. In a dataset, there are a lot of data sample or instances. You can ask the model to take one sample at a time but usually you would let the model to process one batch of several samples. You may create a batch by extracting a slice from the dataset, using the slicing syntax on the tensor. For a better quality of training, you may also want to shuffle the entire dataset on each epoch so no two batch would be the same in the entire training loop. Sometimes, you may introduce **data augmentation** to manually introduce more variance to the data. This is common for image-related tasks, which you can randomly tilt or zoom the image a bit to generate a lot of data sample from a few images.
>
> You can imagine there can be a lot of code to write to do all these. But it is much easier with the `DataLoader`.



### 2. MNIST dataset

- Dimension

```python
Dataset MNIST
    Number of datapoints: 60000
    Root location: ./dataset
    Split: Train
    StandardTransform
Transform: Compose(
               ToTensor()
               Normalize(mean=(0.1307,), std=(0.3081,))
           )  
----------------------------------------
Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
```

- Visualiztion

  ![mnist](/Users/liuyandong/DLU-work-2024/1-courses/5-数据挖掘/code/vit/mnist.png)

## Vit Forward

### 1. Framework & Parameter

<img src="/Users/liuyandong/Library/Application Support/typora-user-images/image-20240329105831235.png" alt="image-20240329105831235" style="zoom:50%;" />

```python
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

  # total parameters
cls_token has 8 parameters (the same as patch_embedding)
pos_embed has 136 parameters
patch_embed.proj.weight has 392 parameters (cnn kernel:7x7 out_put channel:8 7x7x8)
patch_embed.proj.bias has 8 parameters (cnn out_put bias)
blocks.0.norm1.weight has 8 parameters (nn.LayerNorm )
blocks.0.norm1.bias has 8 parameters
blocks.0.attn.qkv.weight has 192 parameters (MLP: input-output 8x24=192)
blocks.0.attn.qkv.bias has 24 parameters
blocks.0.attn.proj.weight has 64 parameters (MLP: input-output 8x8=64)
blocks.0.attn.proj.bias has 8 parameters
blocks.0.norm2.weight has 8 parameters (nn.LayerNorm)
blocks.0.norm2.bias has 8 parameters
blocks.0.mlp.fc1.weight has 128 parameters
blocks.0.mlp.fc1.bias has 16 parameters
blocks.0.mlp.fc2.weight has 128 parameters
blocks.0.mlp.fc2.bias has 8 parameters
...
blocks.1. ...
...
norm.weight has 8 parameters
norm.bias has 8 parameters
head.weight has 80 parameters
head.bias has 10 parameters
Total trainable parameters: 1850
```



### 2. Multi_head Attention

<img src="/Users/liuyandong/Library/Application Support/typora-user-images/image-20240329105736110.png" alt="image-20240329105736110" style="zoom:50%;" />

> https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html

![image-20240329110658205](/Users/liuyandong/Library/Application Support/typora-user-images/image-20240329110658205.png)



<img src="/Users/liuyandong/Library/Application Support/typora-user-images/image-20240330080026299.png" alt="image-20240330080026299" style="zoom:50%;" />

https://medium.com/@geetkal67/attention-networks-a-simple-way-to-understand-self-attention-f5fb363c736d



## Training

(1) Loss

```
nn.CrossEntropyLoss()
```

(2) Optimizator

```python
optimizer = optim.Adam()
linear_warmup = optim.lr_scheduler.LinearLR()
cos_decay = optim.lr_scheduler.CosineAnnealingLR()
```

(3) Training Loop

<img src="/Users/liuyandong/Library/Application Support/typora-user-images/image-20240330075526164.png" alt="image-20240330075526164" style="zoom:50%;" />



Best test acc: 95.90%

Adjusting learning rate of group 0 to 1.0000e-05.
Ended at 2024-03-29 18:41:04
Duration: 2:24:45.417582



## Inference

(1) build model

(2) Load weights and bias

(3) Inference results

```
Idx:0 GT:7 Pred:7 isAcc:True
------------------------------
Idx:1 GT:2 Pred:2 isAcc:True
------------------------------
Idx:2 GT:1 Pred:1 isAcc:True
------------------------------
Idx:3 GT:0 Pred:0 isAcc:True
------------------------------
Idx:4 GT:4 Pred:4 isAcc:True
------------------------------
Idx:5 GT:1 Pred:1 isAcc:True
------------------------------
Idx:6 GT:4 Pred:4 isAcc:True
------------------------------
Idx:7 GT:9 Pred:9 isAcc:True
------------------------------
Idx:8 GT:5 Pred:5 isAcc:True
------------------------------
Idx:9 GT:9 Pred:9 isAcc:True
------------------------------
Idx:10 GT:0 Pred:0 isAcc:True
------------------------------
Idx:11 GT:6 Pred:6 isAcc:True
------------------------------
Idx:12 GT:9 Pred:9 isAcc:True
------------------------------
Idx:13 GT:0 Pred:0 isAcc:True
------------------------------
Idx:14 GT:1 Pred:1 isAcc:True
------------------------------
Idx:15 GT:5 Pred:5 isAcc:True
------------------------------
Idx:16 GT:9 Pred:9 isAcc:True
------------------------------
Idx:17 GT:7 Pred:7 isAcc:True
------------------------------
Idx:18 GT:3 Pred:3 isAcc:True
------------------------------
Idx:19 GT:4 Pred:4 isAcc:True
------------------------------
Idx:20 GT:9 Pred:9 isAcc:True
------------------------------
Idx:21 GT:6 Pred:6 isAcc:True
------------------------------
Idx:22 GT:6 Pred:6 isAcc:True
------------------------------
Idx:23 GT:5 Pred:5 isAcc:True
------------------------------
Idx:24 GT:4 Pred:4 isAcc:True
------------------------------
Idx:25 GT:0 Pred:0 isAcc:True
------------------------------
Idx:26 GT:7 Pred:7 isAcc:True
------------------------------
Idx:27 GT:4 Pred:4 isAcc:True
------------------------------
Idx:28 GT:0 Pred:0 isAcc:True
------------------------------
Idx:29 GT:1 Pred:1 isAcc:True
------------------------------
Idx:30 GT:3 Pred:3 isAcc:True
------------------------------
Idx:31 GT:1 Pred:1 isAcc:True
------------------------------
Idx:32 GT:3 Pred:3 isAcc:True
------------------------------
Idx:33 GT:4 Pred:4 isAcc:True
------------------------------
Idx:34 GT:7 Pred:7 isAcc:True
------------------------------
Idx:35 GT:2 Pred:2 isAcc:True
------------------------------
Idx:36 GT:7 Pred:7 isAcc:True
------------------------------
Idx:37 GT:1 Pred:1 isAcc:True
------------------------------
Idx:38 GT:2 Pred:2 isAcc:True
------------------------------
Idx:39 GT:1 Pred:1 isAcc:True
------------------------------
Idx:40 GT:1 Pred:1 isAcc:True
------------------------------
Idx:41 GT:7 Pred:7 isAcc:True
------------------------------
Idx:42 GT:4 Pred:4 isAcc:True
------------------------------
Idx:43 GT:2 Pred:2 isAcc:True
------------------------------
Idx:44 GT:3 Pred:3 isAcc:True
------------------------------
Idx:45 GT:5 Pred:5 isAcc:True
------------------------------
Idx:46 GT:1 Pred:1 isAcc:True
------------------------------
Idx:47 GT:2 Pred:2 isAcc:True
------------------------------
Idx:48 GT:4 Pred:4 isAcc:True
------------------------------
Idx:49 GT:4 Pred:4 isAcc:True
------------------------------
Idx:50 GT:6 Pred:6 isAcc:True
------------------------------
Idx:51 GT:3 Pred:3 isAcc:True
------------------------------
Idx:52 GT:5 Pred:5 isAcc:True
------------------------------
Idx:53 GT:5 Pred:5 isAcc:True
------------------------------
Idx:54 GT:6 Pred:6 isAcc:True
------------------------------
Idx:55 GT:0 Pred:0 isAcc:True
------------------------------
Idx:56 GT:4 Pred:4 isAcc:True
------------------------------
Idx:57 GT:1 Pred:1 isAcc:True
------------------------------
Idx:58 GT:9 Pred:9 isAcc:True
------------------------------
Idx:59 GT:5 Pred:5 isAcc:True
------------------------------
Idx:60 GT:7 Pred:7 isAcc:True
------------------------------
Idx:61 GT:8 Pred:8 isAcc:True
------------------------------
Idx:62 GT:9 Pred:5 isAcc:False
------------------------------
Idx:63 GT:3 Pred:3 isAcc:True
------------------------------
```



## Q&A Critical Thinking