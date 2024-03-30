import torch
import torch.nn as nn
import os
from torch import optim
from vit import VisionTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from mnist_dataset import mnist_dataset
from functools import partial
from torch.utils.tensorboard import SummaryWriter



class Solver(object):

    def __init__(self, epochs, warmup_epochs, batch_size, n_classes, num_workers, lr, dataset,
                 data_path, model_path, is_cuda=False, load_model=False):

        self.train_loader, self.test_loader = mnist_dataset()
        self.load_model = load_model
        self.model_path = model_path
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.is_cuda = is_cuda
        self.n_classes = n_classes

        self.model = VisionTransformer(img_size=[28],
                                       patch_size=7,
                                       in_chans=1,  # Linear CNN input dim
                                       num_classes=10,
                                       embed_dim=8,  # Linear CNN output feature
                                       depth=2,  # layer number
                                       num_heads=2,  # multi head
                                       mlp_ratio=2,  # hidden dim
                                       qkv_bias=True,
                                       norm_layer=partial(nn.LayerNorm, eps=1e-6))

        print('--------Network--------')
        print(self.model)

        if self.load_model:
            print("Using pretrained model")
            self.model.load_state_dict(torch.load(os.path.join(self.model_path, 'ViT_model.pt')))

        self.loss = nn.CrossEntropyLoss()

    def test_dataset(self):
        self.model.eval()
        actual = []
        pred = []

        for (x, y) in self.test_loader:

            with torch.no_grad():
                logits = self.model(x)
            predicted = torch.max(logits, 1)[1]

            actual += y.tolist()
            pred += predicted.tolist()

        acc = accuracy_score(y_true=actual, y_pred=pred)
        cm = confusion_matrix(y_true=actual, y_pred=pred, labels=range(self.n_classes))

        return acc, cm

    def test(self, train=True):
        if train:
            acc, cm = self.test_dataset()
            print(f"Train acc: {acc:.2%}\nTrain Confusion Matrix:")
            print(cm)

        acc, cm = self.test_dataset()
        print(f"Test acc: {acc:.2%}\nTest Confusion Matrix:")
        print(cm)

        return acc

    def train(self):
        iter_per_epoch = len(self.train_loader)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
        linear_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=1 / self.warmup_epochs, end_factor=1.0,
                                                    total_iters=self.warmup_epochs, last_epoch=-1, verbose=True)
        cos_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                         T_max=self.epochs - self.warmup_epochs, eta_min=1e-5,
                                                         verbose=True)

        best_acc = 0
        writer = SummaryWriter('runs/experiment_1')

        for epoch in range(self.epochs):

            self.model.train()
            for i, (x, y) in enumerate(self.train_loader):
                if self.is_cuda:
                    x, y = x.cuda(), y.cuda()

                logits = self.model(x)
                loss = self.loss(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                writer.add_scalar('Training loss', loss.item(), epoch * len(self.train_loader) + i)

                if i % 50 == 0 or i == (iter_per_epoch - 1):
                    print(f'Ep: {epoch + 1}/{self.epochs}, It: {i + 1}/{iter_per_epoch}, loss: {loss:.4f}')

            test_acc = self.test(train=((epoch + 1) % 25 == 0))  # Test training set every 25 epochs
            best_acc = max(test_acc, best_acc)
            print(f"Best test acc: {best_acc:.2%}\n")
            if best_acc > test_acc:
                torch.save(self.model.state_dict(), os.path.join(self.model_path, "ViT_model.pt"))

            if epoch < self.warmup_epochs:
                linear_warmup.step()
            else:
                cos_decay.step()

        writer.close()

