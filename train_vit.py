import datetime
import os
from train_solver import Solver


def main(epochs, warmup_epochs, batch_size, n_classes, num_workers, lr, dataset, data_path):
    model_path = './models'
    os.makedirs(model_path, exist_ok=True)
    solver = Solver(epochs, warmup_epochs, batch_size, n_classes, num_workers, lr, dataset, data_path, model_path)
    solver.train()


if __name__ == '__main__':
    epochs = 200
    warmup_epochs = 10
    batch_size = 256
    n_classes = 10
    num_workers = 4
    lr = 5e-4
    dataset = 'mnist'
    data_path = './dataset/'
    # ViT training time
    start_time = datetime.datetime.now()
    print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))
    # training
    main(epochs, warmup_epochs, batch_size, n_classes, num_workers, lr, dataset, data_path)
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
    print("Duration: " + str(duration))