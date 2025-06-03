if __name__ == '__main__':
    import torch

    model_config = {
        "embedding_in": 768,
        "embedding_out": 128,
        "noise_size": 100,
        "image_channels": 3,
        "image_size": 64,
        "ngf": 64,
        "ndf": 64,
    }

    train_config = {
        "image_root": '/home/matej/Documents/DIPLOMSKI/2 SEMESTAR/SEMINAR/dataset/CUB/images',
        "embeddings_root": '/home/matej/Documents/DIPLOMSKI/2 SEMESTAR/SEMINAR/dataset/CUB/embeddings',
        "data_split": (0.7, 0.15, 0.15),
        "batch_size": 256,
        "start_lr": 0.0002,
        "patience": 5,
        "factor": 0.1,
        "num_epochs": 2500,
        "l1_coef": 50,
        "l2_coef": 100,
        "real_label": 1.,
        "fake_label": 0.,
        "seed": 75130,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    }

    from checkpoint import Checkpoint

    checkpoint = Checkpoint("./checkpoints/debugging", 5)
    checkpoint.init_checkpoint(train_config, model_config)

    from train import TrainingManager

    tm = TrainingManager(checkpoint)

    tm.train()