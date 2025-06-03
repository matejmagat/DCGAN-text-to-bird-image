if __name__ == '__main__':
    from checkpoint import Checkpoint

    checkpoint = Checkpoint("../trained_models/train1", 5)

    from train import TrainingManager

    tm = TrainingManager(checkpoint)

    tm.train()