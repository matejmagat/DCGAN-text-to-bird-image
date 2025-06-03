import os
from pathlib import Path
from typing import Dict
import json

import numpy as np
import torch



class Checkpoint:
    def __init__(self, checkpoint_path, save_frequency):
        self.checkpoint_path = Path(checkpoint_path)
        self.saved_models_path = self.checkpoint_path / 'saved_models'
        self.save_frequency = save_frequency

    def init_checkpoint(self, train_conf:Dict, model_conf:Dict):
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.saved_models_path.mkdir(parents=True, exist_ok=True)
        train_conf["device"] = str(train_conf["device"])

        with open(self.checkpoint_path / 'train_conf.json', 'w') as json_file:
            json.dump(train_conf, json_file, indent=4)

        with open(self.checkpoint_path / 'model_conf.json', 'w') as json_file:
            json.dump(model_conf, json_file, indent=4)


    def __call__(self, epoch, lr, models, optimizers, losses, last=False):
        if epoch % self.save_frequency == 0 or last:
            checkpoint = {
                "discriminator": models["discriminator"].state_dict(),
                "d_optimizer": optimizers["discriminator"].state_dict(),
                "d_lr": lr["discriminator"],
                "generator": models["generator"].state_dict(),
                "g_optimizer": optimizers["generator"].state_dict(),
                "g_lr": lr["generator"],
            }
            filename = f"ep{epoch}.tar"
            path = self.saved_models_path / filename
            torch.save(checkpoint, path)

            temp_losses = {
                "d_loss_train": torch.tensor(losses["d_loss_train"]),
                "g_loss_train": torch.tensor(losses["g_loss_train"]),
                "d_loss_val": torch.tensor(losses["d_loss_val"]),
                "g_loss_val": torch.tensor(losses["g_loss_val"]),
            }
            filename = f"losses.pt"
            path = self.checkpoint_path / filename
            torch.save(temp_losses, path)

            print(f"Checkpoint saved on: {self.checkpoint_path}.\n\tepoch: {epoch}\n\tlr: {lr}")

    def load_checkpoint(self, models:Dict, optimizers:Dict, map_location=None, epoch=None):
        if not self.checkpoint_path.exists():
            raise Exception(f"Checkpoint {self.checkpoint_path} does not exist.")

        model_config = json.load(open(self.checkpoint_path / "model_conf.json", "r"))
        train_config = json.load(open(self.checkpoint_path / "train_conf.json", "r"))

        if len(list(self.saved_models_path.iterdir())) == 0:
            last_epoch = 0
            last_lr = {
                "discriminator": train_config["start_lr"],
                "generator": train_config["start_lr"],
            }
            losses = {
                "d_loss_train": list(),
                "g_loss_train": list(),
                "d_loss_val": list(),
                "g_loss_val": list(),
            }
            return train_config, model_config, last_epoch, last_lr,losses

        if epoch is None:
            last_epoch = max([int(checkpoint.stem[2:]) for checkpoint in self.saved_models_path.iterdir()])
        else:
            last_epoch = epoch
        checkpoint = torch.load(self.saved_models_path / f"ep{last_epoch}.tar", map_location=map_location)

        models["discriminator"].load_state_dict(checkpoint["discriminator"])
        optimizers["discriminator"].load_state_dict(checkpoint["d_optimizer"])
        models["generator"].load_state_dict(checkpoint["generator"])
        optimizers["generator"].load_state_dict(checkpoint["g_optimizer"])

        try:
            current_lr = {
                "discriminator": checkpoint["d_lr"],
                "generator": checkpoint["g_lr"],
            }
        except KeyError:
            current_lr = {
                "discriminator": train_config["start_lr"],
                "generator": train_config["start_lr"],
            }

        losses = torch.load(self.checkpoint_path / "losses.pt", map_location=map_location)
        losses["d_loss_train"] = losses["d_loss_train"].tolist()
        losses["g_loss_train"] = losses["g_loss_train"].tolist()
        losses["d_loss_val"] = losses["d_loss_val"].tolist()
        losses["g_loss_val"] = losses["g_loss_val"].tolist()

        print(f"Checkpoint loaded from: {self.checkpoint_path}.\n\tepoch: {last_epoch}\n\tlr: {current_lr}")

        return train_config, model_config, last_epoch, current_lr, losses

    def load_generator(self, generator, epoch, map_location=None):
        if not self.checkpoint_path.exists():
            raise Exception(f"Checkpoint {self.checkpoint_path} does not exist.")

        model_config = json.load(open(self.checkpoint_path / "model_conf.json", "r"))

        if len(list(self.saved_models_path.iterdir())) == 0:
            return model_config

        checkpoint = torch.load(self.saved_models_path / f"ep{epoch}.tar", map_location=map_location)

        generator.load_state_dict(checkpoint["generator"])
        print(f"Generator loaded: {self.checkpoint_path}, epoch: {epoch}")

        return model_config

    def __str__(self):
        return f"Checkpoint: {self.checkpoint_path}"