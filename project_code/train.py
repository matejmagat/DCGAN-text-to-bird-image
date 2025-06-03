from pathlib import Path

from torch.optim.lr_scheduler import ReduceLROnPlateau

from DCGAN import *
from checkpoint import Checkpoint
from cub_dataset import get_cub_dataloader
from tqdm import tqdm
from image_transform import get_inv_image_transform

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
    "image_root":'/home/matej/Documents/DIPLOMSKI/2 SEMESTAR/SEMINAR/dataset/CUB/images',
    "embeddings_root":'/home/matej/Documents/DIPLOMSKI/2 SEMESTAR/SEMINAR/dataset/CUB/embeddings',
    "data_split": (0.7, 0.15, 0.15),
    "batch_size" : 256,
    "start_lr" : 0.02, #0.0002
    "patience": 5,
    "factor": 0.1,
    "num_epochs" : 2500,
    "l1_coef" : 50,
    "l2_coef" : 100,
    "real_label" : 1.,
    "fake_label" : 0.,
    "seed" : 1234,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}


class TrainingManager:
    def __init__(self, checkpoint: Checkpoint, last_epoch=None):
        self.D = Discriminator()
        self.D.apply(weights_init)

        self.G = Generator()
        self.G.apply(weights_init)

        self.optim_D = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optim_G = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.criterion = nn.BCELoss()
        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.D.to(self.device)
        self.G.to(self.device)

        train_config, model_config, \
            last_epoch, current_lr, \
            losses = checkpoint.load_checkpoint({"discriminator": self.D, "generator": self.G},
                                                {"discriminator": self.optim_D,"generator": self.optim_G},
                                                                                    map_location=self.device,
                                                epoch=last_epoch)

        self.real_label = train_config["real_label"]
        self.fake_label = train_config["fake_label"]
        self.lr = current_lr
        self.l1_coef = train_config["l1_coef"]
        self.l2_coef = train_config["l2_coef"]

        for param_group in self.optim_D.param_groups:
            param_group['lr'] = self.lr["discriminator"]

        for param_group in self.optim_G.param_groups:
            param_group['lr'] = self.lr["generator"]

        dataloaders = get_cub_dataloader(train_config["image_root"], train_config["embeddings_root"],
                                         train_config["batch_size"],
                           split=train_config["data_split"], seed=train_config["seed"],
                           image_size=model_config["image_size"], num_workers=0, device=self.device)

        self.train_loader = dataloaders["train"]
        self.val_loader = dataloaders["val"]
        self.test_loader = dataloaders["test"]

        self.train_config = train_config
        self.model_config = model_config
        self.last_epoch = last_epoch
        self.losses = losses

        self.checkpoint = checkpoint
        print(self.train_config)

        self.scheduler_D = ReduceLROnPlateau(
            self.optim_D,
            mode='min',
            factor=train_config["factor"],
            patience=train_config["patience"],
            min_lr=1e-6
        )

        self.scheduler_G = ReduceLROnPlateau(
            self.optim_G,
            mode='min',
            factor=train_config["factor"],
            patience=train_config["patience"],
            min_lr=1e-6
        )

    def train_discriminator(self, images, embeddings, wrong_images, backprop=True):
        batch_size = images.shape[0]
        self.optim_D.zero_grad()
        noise = torch.randn(batch_size, self.model_config["noise_size"]).to(self.device)
        fake_images = self.G(embeddings, noise)
        real_out, real_act = self.D(images, embeddings)
        d_loss_real = self.criterion(real_out, torch.full_like(real_out, self.real_label))
        wrong_out, wrong_act = self.D(wrong_images, embeddings)
        d_loss_wrong = self.criterion(wrong_out, torch.full_like(wrong_out, self.fake_label))
        fake_out, fake_act = self.D(fake_images.detach(), embeddings)
        d_loss_fake = self.criterion(fake_out, torch.full_like(fake_out, self.fake_label))
        d_loss = d_loss_real + d_loss_wrong + d_loss_fake

        if backprop:
            d_loss.backward()
            self.optim_D.step()
            self.losses["d_loss_train"].append(d_loss.item())

        return d_loss.item()

    def train_generator(self, images, embeddings, backprop=True):
        batch_size = images.shape[0]
        self.optim_G.zero_grad()
        noise = torch.randn(batch_size, self.model_config["noise_size"]).to(self.device)
        fake_images = self.G(embeddings, noise)
        out_fake, act_fake = self.D(fake_images, embeddings)
        out_real, act_real = self.D(images, embeddings)
        g_bce = self.criterion(out_fake, torch.full_like(out_fake, self.real_label))
        g_l1 = self.l1_coef * self.l1_loss(fake_images, images)
        g_l2 = self.l2_coef * self.l2_loss(torch.mean(act_fake, 0), torch.mean(act_real, 0).detach())
        g_loss = g_bce + g_l1 + g_l2

        if backprop:
            g_loss.backward()
            self.optim_G.step()
            self.losses["g_loss_train"].append(g_loss.item())
            return g_loss.item()

        return g_loss.item(), fake_images.detach()

    def train(self):
        print(f"Training started: \n\tfor {self.train_config['num_epochs']} epochs \n\twith lr: {self.lr}\n\ton {self.device}")
        last = False
        for epoch in range(self.last_epoch+1, self.train_config["num_epochs"]+1):
            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch}/{self.train_config['num_epochs']}",
                      unit="batch") as pbar:
                for images, embeddings, wrong_images in self.train_loader:
                    images = images.to(self.device)
                    embeddings = embeddings.to(self.device)
                    wrong_images = wrong_images.to(self.device)

                    d_loss = self.train_discriminator(images, embeddings, wrong_images)

                    g_loss = self.train_generator(images, embeddings)

                    pbar.set_postfix({
                        "d_loss": f"{d_loss:.2f}",
                        "g_loss": f"{g_loss:.2f}"
                    })
                    pbar.update(1)

            d_val_error, g_val_error = self.validate()
            if d_val_error is not None and g_val_error is not None:
                old_d_lr = self.optim_D.param_groups[0]['lr']
                self.scheduler_D.step(d_val_error)
                new_d_lr = self.optim_D.param_groups[0]['lr']

                old_g_lr = self.optim_G.param_groups[0]['lr']
                self.scheduler_G.step(g_val_error)
                new_g_lr = self.optim_G.param_groups[0]['lr']

                if new_d_lr != old_d_lr:
                    print(f"Discriminator lr reduced from {old_d_lr:.6f} to {new_d_lr:.6f}")

                if new_g_lr != old_g_lr:
                    print(f"Generator lr reduced from {old_g_lr:.6f} to {new_g_lr:.6f}")

                self.lr = {
                    "discriminator": new_d_lr,
                    "generator": new_g_lr,
                }

            if epoch == self.train_config["num_epochs"]:
                last = True

            self.checkpoint(epoch, self.lr,
                            {"discriminator": self.D, "generator": self.G},
                            {"discriminator": self.optim_D, "generator": self.optim_G},
                            self.losses, last=last)

        return self.losses

    def validate(self):
        if self.val_loader is None:
            return None, None

        d_error = 0.0
        g_error = 0.0
        with torch.no_grad():
            for images, embeddings, wrong_images in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                embeddings = embeddings.to(self.device)
                wrong_images = wrong_images.to(self.device)

                d_loss = self.train_discriminator(images, embeddings, wrong_images, backprop=False)
                g_loss, fake_images = self.train_generator(images, embeddings, backprop=False)
                d_error += d_loss
                g_error += g_loss


        d_error /= len(self.val_loader)
        g_error /= len(self.val_loader)
        print(f"Discriminator val error: {d_error:.2f}")
        print(f"Generator val error: {g_error:.2f}")
        self.losses["d_loss_val"].append(d_error)
        self.losses["g_loss_val"].append(g_error)

        return d_error, g_error

    def test(self, image_dump=None):
        print(f"Testing started on model trained for {self.last_epoch} epochs")
        d_error = 0.0
        g_error = 0.0
        batch_idx = 0
        with torch.no_grad():
            for images, embeddings, wrong_images in tqdm(self.test_loader, desc="Testing"):
                batch_size = images.shape[0]
                images = images.to(self.device)
                embeddings = embeddings.to(self.device)
                wrong_images = wrong_images.to(self.device)

                d_loss = self.train_discriminator(images, embeddings, wrong_images, backprop=False)
                g_loss, fake_images = self.train_generator(images, embeddings, backprop=False)
                d_error += d_loss
                g_error += g_loss

                if image_dump is not None:
                    for j in range(batch_size):
                        get_inv_image_transform()(images[j]).save(Path(image_dump) / f"real/{batch_idx}_{j}.png")
                        get_inv_image_transform()(fake_images[j]).save(Path(image_dump) / f"fake/{batch_idx}_{j}.png")

        d_error /= len(self.test_loader)
        g_error /= len(self.test_loader)
        print(f"Discriminator test error: {d_error:.2f}")
        print(f"Generator test error: {g_error:.2f}")
        return d_error, g_error





