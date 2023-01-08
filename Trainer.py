import torch
from torch.nn.functional import binary_cross_entropy_with_logits
import torchvision
from tqdm import tqdm
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import os


class TrainerBlooprint(ABC):
    def __init__(
        self, generator, discriminator, gOptim, dOptim, loader, loss_fn, latentDim
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.gOptim = gOptim
        self.dOptim = dOptim
        self.loader = loader
        self.loss_fn = loss_fn
        self.latentDim = latentDim
        self._generator_epochs = 0
        self._discriminator_epochs = 0

    @property
    def generator_epochs(self):
        return self._generator_epochs

    @property
    def discriminator_epochs(self):
        return self._discriminator_epochs

    @property
    def batch_size(self):
        return self.loader.batch_size

    @abstractmethod
    def train_generator(self):
        pass

    @abstractmethod
    def train_discriminator(self):
        pass


class Trainer(TrainerBlooprint):
    def __init__(
        self,
        generator,
        discriminator,
        gOptim,
        dOptim,
        loader,
        latentDim,
        loss_fn=binary_cross_entropy_with_logits,
        device=None,
        is_DC=False,
    ):

        """Here we want to train gen and disc in the same loop but we perform the
        discriminator and discriminator step different numbers of times
        """
        super().__init__(
            generator, discriminator, gOptim, dOptim, loader, loss_fn, latentDim
        )

        self.device = device
        if self.device is None:
            self.device = torch.device("cpu")

        self._generator_epochs = 0
        self._discriminator_epochs = 0
        self.is_DC = is_DC
        if self.is_DC:
            self.latent_shape = (self.batch_size, latentDim, 1, 1)
        else:
            self.latent_shape = (self.batch_size, latentDim)

    @property
    def generator_epochs(self):
        return self._generator_epochs

    @property
    def discriminator_epochs(self):
        return self._discriminator_epochs

    @property
    def batch_size(self):
        return self.loader.batch_size

    def train_discriminator(self, img):
        self.dOptim.zero_grad()

        real_labels = torch.ones(img.shape[0], device=self.device, dtype=torch.float32)

        z = torch.randn(self.latent_shape, device=self.device, dtype=torch.float32)
        fake_labels = torch.zeros(
            self.latent_shape[0], device=self.device, dtype=torch.float32
        )
        with torch.no_grad():
            fake_imgs = self.generator(z)

        dloss = 0.5 * (
            self.loss_fn(self.discriminator(img), real_labels)
            + self.loss_fn(self.discriminator(fake_imgs), fake_labels)
        )
        dloss.backward()
        self.dOptim.step()
        self._discriminator_epochs += 1

    def train_generator(self, batch_size=32):
        self.gOptim.zero_grad()
        labels = torch.ones(
            self.latent_shape[0], dtype=torch.float32, device=self.device
        )
        z = torch.randn(self.latent_shape, device=self.device, dtype=torch.float32)

        loss = self.loss_fn(self.discriminator(self.generator(z)), labels)
        loss.backward()
        self.gOptim.step()
        self._generator_epochs += 1

    def generator_progress(self, n_images=20, n_row=5, transform=None):
        assert n_images % n_row == 0
        if not next(self.generator.parameters()).is_cuda:
            device = "cpu"
        else:
            device = "cuda"
        if not hasattr(self, "static_noise"):
            if self.is_DC:
                self.static_noise = torch.randn(
                    (n_images, self.latentDim, 1, 1), device=device
                )
            else:
                self.static_noise = torch.randn(
                    (n_images, self.latentDim), device=device
                )

        with torch.no_grad():
            imgs = self.generator(self.static_noise)
        if transform is not None:
            imgs = transform(imgs)
        imgs = torchvision.utils.make_grid(imgs.cpu(), nrow=n_row).permute(1, 2, 0)
        return imgs

    def train(
        self,
        n_steps,
        regime=(1, 1),
        train_reconstruction=False,
        batch_size=32,
        save_every=2000,
        transform=None,
        plot_every=500,
    ):
        """
        Args:
        n_steps: number of steps to train models
        regime: tuple of integers representing the number of
        times to perform the discriminator and generator step respectively
        """
        self.generator.train()
        self.discriminator.train()

        d_steps, g_steps = regime
        for i in tqdm(range(n_steps)):
            for _ in range(d_steps):
                img = next(iter(self.loader)).to(self.device)
                self.train_discriminator(img)
            for _ in range(g_steps):
                if train_reconstruction:
                    self.reconstruction_step(next(iter(self.loader)).to(self.device))
                self.train_generator(batch_size=batch_size)
            if (i + 1) % save_every == 0:
                self.save_models(name=f"/step_{i+1}")

            if (i + 1) % plot_every == 0 or i == 100:
                plt.close("all")
                plt.imshow(self.generator_progress(transform=transform))
                plt.show()

        self.save_models(name=f"/step_{i+1}")

    def reconstuction_step(self, img):
        if not hasattr(self, "reconstructionLoss"):
            self.reconstructionLoss = torch.nn.MSELoss()

        self.gOptim.zero_grad()
        z = torch.randn(
            (img.shape[0], self.latentDim, 1, 1),
            device=self.device,
            dtype=torch.float32,
        )
        fake_img = self.generator(z)
        loss = self.reconstructionLoss(img, fake_img)
        loss.backward()
        self.gOptim.step()

    def save_models(self, name=""):
        save_dir = "models" + name + "/"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.generator.state_dict, save_dir + f"generator.pth")
        torch.save(self.discriminator.state_dict, save_dir + f"discriminator.pth")
