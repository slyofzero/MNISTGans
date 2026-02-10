import torch
import torch.nn as nn
import torch.optim as optim
import os

class Discriminator(nn.Module):
  def __init__(self, img_dim):
    super().__init__()
    self.network = nn.Sequential(
        nn.Linear(img_dim, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 128),
        nn.LeakyReLU(0.2),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )

  def forward(self, X):
    return self.network(X)

class Generator(nn.Module):
  def __init__(self, noise_dim, img_dim):
    super().__init__()
    self.network = nn.Sequential(
        nn.Linear(noise_dim, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 512),
        nn.LeakyReLU(0.2),
        nn.Linear(512, img_dim),
        nn.Tanh()
    )

  def forward(self, X):
    return self.network(X)

class MNIST_cGAN():
  def __init__(self, z_dim, img_dim, targets_dim, lr=3e-4, device="cpu"):
    # # Save folder for model loading
    # self.save_folder = save_folder
    # if not os.path.isdir(self.save_folder):
    #   os.makedirs(self.save_folder)
    # self.save_name = save_name
    # self.save_path = f"{self.save_folder}/{self.save_name}.pth"

    # Model making
    features_dim = img_dim + targets_dim
    noise_dim = z_dim + targets_dim

    self.device = device

    # Networks
    self.discriminator = Discriminator(img_dim=features_dim).to(device)
    self.generator = Generator(noise_dim=noise_dim, img_dim=img_dim).to(device)

    # Optimizers
    self.optimizer_disc = optim.Adam(self.discriminator.parameters(), lr=lr)
    self.optimizer_gen = optim.Adam(self.generator.parameters(), lr=lr)

    self.criterion = nn.BCELoss()

    self.discriminator.train()
    self.generator.train()

  def load_save(self, save_folder="./mnist_checkpoints", save_name="mnist_gan"):
    if not os.path.isdir(save_folder):
      os.makedirs(save_folder)

    save_path = f"{save_folder}/{save_name}.pth"
    metadata = {}
    if not os.path.isfile(save_path):
      raise Exception(f"The mnist_gan save doesn't exist at {save_path}.")
    else:
      ckpt = torch.load(save_path, map_location=self.device)
      model, metadata = ckpt["model"], ckpt["metadata"]

      self.discriminator.load_state_dict(model["discriminator"])
      self.generator.load_state_dict(model["generator"])
      self.optimizer_disc.load_state_dict(model["optimizer_disc"])
      self.optimizer_gen.load_state_dict(model["optimizer_gen"])

      self.discriminator.to(self.device)
      self.generator.to(self.device)

    return metadata

  def save(self, metadata={}, save_path=""):
    info_to_save = {
        "model": {
          "discriminator": self.discriminator.state_dict(),
          "optimizer_disc": self.optimizer_disc.state_dict(),
          "generator": self.generator.state_dict(),
          "optimizer_gen": self.optimizer_gen.state_dict(),
        },
        "metadata": metadata
    }

    torch.save(info_to_save, save_path)

  @classmethod
  def generate_noise(cls, curr_batch_size, z_dim):
    return torch.randn((curr_batch_size, z_dim), device="cpu")

  @classmethod
  def generate_noise_with_labels(cls, curr_batch_size, z_dim, classes_count):
    noise = torch.randn((curr_batch_size, z_dim), device="cpu") # For evaluation
    noisy_classes = torch.randint(low=0, high=classes_count, size=(curr_batch_size,), device="cpu")
    noisy_classes = cls.encode_labels(noisy_classes, classes_count)
    return cls.inject_labels(noise, noisy_classes), noisy_classes

  @classmethod
  def inject_labels(cls, features, labels):
    return torch.cat([features, labels], dim=1)
  
  @classmethod
  def encode_labels(cls, labels, classes_count):
    labels = labels.long()  # enforce dtype
    assert labels.min() >= 0
    assert labels.max() < classes_count
    return nn.functional.one_hot(labels, num_classes=classes_count).float()