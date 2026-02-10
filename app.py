import torch
import torchvision
from flask import Flask, render_template, request
from config import Config
from gan import MNIST_cGAN

app = Flask(__name__)

@app.route("/")
def home():
  return render_template(
    "index.html", 
    models=Config.models, 
    default_model=Config.default_model
  )

@app.route("/generate", methods=["POST"])
def generate():
  form_data = request.form
  digit = form_data["digit"]
  model_name = form_data["model_name"]

  generate_image(model_name, digit)

  return render_template(
    "index.html", 
    models=Config.models,
    default_model=model_name,
    digit=digit,
    showimage=True, 
    image_path="output.png"
  )

def generate_image(model_name, digit):
  if digit is None:
    return None
  digit = int(digit)

  # Parameters
  z_dim = 64
  classes_count = 10
  img_dim = 1 * 28 * 28
  batch_size = 64

  # Model
  gan = MNIST_cGAN(z_dim, img_dim, classes_count)
  gan.load_save(Config.model_save_folder, model_name)
  
  # Noise generation
  with torch.no_grad():
    gan.generator.eval()
    noise = MNIST_cGAN.generate_noise(batch_size, z_dim)
    labels = torch.full((batch_size,), digit)
    noise_labels = MNIST_cGAN.encode_labels(labels, classes_count)
    noise_injected = torch.cat([noise, noise_labels], dim=1)

    fake = gan.generator(noise_injected).reshape(-1, 1, 28, 28).float()
    fake = fake.detach().cpu()
    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
    torchvision.utils.save_image(img_grid_fake, "./static/output.png")

if __name__ == "__main__":
  app.run(debug=True)