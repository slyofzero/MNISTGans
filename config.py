class Config:
    model_save_folder = "./models"
    default_model = "mnist_cgan_v2_e60"
    models = {
        "mnist_cgan_v2_e10": {
            "type": "cGAN",
            "description": "Conditional GAN trained for 10 epochs"
        },
        "mnist_cgan_v2_e20": {
            "type": "cGAN",
            "description": "Conditional GAN trained for 20 epochs"
        },
        "mnist_cgan_v2_e30": {
            "type": "cGAN",
            "description": "Conditional GAN trained for 30 epochs"
        },
        "mnist_cgan_v2_e40": {
            "type": "cGAN",
            "description": "Conditional GAN trained for 40 epochs"
        },
        "mnist_cgan_v2_e50": {
            "type": "cGAN",
            "description": "Conditional GAN trained for 50 epochs"
        },
        "mnist_cgan_v2_e60": {
            "type": "cGAN",
            "description": "Conditional GAN trained for 60 epochs"
        },
        "mnist_cgan_v2_e70": {
            "type": "cGAN",
            "description": "Conditional GAN trained for 70 epochs"
        },
        "mnist_cgan_v2_e80": {
            "type": "cGAN",
            "description": "Conditional GAN trained for 80 epochs"
        },
        "mnist_cgan_v2_e90": {
            "type": "cGAN",
            "description": "Conditional GAN trained for 90 epochs"
        },
        "mnist_cgan_v2_e100": {
            "type": "cGAN",
            "description": "Conditional GAN trained for 100 epochs"
        },
    }
    