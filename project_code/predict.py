
from pathlib import Path

import torch

from checkpoint import Checkpoint
from DCGAN import Generator
from bert_encoder import BERTWrapper
from image_transform import get_inv_image_transform


if __name__ == '__main__':
    model_path = Path("./checkpoints/train1/saved_models/ep740.tar")
    prompt = "the medium sized bird has a dark grey color, a black downward curved beak, and long wings."

    checkpoint_path = model_path.parent.parent
    checkpoint = Checkpoint(checkpoint_path, 1)

    model = Generator()
    model.eval()
    epoch = int(model_path.stem[2::])
    model_config = checkpoint.load_generator(model, epoch, map_location=torch.device('cpu'))

    embedding = BERTWrapper()(prompt).unsqueeze(0)
    noise = torch.randn(1, model_config["noise_size"])

    image = get_inv_image_transform()(model(embedding, noise)[0])
    image.save("train.png")
    print("Image generated")
