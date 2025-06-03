from torchvision import transforms

def get_image_transform(image_size=64):
    return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

def get_inv_image_transform():
    return transforms.Compose([
        transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0]),
        transforms.ToPILImage()
    ])
