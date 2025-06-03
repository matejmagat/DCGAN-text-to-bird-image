import torch
from torch import nn

# copied from:
# https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)



class Generator(nn.Module):
    def __init__(self, nx=768, embedding_out=128, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()

        self.projection = nn.Sequential(
            # from https://github.com/yashashwita20/text-to-image-using-GAN/blob/main/models/dcgan_model.py
            nn.Linear(nx, embedding_out),
            nn.BatchNorm1d(embedding_out),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(embedding_out + nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, embedding, z):
        embedding = self.projection(embedding) # [batch, 768] -> [batch, 128]
        x = torch.cat((embedding, z), dim=1) # z = [batch, 100]: x = [batch, 228]
        x = x.view(x.shape[0], x.shape[1], 1, 1) # expand into a 4d tensor

        return self.main(x)
    

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, embedding_in=768, embedding_out=128):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``

        )

        self.projection = nn.Sequential(
            # from https://github.com/yashashwita20/text-to-image-using-GAN/blob/main/models/dcgan_model.py
            nn.Linear(embedding_in, embedding_out),
            nn.BatchNorm1d(embedding_out),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.output = nn.Sequential(
            nn.Conv2d(ndf * 8 + embedding_out, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, embedding):
        x_out = self.main(x)
        embedding = self.projection(embedding)
        embedding = embedding.repeat(4, 4, 1, 1).permute(2, 3, 0, 1)
        out = torch.cat([x_out, embedding], 1)
        return self.output(out).squeeze(), x_out



if __name__ == '__main__':
    image = torch.rand(64,3,64,64)
    embedding = torch.randn(64,768)
    z = torch.randn(64,100)
    generator = Generator()
    discriminator = Discriminator()

    fake = generator(embedding, z)
    prob, x_out = discriminator(fake, embedding)
    print(fake.shape, x_out.shape)
    print(prob.shape)


