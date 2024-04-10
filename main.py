import torch
from torch import nn
from torchvision import datasets, transforms
from torch.autograd.variable import Variable
import torchvision
import os

# Load and preprocess the celebA dataset
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
dataset = datasets.ImageFolder(
    root='/home/shivam/GANs/archive/img_align_celeba/', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the Generator and Discriminator


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 32 x 32
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. 3 x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 64 x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 256 x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 512 x 4 x 4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

    # Initialize the Generator and Discriminator
generator = Generator()
generator = generator.to(device)
discriminator = Discriminator()
discriminator = discriminator.to(device)

# Define loss function and optimizers
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)

# Before the training loop, create a fixed batch of random noise that you will use to visualize the progress of the generator
# 64 is the number of images in the batch, 100 is the dimension of the noise vector
fixed_z = torch.randn(32, 100).to(device)
fixed_z = fixed_z.view(fixed_z.size(0), 100, 1, 1)
num_epochs = 100
display_step = 10

# Training loop
for epoch in range(num_epochs):
    # Load weights if starting from previously saved weights
    if epoch == 0 and os.path.exists('generator.pth') and os.path.exists('discriminator.pth'):
        generator.load_state_dict(torch.load('generator.pth'))
        discriminator.load_state_dict(torch.load('discriminator.pth'))
    for real_images, _ in dataloader:
        # Train the Discriminator with real and fake images
        real_images = real_images.to(device)
        real_labels = torch.ones(real_images.size(0), 1).to(device)
        real_labels = real_labels.view(-1)
        fake_labels = torch.zeros(real_images.size(0), 1).to(device)
        fake_labels = fake_labels.view(-1)

        # Real images
        d_optimizer.zero_grad()
        outputs = discriminator(real_images)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()

        # Fake images
        z = torch.randn(real_images.size(0), 100).to(device)
        z = z.view(z.size(0), 100, 1, 1)
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.step()

        # Train the Generator
        g_optimizer.zero_grad()
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        g_optimizer.step()
    # Generate and save images after certain number of epochs
    if epoch % display_step == 0:  # display_step is how often you want to display images
        with torch.no_grad():
            fake_images = generator(fixed_z)
            torchvision.utils.save_image(
                fake_images, f'fake_images_{epoch}.png')

    print(
        f'Epoch [{epoch}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')
    # Save weights after each epoch
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')
