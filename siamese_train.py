import os
import time

from scipy.stats import multivariate_normal

from siamese_dataloader import *
from siamese_net import *


"""
Get training data
"""


class Config:
    training_dir = "drones_data/cropped_train/"
    testing_dir = "drones_data/cropped_test/"
    train_batch_size = 128
    train_number_epochs = 100


folder_dataset = dset.ImageFolder(root=Config.training_dir)

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 128)),
    torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
    torchvision.transforms.ToTensor()
])


def get_gaussian_mask():
    # 128 is image size
    # We will be using 256x128 patch instead of original 128x128 path because we are using for pedestrain with 1:2 AR.
    x, y = np.mgrid[0:1.0:256j, 0:1.0:128j]  # 128 is input size.
    xy = np.column_stack([x.flat, y.flat])
    mu = np.array([0.5, 0.5])
    sigma = np.array([0.22, 0.22])
    covariance = np.diag(sigma ** 2)
    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
    z = z.reshape(x.shape)

    z = z / z.max()
    z = z.astype(np.float32)

    mask = torch.from_numpy(z)

    return mask


# Get dataparser class object
siamese_dataset = SiameseTriplet(imageFolderDataset=folder_dataset, transform=transforms, should_invert=False)
net = SiameseNetwork().cuda()  # Get model class object and put the model on GPU

criterion = TripletLoss(margin=1)
optimizer = optim.Adam(net.parameters(), lr=0.0005)  # changed from 0.0005

print("GPU compute available: ", torch.cuda.is_available())

counter = []
loss_history = []
iteration_number = 0

# PyTorch data parser obeject
train_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=8, batch_size=Config.train_batch_size)

# Multiply each image with mask to give attention to center of the image.
gaussian_mask = get_gaussian_mask().cuda()

for epoch in range(0, Config.train_number_epochs):
    print(len(train_dataloader))
    start = time.time()
    for i, data in enumerate(train_dataloader, 0):

        # Get anchor, positive and negative samples
        anchor, positive, negative = data
        anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()

        # Multiple image patches with gaussian mask. It will act as an attention mechanism which will focus on the
        # center of the patch
        anchor, positive, negative = anchor * gaussian_mask, positive * gaussian_mask, negative * gaussian_mask

        optimizer.zero_grad()  # Reset the optimizer gradients to zero

        anchor_out, positive_out, negative_out = net(anchor, positive, negative)  # Model forward propagation

        # Compute triplet loss (based on cosine simality) on the output feature maps
        triplet_loss = criterion(anchor_out, positive_out, negative_out)
        # Backward propagation to get the gradients w.r.t. model weights
        triplet_loss.backward()
        # Model weights updation using calculated gradient and Adam optimizer
        optimizer.step()

        if i % 10 == 0:  # Print logs after every 10 iterations
            # TODO: Update with tqdm based log printing for better training monitoring
            print("Epoch number {}\n Current loss {}\n".format(epoch, triplet_loss.item()))
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(triplet_loss.item())
    print(time.time() - start)
    if epoch % 2 == 0:  # Model will be saved after every 2 epochs
        if not os.path.exists('ckpts/'):
            os.mkdir('ckpts')
        torch.save(net, 'ckpts/model_channel4_' + str(epoch) + '.pt')

# Loss curve plot to be dumped after full model training. 
show_plot(counter, loss_history, path='ckpts/loss_channel4.png')
