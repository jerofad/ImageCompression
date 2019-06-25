import torch
from torch import optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from network import End_to_End
from utils import loss_function, save_image

CHANNELS = 3
HEIGHT = 96
WIDTH = 96
EPOCHS = 200
LOG_INTERVAL = 50
BATCH_SIZE = 16

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


img_transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = datasets.STL10(root='./data', split='train',download=True, transform=img_transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

testset = datasets.STL10(root='./data', split='test',download=True, transform=img_transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

model = End_to_End()
model.to(device)


optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        optimizer.zero_grad()
        final, residual_img, upscaled_image, com_img, orig_im = model(data.to(device))
        loss = loss_function(final, residual_img, upscaled_image, com_img, orig_im)
        loss.backward()
        train_loss += loss.data.item()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for i, (data, _) in enumerate(test_loader):
            data = Variable(data, volatile=True)
            final, residual_img, upscaled_image, com_img, orig_im = model(data.cuda())
            test_loss += loss_function(final, residual_img, upscaled_image, com_img, orig_im).data.item()
            if epoch == EPOCHS and i == 0:
                #             save_image(final.data[0],'reconstruction_final',nrow=8)
                #             save_image(com_img.data[0],'com_img',nrow=8)
                n = min(data.size(0), 6)
                print("saving the image " + str(n))
                comparison = torch.cat([data[:n],
                                        final[:n].cpu()])
                comparison = comparison.cpu()
                #             print(comparison.data)
                save_image(com_img[:n].data,
                           'compressed_' + str(epoch) + '.png', nrow=n)
                save_image(comparison.data,
                           'reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, EPOCHS+1):
    train(epoch)
    test(epoch)

# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')
torch.save(model.state_dict(), './model.pth')
