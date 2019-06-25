import torch
from torch import nn
from torch.autograd import Variable
from torchvision.transforms import transforms

from network import End_to_End


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = End_to_End()
model.to(device)


def loss_function(final_img, residual_img, upscaled_img, com_img, orig_img):
    com_loss = nn.MSELoss(size_average=False)(orig_img, final_img)
    rec_loss = nn.MSELoss(size_average=False)(residual_img, orig_img - upscaled_img)

    return com_loss + rec_loss

transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                      ])
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

def predict_image(image):
    image_tensor = transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    final, residual_img, upscaled_image, com_img, orig_im = model(input)

    return