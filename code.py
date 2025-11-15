# Cell: imports & device
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import copy
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# Cell: image loading, preprocessing, deprocessing, display
imsize = 512 if torch.cuda.is_available() else 256  # use smaller size on CPU

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])

unloader = transforms.ToPILImage()

def load_image(path):
    image = Image.open(path).convert('RGB')
    image = loader(image).unsqueeze(0)  # add batch dim
    return image.to(device, torch.float)

def tensor_to_pil(tensor):
    t = tensor.cpu().clone().squeeze(0)
    return unloader(t.clamp(0,1))

def show_tensor(tensor, title=None):
    plt.figure(figsize=(6,6))
    plt.imshow(tensor_to_pil(tensor))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# Cell: gram matrix
def gram_matrix(input_tensor):
    # input_tensor: batch x channels x H x W
    b, c, h, w = input_tensor.size()
    features = input_tensor.view(b * c, h * w)
    G = torch.mm(features, features.t())
    # normalize by number of elements
    return G.div(b * c * h * w)
# Cell: get VGG features at intermediate layers
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# these are the layers we use for content/style as in original paper
content_layers_default = ['21']   # conv4_2 -> layer index 21 in vgg19.features
style_layers_default = ['0','5','10','19','28']  # conv1_1,2_1,3_1,4_1,5_1

# normalize using ImageNet stats
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        # reshape so each channel can be normalized
        self.mean = mean.clone().view(-1,1,1)
        self.std = std.clone().view(-1,1,1)
    def forward(self, img):
        return (img - self.mean) / self.std
# Cell: losses (modules)
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        # detach target from graph
        self.target = target.detach()
        self.loss = 0.0
    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = 0.0
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input
# Cell: build model that inserts loss modules into VGG
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    model = nn.Sequential(normalization)
    style_losses = []
    content_losses = []

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        name = None
        if isinstance(layer, nn.Conv2d):
            name = f"{i}"
            i += 1
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i-1}"
            # replace with out-of-place ReLU to keep behavior consistent
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i-1}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i-1}"
        else:
            name = str(layer.__class__.__name__)

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_" + name, content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_" + name, style_loss)
            style_losses.append(style_loss)

    # trim off the layers after last content/style loss
    # find last loss layer index
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:i+1]

    return model, style_losses, content_losses
# Cell: prepare input image and optimizer
def get_input_optimizer(input_img):
    # input_img is a tensor we will optimize
    # LBFGS works well for this problem (original paper)
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer
# Cell: style transfer function
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1e6, content_weight=1):
    print('Building the style-transfer model...')
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing...')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0.0
            content_score = 0.0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0 or run[0] == 1:
                print("run {}:".format(run[0]))
                print('Style Loss: {:4f} Content Loss: {:4f}'.format(float(style_score), float(content_score)))

                print()

            return loss

        optimizer.step(closure)

    # final clamp
    input_img.data.clamp_(0, 1)
    return input_img
# Cell: run example (replace the paths with your images)
content_path = "/content/luffy_pic.jpg"  # <- replace with your selfie path
style_path = "/content/paper_texture.jpg"      # <- replace with your style painting

content_img = load_image(content_path)
style_img = load_image(style_path)

# initialize input as a copy of content image (or random noise)
input_img = content_img.clone()

# view input images
show_tensor(content_img, "Content Image")
show_tensor(style_img, "Style Image")

# run transfer
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                           content_img, style_img, input_img,
                           num_steps=300, style_weight=1e6, content_weight=1)

show_tensor(output, "Output Image")

# save
out_pil = tensor_to_pil(output)
out_pil.save("stylized_output.jpg")
print("Saved stylized_output.jpg")
