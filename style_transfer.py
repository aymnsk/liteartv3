
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from utils import apply_dreamy_effects
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (512, 512)

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)
    def forward(self, img):
        return (img - self.mean) / self.std

def gram_matrix(input_tensor):
    a, b, c, d = input_tensor.size()
    features = input_tensor.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
        self.loss = 0
    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = 0
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

def run_style_transfer(content_file, style_file, effect_mode="none"):
    content_img = Image.open(content_file).convert("RGB")
    style_img = Image.open(style_file).convert("RGB")

    content_tensor = transform(content_img).unsqueeze(0).to(device)
    style_tensor = transform(style_img).unsqueeze(0).to(device)

    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    cnn_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    normalization = Normalization(cnn_mean, cnn_std).to(device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)

    i = 0
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_tensor).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_tensor).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], ContentLoss) or isinstance(model[j], StyleLoss):
            break
    model = model[:j+1]

    input_img = content_tensor.clone().requires_grad_(True)
    optimizer = optim.LBFGS([input_img])
    run = [0]

    while run[0] <= 150:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_score * 1e6 + content_score
            loss.backward()
            run[0] += 1
            return loss
        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)
    save_image(input_img, "stylized_output.png")

    if effect_mode != "none":
        final_img = apply_dreamy_effects("stylized_output.png", effect_mode)
        final_img.save("final_output.png")
        return "final_output.png"
    else:
        return "stylized_output.png"
