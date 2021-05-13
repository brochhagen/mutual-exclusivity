
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


imagenet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

object_transform = transforms.Compose([   # no crop
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_visual_model(vgg_layer):
    vgg = torchvision.models.vgg16(pretrained=True)
    vgg.eval()

    # no fine-tuning (faster)
    for param in vgg.parameters():
        param.requires_grad = False

    if vgg_layer == "fc7":
        vgg.classifier = torch.nn.Sequential(*list(vgg.classifier.children())[:-3])  
    elif vgg_layer == "fc6":
        vgg.classifier = torch.nn.Sequential(*list(vgg.classifier.children())[:-5])
    return vgg.to(device)


def get_image_representation(path, visual_model, vgg_layer):
    """ one image = one object (e.g. dax), no bboxes as in Flickr """
    preprocessed_path = path + "_preprocessed_" + vgg_layer + ".npy"
    try:
        vector = torch.tensor(np.load(preprocessed_path)).squeeze(0)
    except FileNotFoundError:
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

            vector = object_transform(img)
            vector = visual_model(vector.to(device).unsqueeze(0)).cpu().squeeze(0)
            np.save(preprocessed_path, vector.cpu().numpy())

    return vector


def get_bbox_representation(image_path, visual_model, bbox, bbox_path):
    """ try loading preprocessed bbox visual representation, if not compute it using visual model and store it """
    try:
        vector = torch.tensor(np.load(bbox_path + ".npy")).squeeze(0)
    except FileNotFoundError:
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            x, y, w, h = bbox
            vector = object_transform(img.crop((x, y, x + w, y + h)))
            vector = visual_model(vector.to(device).unsqueeze(0)).cpu().squeeze(0)
            np.save(bbox_path, vector.cpu().numpy())
    return vector


def load_image(image_path):
    with open(image_path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
    return imagenet_transform(img)
