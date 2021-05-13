
import torch
import torchvision

class SymbolicVisualExtractor(torch.nn.Module):
    """ A "visual" encoder for symbolic data, that is, simply a vector embedding of object ids """

    def __init__(self, vocab_size, hidden_size, fixed_weights=False, init_range=1.0):
        super(SymbolicVisualExtractor, self).__init__()

        self.visual_encoder = torch.nn.Embedding(vocab_size, hidden_size)
        if init_range:
            self.visual_encoder.weight.data.uniform_(-init_range, init_range)

        self.nobjects = vocab_size

        if fixed_weights:
            for param in self.parameters():
                 param.requires_grad = False

    def forward(self, v):
        v = self.visual_encoder(v)
        return v


class VGGVisualExtractor(torch.nn.Module):
    """ A visual encoder using pre-trained VGG16 model """

    def __init__(self, vocab_size, hidden_size, fixed_weights=False):
        super(VGGVisualExtractor, self).__init__()

        vgg = torchvision.models.vgg16(pretrained=True)
        vgg.eval()
        self.nobjects = None

        # no fine-tuning (faster)
        for param in vgg.parameters():
            param.requires_grad = False

        visual_model = vgg
        classifier = torch.nn.Sequential(*list(visual_model.classifier.children())[:-2])
        visual_model.classifier = classifier

        self.conv_features = visual_model

        self.visual_encoder = torch.nn.Linear(4096, hidden_size, bias=False)

        if fixed_weights:
            for param in self.parameters():
                 param.requires_grad = False
        self.nobjects = None

    def forward(self, v):
        v = self.conv_features(v)
        v = self.visual_encoder(v)
        return v


class PreprocessedExtractor(torch.nn.Module):
    """ A visual encoder which takes outputs of VGG16 computed previously for each object (fixed)
        and adds a trainable layer on top """

    def __init__(self, hidden_size, fixed_weights=False, vgg_layer='lastlayer'):
        super(PreprocessedExtractor, self).__init__()

        # TODO: 1000 is the size of the last layer of VGG16, if we take other layers this should be e.g. 4096
        if vgg_layer == "lastlayer":
            self.visual_encoder = torch.nn.Linear(1000, hidden_size, bias=False)
        else:
            self.visual_encoder = torch.nn.Linear(4096, hidden_size, bias=False)

        if fixed_weights:
            for param in self.parameters():
                 param.requires_grad = False

    def forward(self, v):
        v = self.visual_encoder(v)
        return v
