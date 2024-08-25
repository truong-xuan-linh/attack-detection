import torch
from torchvision import transforms
import numpy as np

class ResNet:
    def __init__(self) -> None:
        self.model = torch.load("models/resnet_nightshade.pt", map_location=torch.device('cpu'))
        self.preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        self.model.eval()
        self.class_mapper = {
            0: "normal",
            1: "attacked"
        }
    def predict(self, image):
        image_tensor = self.preprocess(image)
        image_tensor = image_tensor.unsqueeze(0)
        with torch.no_grad():
            output = self.model(image_tensor)
            index = np.argmax(output.cpu().numpy(), axis=-1)
        return self.class_mapper[index[0]]

        