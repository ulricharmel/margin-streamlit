import os
import net
import utils
import numpy as np
import torch
from torch.autograd import Variable

from PIL import Image
import torchvision.transforms as transforms

eval_transformer = transforms.Compose([
    transforms.Resize((64, 64)),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.ToTensor()])  # transform it into a torch tensor

# Define the model
model = net.ResNet18().cuda() if torch.cuda.is_available() else net.ResNet18()

current_path = os.getcwd()

saveweights = os.path.join(current_path, "static/best.pth.tar")

# Reload weights from the saved file
utils.load_checkpoint(saveweights, model)

model.eval()

def predictor(imagepath, threshold=0.5): 

    image = Image.open(imagepath).convert("RGB")

    image = torch.reshape(eval_transformer(image), (1,3,64,64))

    out = model(image).detach().cpu().numpy()

    if out>threshold:
        prediction = "bad conditions ahead"
    else:
        prediction = "good conditions ahead"
    
    prob = int(100.*np.abs(threshold-out)/threshold)
    
    

    return prediction, prob


