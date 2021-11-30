import torchvision.transforms as transforms
import torch
import numpy as np
import pickle
from skimage import io
import cv2

from pytorch_grad_cam.utils.image import show_cam_on_image

import matplotlib.pyplot as plt

transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

with open('breat_tumor_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('gradcam.pkl', 'rb') as f:
    gradcamplusplus = pickle.load(f)


def get_gradcam_image(image, target_categ, gradcam):
    rgb_img = ((image - image.min()) / (image.max() - image.min()))
    input_tensor = torch.tensor(rgb_img).permute(2, 0, 1).unsqueeze(0)

    grayscale_cam = gradcam(input_tensor=input_tensor, target_category=target_categ)

    # Overlay heatmap on an image
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return visualization
