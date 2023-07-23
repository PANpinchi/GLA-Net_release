import os.path
from collections import OrderedDict
import torch
import torch.nn as nn
import cv2
from torchvision import transforms
import utils_image as util
import warnings

warnings.filterwarnings("ignore")


def main():

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    model = torch.load('model.pth').cuda()

    model.eval()

    with torch.no_grad():
        for i in range(2):

            torch.cuda.empty_cache()

            if i == 0:
                img = cv2.imread('CU.png')
            else:
                img = cv2.imread('test.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform(img).cuda()
            img = torch.unsqueeze(img, 0)

            data_SR = model(img)
            H, W = data_SR.shape[2], data_SR.shape[3]
            resize = transforms.Resize([H, W])
            data_SR = data_SR + resize(img)

            out_dict = OrderedDict()
            out_dict['E'] = data_SR.detach()[0].float().cpu()
            visuals = out_dict

            E_img = util.tensor2uint(visuals['E'])

            save_img_path = os.path.join('.', 'test.png')
            util.imsave(E_img, save_img_path)


if __name__ == '__main__':
    main()
