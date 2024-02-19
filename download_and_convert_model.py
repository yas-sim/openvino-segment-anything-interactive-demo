import os
import subprocess
import urllib.request
from pathlib import Path

import zipfile
from zipfile import ZipFile
from PIL import Image
import numpy as np
import torch

import openvino as ov
import nncf

# Download (clone) EfficientSAM GitHub repository
repo_dir = 'EfficientSAM'

if not os.path.exists(repo_dir):
    subprocess.run('git clone https://github.com/yformer/EfficientSAM.git')

from EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits

os.chdir(repo_dir)
# Since EfficientSAM-S checkpoint file is >100MB, we store the zip file.
with zipfile.ZipFile('./weights/efficient_sam_vits.pt.zip', 'r') as zip_ref:
    zip_ref.extractall('./weights')

# Select a PyTorch model
MODELS_LIST = {'efficient-sam-vitt': build_efficient_sam_vitt, 
               'efficient-sam-vits': build_efficient_sam_vits}
model_id = 'efficient-sam-vits'
pt_model = MODELS_LIST[model_id]()
os.chdir('..')


# Convert the PyTorch model into OpenVINO IR model format
core = ov.Core()
ov_model_path = f'{model_id}.xml'

n=2
dummy_image = torch.from_numpy(np.random.random((1,3,768,1024)).astype(np.float32))
dummy_points = torch.from_numpy(np.zeros((1,1,n,2), dtype=np.int32))
dummy_labels = torch.from_numpy(np.ones((1,1,n), dtype=np.int32))

example_input = ( dummy_image, dummy_points, dummy_labels)

if not os.path.exists(ov_model_path):
    ov_model = ov.convert_model(pt_model, example_input=example_input)
    ov.save_model(ov_model, ov_model_path)
else:
    ov_model = core.read_model(ov_model_path)


# Download COCO128 dataset
if not os.path.exists('./coco128.zip'):
    print('Downloading COCO128 dataset.')
    urllib.request.urlretrieve('https://ultralytics.com/assets/coco128.zip', './coco128.zip')
    with ZipFile('coco128.zip' , "r") as zip_ref:
        zip_ref.extractall('.')


def prepare_input(input_image, points, labels, torch_tensor=True):
    img_tensor = np.ascontiguousarray(input_image)[None, ...].astype(np.float32) / 255
    img_tensor = np.transpose(img_tensor, (0, 3, 1, 2))
    pts_sampled = np.reshape(np.ascontiguousarray(points), [1, 1, -1, 2])
    pts_labels = np.reshape(np.ascontiguousarray(labels), [1, 1, -1])
    if torch_tensor:
        img_tensor = torch.from_numpy(img_tensor)
        pts_sampled = torch.from_numpy(pts_sampled)
        pts_labels = torch.from_numpy(pts_labels)
    return img_tensor, pts_sampled, pts_labels


class COCOLoader(torch.utils.data.Dataset):
    def __init__(self, images_path):
        self.images = list(Path(images_path).iterdir())
        self.labels_dir = images_path.parents[1] / 'labels' / images_path.name

    def get_points(self, image_path, image_width, image_height):
        file_name = image_path.name.replace('.jpg', '.txt')
        label_file =  self.labels_dir / file_name
        if not label_file.exists():
            x1, x2 = np.random.randint(low=0, high=image_width, size=(2, ))
            y1, y2 = np.random.randint(low=0, high=image_height, size=(2, ))
        else:    
            with label_file.open("r") as f:
                box_line = f.readline()
            _, x1, y1, x2, y2 = box_line.split()
            x1 = int(float(x1) * image_width)
            y1 = int(float(y1) * image_height)
            x2 = int(float(x2) * image_width)
            y2 = int(float(y2) * image_height)
        return [[x1, y1], [x2, y2]]

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path)
        image = image.convert('RGB')
        w, h = image.size
        points = self.get_points(image_path, w, h)
        labels = [1, 1] if index % 2 == 0 else [2, 3]
        batched_images, batched_points, batched_point_labels = prepare_input(image, points, labels, torch_tensor=False)
        return {'batched_images': np.ascontiguousarray(batched_images)[0], 'batched_points': np.ascontiguousarray(batched_points)[0], 'batched_point_labels': np.ascontiguousarray(batched_point_labels)[0]}
    
    def __len__(self):
        return len(self.images)

coco_dataset = COCOLoader(Path('./coco128/images/train2017'))
calibration_loader = torch.utils.data.DataLoader(coco_dataset)

calibration_dataset = nncf.Dataset(calibration_loader)

quantized_model = nncf.quantize(ov_model,
                                calibration_dataset,
                                model_type=nncf.parameters.ModelType.TRANSFORMER,
                                preset=nncf.common.quantization.structs.QuantizationPreset.MIXED, subset_size=128)

ov_model_path_quant = f'{model_id}_quant.xml'
ov.save_model(quantized_model, ov_model_path_quant)

print("model quantization finished")

