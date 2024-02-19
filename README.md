# Segment Anything Model (SAM) interactive demo with OpenVINO

## Description
This demo takes an image file and runs a SAM model using OpenVINO.  
You can point to the input image with the mouse, and the demo program shows the segmentation result on the image.  
This demo supports following models from [yformer/EfficientSAM GitHub repo](https://github.com/yformer/EfficientSAM.git):
- efficient-sam-vitt
- efficient-sam-vits

## Programs / Files
|#|File name|Description|
|---|---|---|
|1|download_and_convert.py|Clone the EfficientSAM GitHub repo and convert the model into OpenVINO IR model.|
|2|efficientsam.py|Run the EfficientSAM interactive demo with OpenVINO|

## How to run

1. Install prerequisites

```sh
python -m venv venv
venv/Scripts/activate
python -m pip install -U pip
pip install -U setuptools wheel
pip install -r requirements.txt
```

2. Download the model
```sh
python download_and_convert.py
```

3. Run the demo
Point and click on the image to kick inferencing.
```sh
python efficientsam.py
  or
python efficientsam.py <image_file_name>
```

![demo](./resources/demo.gif)
## Test environment
- Windows 11
- Python 3.10.3
- OpenVINO 2023.3.0
