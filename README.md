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
python efficientsam.py -i <image_file_name>
```

Command line options:
```sh
options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to an input image file
  -n NUM_POINTS, --num_points NUM_POINTS
                        Number of points for an inference (default:1)
  -f, --full_screen     Full screen mode
```
## Demo  
`python .\efficientsam.py -i .\trolley.jpg -n 3 -f`
![demo](./resources/demo2.gif)

## Test environment  
- Windows 11
- Python 3.10.3
- OpenVINO 2023.3.0
