import sys
import time

import cv2
import numpy as np

import openvino as ov

#model_id = 'efficient-sam-vitt'
model_id = 'efficient-sam-vits'
#model_id += '_quant'
device = 'CPU'
compiled_model = ov.compile_model(f'{model_id}.xml', device_name=device, config={'CACHE_DIR':'./cache'})

# Current mouse cursor coordinates
g_mouse_pos = (0, 0)


def postprocess_results(predicted_logits, predicted_iou):
    sorted_ids = np.argsort(-predicted_iou, axis=-1)
    predicted_iou = np.take_along_axis(predicted_iou, sorted_ids, axis=2)
    predicted_logits = np.take_along_axis(
        predicted_logits, sorted_ids[..., None, None], axis=2
    )
    return predicted_logits[0, 0, 0, :, :] >= 0

def show_mask(mask, img):
    mask_img = np.ones((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_color = (0,255,255)
    mask_img[mask] = mask_color
    alpha = 0.3
    cv2.addWeighted(mask_img, alpha, img, 1-alpha, 0, img)
    return img

def mouse_event_handler(event, x, y, flags, user):
    global g_mouse_pos
    g_mouse_pos = (x, y)


demo_name = 'EfficientSAM demo'

# Load an image
img_path = 'EfficientSAM/figs/examples/dogs.jpg'
if len(sys.argv) > 1:
    img_path = sys.argv[1]      # Use the 1st command line parameter as the input file path if it exists
image = cv2.imread(img_path)

cv2.namedWindow(demo_name)
cv2.setMouseCallback(demo_name, mouse_event_handler)


print('Hit ESC key to exit.\n')
cv2.imshow(demo_name, image)
key = 0
while key != 27:                # Exit on ESC key
    key = cv2.waitKey(30)

    # Prepare input data for inference
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose((2,0,1)) / 255.0
    input_img = np.expand_dims(input_img, axis=0)
    input_pts = np.array(g_mouse_pos).reshape((1,1,-1,2))
    input_lbl = np.array([1]).reshape((1,1,-1))
    inputs = { 'batched_images': input_img,
               'batched_points': input_pts,
               'batched_point_labels' : input_lbl
              }

    stime = time.time()
    res = compiled_model(inputs)                # Inference
    etime = time.time()
    print(f'Inference time : {etime-stime:6.2f} sec', end='\r', flush=True)

    # Post-process and drawing the mask on the input image
    predicted_mask = postprocess_results(predicted_logits=res[0], predicted_iou=res[1])
    new_img = show_mask(predicted_mask, image.copy())
    cv2.imshow(demo_name, new_img)
