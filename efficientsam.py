import random
import argparse

import cv2
import numpy as np

import openvino as ov

def postprocess_results(predicted_logits, predicted_iou):
    sorted_ids = np.argsort(-predicted_iou, axis=-1)
    predicted_iou    = np.take_along_axis(predicted_iou   , sorted_ids                 , axis=2)
    predicted_logits = np.take_along_axis(predicted_logits, sorted_ids[..., None, None], axis=2)
    return predicted_logits[0, 0, 0, :, :] >= 0

def draw_mask(mask, mask_img):
    mask_color = [ random.randint(128, 255) for _ in range(3) ]
    mask_img[mask] = mask_color
    return mask_img

class Mouse:
    curr_pos = (0,0)
    click_pos = (0,0)
    clicked_l = False
    clicked_r = False

    def __init__(self):
        Mouse.curr_pos = (0,0)
        Mouse.click_pos = (0,0)
        Mouse.clicked = False

    def event(event, x, y, flags, user):
        Mouse.curr_pos = (x, y)
        if event==cv2.EVENT_LBUTTONUP:        # Mouse L button release event
            Mouse.click_pos = (x, y)
            Mouse.clicked_l = True
        if event==cv2.EVENT_RBUTTONUP:        # Mouse R button release event
            Mouse.click_pos = (x, y)
            Mouse.clicked_r = True
    
    def clear_click_event():
        Mouse.clicked_l = False
        Mouse.clicked_r = False
    
    def is_clicked(button):
        if button == 0:
            return Mouse.clicked_l
        elif button == 1:
            return Mouse.clicked_r
        return False
    
    def get_click_pos():
        return Mouse.click_pos


def main(args):
    #model_id = 'efficient-sam-vitt'
    model_id = 'efficient-sam-vits'
    #model_id += '_quant'
    device = 'CPU'
    compiled_model = ov.compile_model(f'{model_id}.xml', device_name=device, config={'CACHE_DIR':'./cache'})

    demo_name = 'EfficientSAM demo - Clink on the image to trigger inferencing'
    cv2.namedWindow(demo_name)
    cv2.setMouseCallback(demo_name, Mouse.event)

    # Load an image
    image = cv2.imread(args.input)
    # Scale image size not to run off the screen
    max_image_size = (1920, 1080)
    if image.shape[1]>max_image_size[0]:
        ratio = max_image_size[0]/image.shape[1]
        image = cv2.resize(image, (0,0), fx=ratio, fy=ratio)
    if image.shape[0]>max_image_size[1]:
        ratio = max_image_size[1]/image.shape[0]
        image = cv2.resize(image, (0,0), fx=ratio, fy=ratio)

    # Convert the image into input tensor
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose((2,0,1)) / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    key = 0
    clicked_points = []
    num_max_points = args.num_points
    num_click_count = 0

    msg = 'Click {num} more points to start inferencing.'
    print('Hit ESC key to exit.\n')
    print(msg.format(num=num_max_points-num_click_count))
    cv2.imshow(demo_name, image)

    while key != 27:                # Exit on ESC key
        key = cv2.waitKey(30)

        if Mouse.is_clicked(1):         # R button check
            Mouse.clear_click_event()
            # reset the state
            num_click_count = 0
            clicked_points = []
            cv2.imshow(demo_name, image)
            print(msg.format(num=num_max_points-num_click_count))
            continue

        if Mouse.is_clicked(0):         # L button check
            Mouse.clear_click_event()
            num_click_count += 1
            if num_click_count > num_max_points:
                continue
            clicked_points.append(Mouse.get_click_pos())
            if num_click_count == num_max_points:
                draw_img = image.copy()
                print('*** Inferencing.')
                input_points = np.array(clicked_points, dtype=np.int32).reshape((1, num_click_count, 1, 2))
                input_labels = np.array([1 for _ in range(num_click_count)], dtype=np.int32).reshape((1, num_click_count, 1))
                inputs = {
                    'batched_images' : input_img,
                    'batched_points' : input_points,
                    'batched_point_labels' : input_labels
                }
                res = compiled_model(inputs)                # Inference

                # Post-process and drawing the mask on the input image
                mask_img = np.zeros(image.shape, dtype=np.uint8)
                for i in range(num_click_count):
                    predicted_logits = np.expand_dims(res[0][:,i,:,:,:], axis=1)
                    predicted_iou    = np.expand_dims(res[1][:,i,:], axis=1)
                    predicted_mask = postprocess_results(
                        predicted_logits=predicted_logits, 
                        predicted_iou=predicted_iou)
                    mask_img = draw_mask(predicted_mask, mask_img)
                for i in range(num_click_count):
                    mask_img = cv2.drawMarker(mask_img, clicked_points[i], (0,0,0), cv2.MARKER_CROSS, 20, 2)
                alpha = 0.6
                cv2.addWeighted(mask_img, alpha, draw_img, 1-alpha, 0, draw_img)   # alpha blending
                cv2.imshow(demo_name, draw_img)

                print('Click on the image with R button to clear the result.')
            else:
                print(msg.format(num=num_max_points-num_click_count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EfficientSAM demo')
    parser.add_argument('-i', '--input', default='EfficientSAM/figs/examples/dogs.jpg', type=str, help='Path to an input image file')
    parser.add_argument('-n', '--num_points', default=1, type=int, help='Number of points for an inference')
    args = parser.parse_args()
    main(args)
