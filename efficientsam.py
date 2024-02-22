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

def draw_pointers(image, points, color=(0,255,255)):
    for point in points:
        image = cv2.drawMarker(image, point, color, cv2.MARKER_CROSS, 20, 2)
    return image

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

def fit_image(image, size):
    if image.shape[1]>size[0]:
        ratio = size[0]/image.shape[1]
        image = cv2.resize(image, (0,0), fx=ratio, fy=ratio)
    if image.shape[0]>size[1]:
        ratio = size[1]/image.shape[0]
        image = cv2.resize(image, (0,0), fx=ratio, fy=ratio)
    return image


def main(args):
    #model_id = 'efficient-sam-vitt'
    model_id = 'efficient-sam-vits'
    #model_id += '_quant'
    device = 'CPU'
    compiled_model = ov.compile_model(f'{model_id}.xml', device_name=device, config={'CACHE_DIR':'./cache'})

    demo_name = 'EfficientSAM demo - Clink on the image to trigger inferencing'
    cv2.namedWindow(demo_name, cv2.WINDOW_NORMAL)
    if args.full_screen:
        cv2.setWindowProperty(demo_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(demo_name, Mouse.event)

    # Load an image
    orig_image = cv2.imread(args.input)
    # Scale image size not to run off the screen
    max_image_size = (1920, 1080)
    orig_image = fit_image(orig_image, max_image_size)

    # Convert the image into input tensor
    inference_img = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).transpose((2,0,1)).astype(np.float32) / 255.0
    inference_img = np.expand_dims(inference_img, axis=0)

    key = 0
    clicked_points = []
    num_max_points = args.num_points
    num_click_count = 0

    print('Hit ESC key to exit.\n')

    disp_img = orig_image.copy()
    state = 'sampling_points'                                   # state of the program
    msg = 'Click {num} more points to start inferencing.'
    info_text = msg.format(num=num_max_points-num_click_count)  # Information text to be displayed on the image

    while key != 27:                    # Exit on ESC key
        if Mouse.is_clicked(1):         # R button check
            Mouse.clear_click_event()
            # reset the state
            num_click_count = 0
            clicked_points = []
            disp_img = orig_image.copy()
            cv2.imshow(demo_name, disp_img)
            info_text = msg.format(num=num_max_points-num_click_count)
            state = 'sampling_points'
            continue

        if state == 'inference':

            input_points = np.array(clicked_points, dtype=np.int32).reshape((1, num_click_count, 1, 2))
            input_labels = np.array([1 for _ in range(num_click_count)], dtype=np.int32).reshape((1, num_click_count, 1))
            inputs = {
                'batched_images' : inference_img,
                'batched_points' : input_points,
                'batched_point_labels' : input_labels
            }
            res = compiled_model(inputs)                # Inference

            # Post-process and drawing the mask on the input image
            mask_img = np.zeros(orig_image.shape, dtype=np.uint8)
            for i in range(num_click_count):
                predicted_logits = np.expand_dims(res[0][:,i,:,:,:], axis=1)
                predicted_iou    = np.expand_dims(res[1][:,i,:], axis=1)
                predicted_mask = postprocess_results(predicted_logits=predicted_logits, predicted_iou=predicted_iou)
                mask_img = draw_mask(predicted_mask, mask_img)
            alpha = 0.6
            disp_img = orig_image.copy()
            cv2.addWeighted(mask_img, alpha, disp_img, 1-alpha, 0, disp_img)   # alpha blending

            mask_img = draw_pointers(mask_img, clicked_points, (0,0,0))
            info_text = 'Click on the image with R button to clear the result.'
            state = 'sampling_points'

        elif state == 'sampling_points':
            if Mouse.is_clicked(0):         # L button check
                Mouse.clear_click_event()
                num_click_count += 1
                if num_click_count < num_max_points:
                    clicked_points.append(Mouse.get_click_pos())
                    disp_img = orig_image.copy()
                    disp_img = draw_pointers(disp_img, clicked_points, (0,255,255))
                    info_text = msg.format(num=num_max_points-num_click_count)
                    state = 'sampling_points'
                elif num_click_count == num_max_points:
                    clicked_points.append(Mouse.get_click_pos())
                    disp_img = orig_image.copy()
                    disp_img = draw_pointers(disp_img, clicked_points, (0,255,255))
                    info_text = 'Inferencing'
                    state = 'inference'
                else:
                    pass            # ignore the clicking
        disp_img_tmp = cv2.putText(disp_img, info_text, (0, 20), cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,0,0), thickness=2)
        disp_img_tmp = cv2.putText(disp_img, info_text, (0, 20), cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,255,255), thickness=1)
        cv2.imshow(demo_name, disp_img_tmp)
        key = cv2.waitKey(30)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EfficientSAM demo')
    parser.add_argument('-i', '--input', default='EfficientSAM/figs/examples/dogs.jpg', type=str, help='Path to an input image file')
    parser.add_argument('-n', '--num_points', default=1, type=int, help='Number of points for an inference (default:1)')
    parser.add_argument('-f', '--full_screen', action='store_true', help='Full screen mode')
    args = parser.parse_args()
    main(args)
