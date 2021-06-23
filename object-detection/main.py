from PIL.Image import NONE
import numpy as np
import cv2
import time
import argparse
import os
import tensorflow as tf
import warnings
import json

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from tensorflow.python.util.deprecation import silence


warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

SAMPLING = 10  # Classify every n frames (use tracking in between)
CONFIDENCE = 0.65  # Confidence threshold to filter iffy objects


# PATH_TO_MODEL_DIR = "/home/kora/Documents/RI/MATF-RI/object-detection/exported_models/ssd_mobilenet_v2_fpnlite_640x640_retrained/"
# PATH_TO_LABELS = "/home/kora/Documents/RI/MATF-RI/object-detection/annotations/label_map.pbtxt"
# cap = cv2.VideoCapture("/home/kora/Documents/RI/object-detection/src/video/1.mp4")

PATH_TO_MODEL_DIR = ""
PATH_TO_LABELS = ""
cap = None
RESOLUTION = {
    "width": 0,
    "height": 0,
}

def var_init():
    global SAMPLING, CONFIDENCE, cap
    # Initiate the parser
    parser = argparse.ArgumentParser(description="Start the model for counting vehicles on the road.")
    # Add different arguments
    parser.add_argument(
        "--model-path", "-mp", required=True,
        help="Path to the model.", type=str)
    parser.add_argument(
        "--label-path", "-lp", required=True,
        help="Path to the labels.\t ./*.pbtxt", type=str)
    parser.add_argument(
        "--source-video-path", "-sp", required=True,
        help="Path to the video source.", type=str)
    parser.add_argument(
        "--output-video-path", "-op", required=False,
        help="Path to the video output with all the labels.", type=str)
    parser.add_argument(
        "--sampling-rate", "-sr", required=False,
        help="Frame processing rate, could be set lower on faster GPUs. Default value: " + str(SAMPLING), type=int)
    parser.add_argument(
        "--confidence", "-c", required=False,
        help="Confidence of the objects that will be counted. Default value: " + str(CONFIDENCE), type=float)

    args = parser.parse_args()    
    
    

    # Open the source video and set the resolution
    cap = cv2.VideoCapture(args.source_video_path)
    global RESOLUTION
    RESOLUTION["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    RESOLUTION["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    global PATH_TO_MODEL_DIR, PATH_TO_LABELS
    PATH_TO_MODEL_DIR = args.model_path
    PATH_TO_LABELS = args.label_path
    if not args.sampling_rate == None:
        if args.sampling_rate < 0:
            raise Exception("Sampling rate must be a positive integer!")
        SAMPLING = args.sampling_rate
    if not args.confidence == None:
        if args.confidence < 0 or args.confidence > 1:
            raise Exception("Confidence level must be between 0 and 1!")
        CONFIDENCE = args.confidence


def load_model(model_path):
    print('Loading model...', end='')
    start_time = time.time()

    # Load saved model and build the detection function
    model = tf.saved_model.load(model_path)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    return model



def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]
    
    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                                    output_dict['detection_masks'], output_dict['detection_boxes'],
                                    image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
    return output_dict



def run_inference(cap, model, category_index):
    # Makes a recording
    # Open output video file
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 5, (RESOLUTION["width"], RESOLUTION["height"]))
    count = 0
    while cap.isOpened():
        ret, image_np_frame = cap.read()
        if ret:
            # Gets every n-th frame
            count += SAMPLING # Changes the sampling rate of the video
            cap.set(1, count)

            # Starts fps counter
            start_time = time.time()

            # Actual detection.
            output_dict = run_inference_for_single_image(model, image_np_frame)

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np_frame,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=CONFIDENCE)


            # Ends timer for the length of image processing
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Counts the detected cars in the frame
            num_of_cars = 0
            for i in output_dict["detection_scores"]:
                if i > CONFIDENCE:
                    num_of_cars += 1
                else:
                    break

            # Prints elapsed time and number of cars in frame
            print('Iteration took {} seconds. Number of vehicles in frame: {}'.format('%.3f'%(elapsed_time), num_of_cars))
        
            # Displays the FPS and frame time on the video
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            fontScale              = 1
            fontColor              = (255,255,255)
            lineType               = 2            
            cv2.putText(image_np_frame, 'FPS: ' + str('%.3f'%(1 / elapsed_time)), (10,1030), font, fontScale, fontColor, lineType)
            cv2.putText(image_np_frame, "Frame time: " + '%.3f'%(elapsed_time) + "s", (10,1060), font, fontScale, fontColor, lineType)
            

            # Shows the frame
            cv2.imshow('object_detection', cv2.resize(image_np_frame, (RESOLUTION["width"], RESOLUTION["height"])))
            
            # Write the marked frame to the video
            out.write(image_np_frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                out.release()
                cap.release()
                cv2.destroyAllWindows()
                break
        else:
            cap.release()
            break

def main():
    var_init()

    # Enable GPU dynamic memory allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Load the model and labels
    detection_model = load_model(PATH_TO_MODEL_DIR)
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    # Run program
    run_inference(cap, detection_model, category_index) # capture, model, map with classes and ids

if __name__ == '__main__':
    main()