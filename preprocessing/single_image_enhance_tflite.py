import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.utils import save_img

from preprocessing.utils import read_image, post_enhance_iteration

def tflite_run_inference(tflite_path:str, img_info:str, iteration:int = 6):
    '''
    Run inference on a single resized image.
    args:
        tflite_path: path to tflite model
        img_info: path to image file
        iteration: number of Post Ehnancing iterations
    return: None
    '''

    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)
    
    # # Get image name from path
    # image_name = (img_info.split('/')[-1]).split('.')[0]
    
    # # Get model name from model path
    # model_name = (tflite_path.split('/')[-1]).split('.')[0]
    
    
    with tf.device('/GPU:0'):
        # Load model
        interpreter = tf.lite.Interpreter(model_path=tflite_path, num_threads=4)
        interpreter.allocate_tensors()
        
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Read image
        img_h = int(input_details[0]['shape'][1])
        img_w = int(input_details[0]['shape'][2])
        resize_image, original_image = read_image(img_info, img_h=img_h, img_w=img_w)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], resize_image.numpy())
        interpreter.invoke()
        a_maps = tf.cast(interpreter.get_tensor(output_details[1]['index']), tf.float32)
        enhanced_img = tf.cast(interpreter.get_tensor(output_details[0]['index']), tf.float32)
        
        return post_enhance_iteration(original_image, a_maps, iteration) # enhanced_original_image


def zeroDCE(img_info, iteration=6):
    return tflite_run_inference('preprocessing/TFLITE_models/zero_dce_lite_160x160_iter8_30.tflite', img_info, iteration=6)