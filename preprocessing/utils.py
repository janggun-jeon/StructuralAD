import os
from typing import Union, Tuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import torch
import numpy as np

def normalizing_img(original_image, img_h, img_w):
        original_image = tf.cast(original_image, dtype=tf.float32)
        original_image = original_image/255.0

        resized_image = tf.image.resize(original_image, [img_h, img_w])
        resized_image = tf.expand_dims(resized_image, axis=0)

        return resized_image, original_image
    
def read_image(path_or_img: Union[str, torch.Tensor, np.ndarray], img_h: int = 200, img_w:int = 300) -> Tuple[tf.Tensor, tf.Tensor]:
    '''
    Decode image from path and resize it to img_h x img_w.
    Returns the resized image and the original image.
    arg:
        path:str, path to image
        img_h:int, height of image
        img_w:int, width of image
    return:
        resized_image:tf.Tensor, image tensor
        original_image:tf.Tensor, original image tensor

    '''
    # assert isinstance(path_or_img, str) , 'path must be a string'
    # assert os.path.exists(path_or_img) , 'Image path must be a valid path'
    assert isinstance(img_h, int) , 'img_h must be an integer'
    assert isinstance(img_w, int) , 'img_w must be an integer'

    if isinstance(path_or_img, str):
        image_raw = tf.io.read_file(path_or_img)
        original_image = tf.io.decode_image(image_raw, channels=3)

        return normalizing_img(original_image, img_h, img_w)
    if isinstance(path_or_img, torch.Tensor):
        original_image = tf.convert_to_tensor(path_or_img.detach().cpu().numpy())
        original_image = tf.transpose(original_image, [1, 2, 0])
        original_image = tf.cond(tf.equal(tf.rank(original_image), 2), lambda:tf.image.grayscale_to_rgb(original_image[..., tf.newaxis]), lambda:original_image)
        
        return normalizing_img(original_image, img_h, img_w)
    else:
        original_image = tf.convert_to_tensor(path_or_img)
        # original_image = tf.transpose(original_image, [1, 2, 0])
        original_image = tf.cond(tf.equal(tf.rank(original_image), 2), lambda:tf.image.grayscale_to_rgb(original_image[..., tf.newaxis]), lambda:original_image)
        
        return normalizing_img(original_image, img_h, img_w)

def post_enhance_iteration(original_image:tf.Tensor, alpha_maps:tf.Tensor, iteration:int = 6)-> tf.Tensor:
    '''
    Enhance the original image by iteratively applying predicted alpha maps.
    arg:
        original_image:tf.Tensor, original image tensor
        alpha_maps:tf.Tensor, alpha maps tensor
        iteration:int, number of iterations
    return:
        enhanced_image:tf.Tensor, enhanced image tensor
    '''
    assert isinstance(original_image, tf.Tensor) , 'original_image must be a tensor'
    assert isinstance(alpha_maps, tf.Tensor) , 'alpha_maps must be a tensor'
    assert iteration < 10, 'iteration must be between 1 and 10'

    if iteration == 0:
        iteration = 1
    # Check if image and alpha map has batch dimension
    if original_image.shape.rank == 4:
        original_image = tf.squeeze(original_image, axis=0)
    if alpha_maps.shape.rank == 4:
        alpha_maps = tf.squeeze(alpha_maps, axis=0)

    # get original image height and width
    h, w, _ = original_image.shape

    # Resize alpha maps to original image size
    a_maps = tf.image.resize(alpha_maps, [h,w], method=tf.image.ResizeMethod.BICUBIC)
    # a_maps = (a_maps-1)/2
    for _ in range(iteration):
        original_image = original_image + (a_maps)*(tf.square(original_image) - original_image)
    
    ehnanced_original_image = tf.cast(original_image*255, dtype=tf.uint8)
    ehnanced_original_image = tf.clip_by_value(ehnanced_original_image, 0, 255)

    return ehnanced_original_image