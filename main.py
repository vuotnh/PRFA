from PRFA_attack import *
import tensorflow as tf
from data_loader import *
import torch
from criterion import *

if __name__ == '__main__':
    """
            img_metas (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            The "img_meta" item is always populated.  The contents of the "img_meta"
            dictionary depends on "meta_keys". By default this includes:

                - "img_shape": shape of the image input to the network as a tuple \
                    (h, w, c).  Note that images may be zero padded on the \
                    bottom/right if the batch tensor is larger than this shape.

                - "scale_factor": a float indicating the preprocessing scale

                - "flip": a boolean indicating if image flip transform was used

                - "filename": path to the image file

                - "ori_shape": original shape of the image as a tuple (h, w, c)

                - "pad_shape": image shape after padding

                - "img_norm_cfg": a dict of normalization information:

                    - mean - per channel mean subtraction
                    - std - per channel std divisor
                    - to_rgb - bool indicating if bgr was converted to rgb
            Default: ``('filename', 'ori_filename', 'ori_shape', 'img_shape',
            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
            'img_norm_cfg')``
            """
    yolov5 = tf.keras.models.load_model('./yolov5s_saved_model')

    image_dataset = train_loader('./data', 8)
    image = next(iter(image_dataset))[1]
    data = {'img_metas': {'img_shape': (640, 640, 3), 'scale_factor': 0.1, 'flip': True, 'filename': 'test.png',
                          'ori_shape': (640, 640, 3), 'pad_shape': (640, 640, 3), 'img_norm_cfg': None
                          }, 'img': [image.reshape(1, 3, 640, 640)]}
    input_size = 640
    mse = torch.nn.MSELoss()
    attacker = IoUSSAttack(max_loss_queries=1000, epsilon=0.01,
                           p='inf', p_init=4, lb=0., ub=1., name='IoUsquare',
                           attack_model=yolov5, attack_mode=True, targeted=False,
                           ori_img=image, model_name='YOLO', zeta=0.5, lambda1=0.5, patch_attack='square',
                           keypoints_models=None, loss='iou_loss')
    attacker.run(data=data, loss_fct=mse, early_stop_crit_fct=early_stop_crit_fct_with_iou)
