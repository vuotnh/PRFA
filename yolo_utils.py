import tensorflow as tf
from data_loader import *
import matplotlib.pyplot as plt


def get_predict_bbox_single_image(model, input_size, image_tensor):
    # image = image_tensor.permute(1, 2, 0).numpy()
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.reshape(1, image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2])
    output = model(image_tensor.reshape(1, input_size, input_size, 3))[0]
    pred_bbox_conf = output[:, 4]
    pred_xywh = output[:, 0:4]
    pred_coor = tf.concat([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                           pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=1)

    # return pred_bbox_conf, pred_xywh, pred_coor
    return output


if __name__ == '__main__':
    yolov5 = tf.keras.models.load_model('./yolov5s_saved_model')

    image_dataset = train_loader('./data', 8)
    image = next(iter(image_dataset))[1]
    input_size = 640
    pred_bbox_conf, pred_xywh, pred_coor = get_predict_bbox_single_image(yolov5, 640, image)
