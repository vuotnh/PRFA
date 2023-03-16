import tensorflow as tf
from data_loader import *
import numpy as np
import matplotlib.pyplot as plt


def label_smooth(scores, labels, n_cls):
    result = np.ones([len(labels), n_cls])
    result = ((1 - scores) / (n_cls - 1)).numpy().reshape(len(labels), 1) * result
    result[np.arange(len(labels)), labels] = scores
    return result


def get_predict_bbox_single_image(model, input_size, image_tensor, num_classes, to_Tensor=False):
    # image = image_tensor.permute(1, 2, 0).numpy()
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.reshape(1, image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2])
    output = model(image_tensor.reshape(1, input_size, input_size, 3))[0][0]

    pred_bbox_scores = output[:, 4]
    pred_xywh = output[:, 0:4]
    pred_bbox_prob = output[:, 5:]
    labels = np.argmax(pred_bbox_prob, axis=-1)
    pred_coors = tf.concat([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                           pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=1)

    if not to_Tensor:
        scores_clean = pred_bbox_scores
        labels_clean = labels
        bbox_coors = pred_coors
        scores_smooth_clean = label_smooth(scores_clean, labels_clean, num_classes)
    else:
        scores_clean = pred_bbox_scores
        labels_clean = labels.long()
        bbox_coors = pred_coors
        return bbox_coors, scores_clean, labels_clean
    return bbox_coors, scores_smooth_clean, labels_clean


if __name__ == '__main__':
    yolov5 = tf.keras.models.load_model('./yolov5s_saved_model')

    image_dataset = train_loader('./data', 8)
    image = next(iter(image_dataset))[1]
    input_size = 640
    pred_bbox_conf, pred_xywh, pred_coor = get_predict_bbox_single_image(yolov5, 640, image)
