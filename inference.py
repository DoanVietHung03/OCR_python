# inference.py
import cv2
import numpy as np
from config import Detection, CHARSET
from utils import letterbox_yolo


def infer_yolo(
    session, input_names, output_names, img, conf_thresh=0.5, allowed_classes=None
):
    if allowed_classes is None:
        allowed_classes = []

    input_shape = session.get_inputs()[0].shape
    input_h = (
        input_shape[2]
        if len(input_shape) >= 4 and isinstance(input_shape[2], int)
        else 640
    )
    input_w = (
        input_shape[3]
        if len(input_shape) >= 4 and isinstance(input_shape[3], int)
        else 640
    )

    padded_img, ratio, pad_w, pad_h = letterbox_yolo(img, input_w, input_h)
    blob = cv2.dnn.blobFromImage(
        padded_img, 1.0 / 255.0, (input_w, input_h), (0, 0, 0), swapRB=True, crop=False
    )

    outputs = session.run(output_names, {input_names[0]: blob})
    out_data = outputs[0]
    shape = out_data.shape

    boxes_list = []
    scores_list = []
    class_ids_list = []

    is_end2end_nms = (len(shape) == 3 and shape[2] in [6, 7] and shape[1] < 2000) or (
        len(shape) == 2 and shape[1] in [6, 7]
    )

    if is_end2end_nms:
        data_matrix = out_data[0] if len(shape) == 3 else out_data
        offset = 1 if shape[-1] == 7 else 0

        if len(data_matrix) > 0:
            val1 = data_matrix[:, offset + 4]
            val2 = data_matrix[:, offset + 5]

            mask_val1_int = np.abs(val1 - np.round(val1)) < 1e-5
            classes = np.where(mask_val1_int, val1, val2).astype(int)
            scores = np.where(mask_val1_int, val2, val1)

            conf_mask = scores > conf_thresh
            if allowed_classes:
                class_mask = np.isin(classes, allowed_classes)
                valid_mask = conf_mask & class_mask
            else:
                valid_mask = conf_mask

            valid_matrix = data_matrix[valid_mask]

            if len(valid_matrix) > 0:
                x1_raw = valid_matrix[:, offset + 0]
                y1_raw = valid_matrix[:, offset + 1]
                x2_raw = valid_matrix[:, offset + 2]
                y2_raw = valid_matrix[:, offset + 3]

                x1 = (x1_raw - pad_w) / ratio
                y1 = (y1_raw - pad_h) / ratio
                w = (x2_raw - x1_raw) / ratio
                h = (y2_raw - y1_raw) / ratio

                boxes_list = np.column_stack((x1, y1, w, h)).astype(int).tolist()
                scores_list = scores[valid_mask].tolist()
                class_ids_list = classes[valid_mask].tolist()

    else:
        data_matrix = out_data[0]
        if data_matrix.shape[0] < data_matrix.shape[1]:
            data_matrix = data_matrix.T

        boxes_data = data_matrix[:, 0:4]
        scores_data = data_matrix[:, 4:]

        max_scores = np.max(scores_data, axis=1)
        max_score_ids = np.argmax(scores_data, axis=1)

        conf_mask = max_scores > conf_thresh

        if allowed_classes:
            class_mask = np.isin(max_score_ids, allowed_classes)
            valid_mask = conf_mask & class_mask
        else:
            valid_mask = conf_mask

        valid_boxes = boxes_data[valid_mask]
        valid_scores = max_scores[valid_mask]
        valid_class_ids = max_score_ids[valid_mask]

        if len(valid_boxes) > 0:
            cx = valid_boxes[:, 0]
            cy = valid_boxes[:, 1]
            bw = valid_boxes[:, 2]
            bh = valid_boxes[:, 3]

            x1 = ((cx - 0.5 * bw) - pad_w) / ratio
            y1 = ((cy - 0.5 * bh) - pad_h) / ratio
            width = bw / ratio
            height = bh / ratio

            boxes_list = np.column_stack((x1, y1, width, height)).astype(int).tolist()
            scores_list = valid_scores.tolist()
            class_ids_list = valid_class_ids.tolist()

    indices = cv2.dnn.NMSBoxes(boxes_list, scores_list, conf_thresh, 0.5)
    final_dets = []

    if len(indices) > 0:
        for idx in indices.flatten():
            x, y, w, h = boxes_list[idx]
            safe_x = max(0, x)
            safe_y = max(0, y)
            safe_w = min(img.shape[1] - safe_x, w)
            safe_h = min(img.shape[0] - safe_y, h)
            if safe_w > 0 and safe_h > 0:
                final_dets.append(
                    Detection(
                        [safe_x, safe_y, safe_w, safe_h],
                        scores_list[idx],
                        class_ids_list[idx],
                    )
                )

    return final_dets


def decode_parseq(logits_data, seq_len, num_classes):
    result = ""
    confidences = []

    for i in range(seq_len):
        logits_step = logits_data[i]
        max_idx = np.argmax(logits_step)
        max_val = logits_step[max_idx]

        if max_idx == 0:
            break

        sum_exp = np.sum(np.exp(logits_step - max_val))
        char_confidence = 1.0 / sum_exp

        if 0 < max_idx <= len(CHARSET):
            result += CHARSET[max_idx - 1]
            confidences.append(char_confidence)

    if not confidences:
        return "", 0.0

    avg_conf = sum(confidences) / len(confidences)
    min_conf = min(confidences)

    out_confidence = (avg_conf * 0.4) + (min_conf * 0.6)

    return result, out_confidence