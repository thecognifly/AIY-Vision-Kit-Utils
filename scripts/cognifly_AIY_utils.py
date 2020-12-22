import sys

from PIL import ImageDraw
import numpy as np

def area(box_encodings):
  widths = box_encodings[:,2]
  heights = box_encodings[:,3]

  return widths*heights


def intersection(box_encodings1, box_encodings2):
    x1 = box_encodings1[:,0]
    y1 = box_encodings1[:,1]
    width1 = box_encodings1[:,2]
    height1 = box_encodings1[:,3]

    x2 = box_encodings2[:,0]
    y2 = box_encodings2[:,1]
    width2 = box_encodings2[:,2]
    height2 = box_encodings2[:,3]

    x = np.maximum(x1,x2)
    y = np.maximum(y1,y2)
    
    width = np.maximum(np.minimum(x1 + width1, x2 + width2) - x, 0)
    height = np.maximum(np.minimum(y1 + height1, y2 + height2) - y, 0)
    
    return width * height


def IoU(box_encodings1, box_encodings2):
    intersection_area = intersection(box_encodings1, box_encodings2)
    union_area = area(box_encodings1) + area(box_encodings2) - intersection_area
    result = np.zeros(len(intersection_area))
    result[union_area>0] = intersection_area[union_area>0] / union_area[union_area>0]

    return result

def draw_boxes(img, output):
    draw = ImageDraw.Draw(img)
    w,h = img.size
    highest_score = output[1][0].argsort()[-1]
    highest_score_val = output[1][0][highest_score]
    best_box = output[0][0][highest_score]
    best_class_id = output[2][0][highest_score]

    for box,score,class_id in zip(output[0][0], output[1][0],output[2][0]):
        draw.rectangle([box[1]*w, box[0]*h, box[3]*w, box[2]*h], fill=None, outline=None)
        draw.text((box[1]*w+2, box[0]*h+2), str(int(class_id))+": "+str(score))

    draw.rectangle([best_box[1]*w, best_box[0]*h, best_box[3]*w, best_box[2]*h], fill=None, outline="red", width=2)
    draw.text((best_box[1]*w+2, best_box[0]*h+2), str(int(best_class_id))+": "+str(highest_score_val), fill="red")

    return img

# based on https://github.com/google/aiyprojects-raspbian/blob/aiyprojects/src/examples/vision/object_detection.py
def decode_box_encoding(box_encodings, anchors):
    """Decodes bounding box encoding.

    Args:
      box_encodings: numpy array with shape (number of boxes, 4)
      anchors: numpy array with shape (number of boxes, 4)
    Returns:
      (xmin, ymin, xmax, ymax), each has range [0.0, 1.0].
    """
    y_scale = 10.0
    x_scale = 10.0
    height_scale = 5.0
    width_scale = 5.0

    rows, columns = anchors.shape

    # rel_y_translation = box_encoding[0] / y_scale
    # rel_x_translation = box_encoding[1] / x_scale
    # rel_height_dilation = box_encoding[2] / height_scale
    # rel_width_dilation = box_encoding[3] / width_scale
    rel_yxhw = box_encodings.reshape((rows, columns))/np.array([y_scale,x_scale,height_scale,width_scale])
    
    # anchor_ymin, anchor_xmin, anchor_ymax, anchor_xmax = anchor
    # anchor_ycenter = (anchor_ymax + anchor_ymin) / 2
    # anchor_xcenter = (anchor_xmax + anchor_xmin) / 2
    # anchor_height = anchor_ymax - anchor_ymin
    # anchor_width = anchor_xmax - anchor_xmin
    anchor_ycenter = (anchors[:,2] + anchors[:,0]) / 2
    anchor_xcenter = (anchors[:,3] + anchors[:,1]) / 2
    anchor_height = anchors[:,2] - anchors[:,0]
    anchor_width = anchors[:,3] - anchors[:,1]

    # ycenter = anchor_ycenter + anchor_height * rel_y_translation
    # xcenter = anchor_xcenter + anchor_width * rel_x_translation
    # height = math.exp(rel_height_dilation) * anchor_height
    # width = math.exp(rel_width_dilation) * anchor_width
    ycenter = anchor_ycenter + anchor_height * rel_yxhw[:,0]
    xcenter = anchor_xcenter + anchor_width * rel_yxhw[:,1]
    height = np.exp(rel_yxhw[:,2]) * anchor_height
    width = np.exp(rel_yxhw[:,3]) * anchor_width

    # Finally it will clamp values to [0.0, 1.0] range, 
    # (the bounding boxes are relative to the img size)
    # so they will not fall outside of the image.
    xmin = np.clip(xcenter - width / 2, 0, 1)
    ymin = np.clip(ycenter - height / 2, 0, 1)
    xmax = np.clip(xcenter + width / 2, 0, 1)
    ymax = np.clip(ycenter + height / 2, 0, 1)


    return np.asarray([xmin, ymin, xmax, ymax]).T


def logistic(scores):
    return 1.0 / (1.0 + np.exp(-scores))


def draw_boxes_raw(img, boxes, scores, class_id=1, threshold=1.0, color = "green"):
    draw = ImageDraw.Draw(img)
    w,h = img.size

    for i,(box, score) in enumerate(zip(boxes, scores)):
        if score >= threshold:
            draw.rectangle([box[0]*w, box[1]*h, box[2]*w, box[3]*h], fill=None, outline=color)
            draw.text((box[0]*w+2, box[1]*h+2), f"{int(class_id)} : {score:0.3f} => {i}", fill=color)

    return img


def process_output_tensor(concat, concat_1, aiy_anchors, classes=[1], IoU_thres=0.5, raw_boxes=False, score_threshold=0):

  score_threshold = max(score_threshold, sys.float_info.epsilon)
  logit_score_threshold = np.log(score_threshold / (1 - score_threshold))

  #
  # I haven't test this when there're more than just background and one class...
  box_encodings = decode_box_encoding(concat, aiy_anchors)

  # Reshaping just to make like easier...
  logit_scores = concat_1.reshape(concat_1.shape[1],concat_1.shape[2]) 
  # logit_scores values are the sum of all outputs so they are not limited 0 to 1.

  detection_boxes, detection_scores, detection_classes = {}, {}, []

  for selected_class in classes:
    #
    # Everything that is ABOVE logit_score_threshold will be used
    #
    selected_indices = np.arange(len(logit_scores))[(logit_scores>logit_score_threshold)[:,selected_class]]

    if len(selected_indices)==0:
      break

    selected_val = logit_scores[selected_indices,selected_class]

    if raw_boxes:
      detection_boxes[selected_class] = box_encodings[selected_indices]
      detection_scores[selected_class] = selected_val
      detection_classes.append(selected_class)
      break

    #
    # Sort original indices according to the output values
    #
    selected_class_indices_argsorted = selected_indices[logit_scores[selected_indices][:,selected_class].argsort()[::-1]]

    bj = selected_class_indices_argsorted
    IoU_winners = []
    while True:
      bi = bj[0] # the index for the highest value
      IoU_res = IoU(box_encodings[bi].reshape((1,4)),box_encodings[bj[1:]])
      test = IoU_res>IoU_thres
      swallowed = bj[1:][test==True]
      IoU_winners.append((bi,
                          logit_scores[bi, selected_class],
                          len(swallowed)))
      if (test).all(): # there's nothing to test
        break
      else:
        bj = bj[1:][test==False]

    boxes = [i[0] for i in IoU_winners]
    scores = np.asarray([logistic(i[1]) for i in IoU_winners])    

    detection_boxes[selected_class] = box_encodings[boxes]
    detection_scores[selected_class] = scores
    detection_classes.append(selected_class)

  return detection_boxes, detection_scores, detection_classes

