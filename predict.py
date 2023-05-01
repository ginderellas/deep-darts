import argparse
from yacs.config import CfgNode as CN
import os.path as osp
import os
from dataloader import get_splits
import cv2
import numpy as np
from time import time
from dataset.annotate import draw, get_dart_scores
import pickle


def bboxes_to_xy(bboxes, max_darts=3):
    """
    This function converts bounding boxes to a set of (x, y) coordinates for darts and calibration points.

    Args:
        bboxes (numpy array): A numpy array of shape (n, 5), where n is the number of bounding boxes.
            Each bounding box is represented by a row in the array, where the first four columns
            represent the x and y coordinates of the top-left and bottom-right corners of the bounding box
            and the last column represents the class of the bounding box (0 for darts and 1-4 for calibration points).
        max_darts (int): The maximum number of darts to include in the output. Defaults to 3.

    Returns:
        xy (numpy array): A numpy array of shape (4 + max_darts, 3) representing the (x, y) coordinates of the
            darts and calibration points. The first four rows represent the calibration points, and the remaining
            rows represent the darts. The first two columns represent the (x, y) coordinates, and the last column
            is a binary value indicating whether the point has been estimated (1) or not (0).
    """
    xy = np.zeros((4 + max_darts, 3), dtype=np.float32)
    for cls in range(5):
        if cls == 0:
            dart_xys = bboxes[bboxes[:, 4] == 0, :2][:max_darts]
            xy[4:4 + len(dart_xys), :2] = dart_xys
        else:
            cal = bboxes[bboxes[:, 4] == cls, :2]
            if len(cal):
                xy[cls - 1, :2] = cal[0]
    xy[(xy[:, 0] > 0) & (xy[:, 1] > 0), -1] = 1
    if np.sum(xy[:4, -1]) == 4:
        return xy
    else:
        xy = est_cal_pts(xy)
    return xy


def est_cal_pts(xy):
    """
    This function estimates the missing calibration points in a 3x4 numpy array xy, 
    which represents a set of four 3D points in homogeneous coordinates.
    
    Args:
        xy: a 3x4 numpy array of 3D points in homogeneous coordinates, 
            where each row represents x, y, and z coordinates, respectively. 
            The last column indicates whether a point is missing (0) or not (1).
          
    Returns:
        xy: a 3x4 numpy array of 3D points in homogeneous coordinates, 
            where missing calibration points are estimated based on the other three points.
            If there are more than one missing points, the function prints an error message.
    """
    missing_idx = np.where(xy[:4, -1] == 0)[0]
    if len(missing_idx) == 1:
        if missing_idx[0] <= 1:
            center = np.mean(xy[2:4, :2], axis=0)
            xy[:, :2] -= center
            if missing_idx[0] == 0:
                xy[0, 0] = -xy[1, 0]
                xy[0, 1] = -xy[1, 1]
                xy[0, 2] = 1
            else:
                xy[1, 0] = -xy[0, 0]
                xy[1, 1] = -xy[0, 1]
                xy[1, 2] = 1
            xy[:, :2] += center
        else:
            center = np.mean(xy[:2, :2], axis=0)
            xy[:, :2] -= center
            if missing_idx[0] == 2:
                xy[2, 0] = -xy[3, 0]
                xy[2, 1] = -xy[3, 1]
                xy[2, 2] = 1
            else:
                xy[3, 0] = -xy[2, 0]
                xy[3, 1] = -xy[2, 1]
                xy[3, 2] = 1
            xy[:, :2] += center
    else:
        # TODO: if len(missing_idx) > 1
        print('Missed more than 1 calibration point')
    return xy


def predict(
        yolo,
        cfg,
        labels_path='./dataset/labels.pkl',
        dataset='d1',
        split='val',
        max_darts=3,
        write=False):
    """
    Perform predictions on a dataset of images using a YOLO object detection model.

    Args:
        yolo (YoloV3): A YOLO object detection model.
        cfg (Config): A configuration object.
        labels_path (str): Path to the labels file. Defaults to './dataset/labels.pkl'.
        dataset (str): The name of the dataset to use. Defaults to 'd1'.
        split (str): The name of the split to use. Defaults to 'val'.
        max_darts (int): The maximum number of darts to predict. Defaults to 3.
        write (bool): Whether to write images with predictions overlaid. Defaults to False.

    Returns:
        results (dict): A dictionary containing the results of the predictions.
    """
    np.random.seed(0)

    write_dir = osp.join('./models', cfg.model.name, 'preds', split)
    if write:
        os.makedirs(write_dir, exist_ok=True)

    data = get_splits(labels_path, dataset, split)
    img_prefix = osp.join(cfg.data.path, 'cropped_images', str(cfg.model.input_size))
    img_paths = [osp.join(img_prefix, folder, name) for (folder, name) in zip(data.img_folder, data.img_name)]

    xys = np.zeros((len(data), 7, 3))  # third column for visibility
    data.xy = data.xy.apply(np.array)
    for i, _xy in enumerate(data.xy):
        xys[i, :_xy.shape[0], :2] = _xy
        xys[i, :_xy.shape[0], 2] = 1
    xys = xys.astype(np.float32)

    preds = np.zeros((len(img_paths), 4 + max_darts, 3))
    print('Making predictions with {}...'.format(cfg.model.name))

    for i, p in enumerate(img_paths):
        if i == 1:
            ti = time()
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxes = yolo.predict(img)
        preds[i] = bboxes_to_xy(bboxes, max_darts)

        if write:
            write_dir = osp.join('./models', cfg.model.name, 'preds', split, p.split('/')[-2])
            os.makedirs(write_dir, exist_ok=True)
            xy = preds[i]
            xy = xy[xy[:, -1] == 1]
            error = sum(get_dart_scores(preds[i, :, :2], cfg, numeric=True)) - sum(get_dart_scores(xys[i, :, :2], cfg, numeric=True))
            if not args.fail_cases or (args.fail_cases and error != 0):
                img = draw(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), xy[:, :2], cfg, circles=False, score=True)
                cv2.imwrite(osp.join(write_dir, p.split('/')[-1]), img)

    fps = (len(img_paths) - 1) / (time() - ti)
    print('FPS: {:.2f}'.format(fps))

    ASE = []  # absolute score error
    for pred, gt in zip(preds, xys):
        ASE.append(abs(
            sum(get_dart_scores(pred[:, :2], cfg, numeric=True)) -
            sum(get_dart_scores(gt[:, :2], cfg, numeric=True))))

    ASE = np.array(ASE)
    PCS = len(ASE[ASE == 0]) / len(ASE) * 100
    MASE = np.mean(ASE)

    print('Percent Correct Score (PCS): {:.1f}%'.format(PCS))
    print('Mean Absolute Score Error (MASE): {:.2f}'.format(MASE))

    results = {
        'img_paths': img_paths,
        'preds': preds,
        'gt': xys,
        'fps': fps,
        'ASE': ASE,
        'PCS': PCS,
        'MASE': MASE
    }

    pickle.dump(results, open(osp.join('./models', cfg.model.name, 'results.pkl'), 'wb'))
    print('Saved results.')


if __name__ == '__main__':
    from train import build_model
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='deepdarts_d1')
    parser.add_argument('-s', '--split', default='val')
    parser.add_argument('-w', '--write', action='store_true')
    parser.add_argument('-f', '--fail-cases', action='store_true')
    args = parser.parse_args()

    cfg = CN(new_allowed=True)
    cfg.merge_from_file(osp.join('configs', args.cfg + '.yaml'))
    cfg.model.name = args.cfg

    yolo = build_model(cfg)
    yolo.load_weights(osp.join('models', args.cfg, 'weights'), cfg.model.weights_type)

    predict(yolo, cfg,
            dataset=cfg.data.dataset,
            split=args.split,
            write=args.write)