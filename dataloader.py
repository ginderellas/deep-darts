import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os.path as osp
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from dataset.annotate import draw, transform
from yacs.config import CfgNode as CN
from yolov4.tf.dataset import cut_out


d1_val = ['d1_02_06_2020', 'd1_02_16_2020', 'd1_02_22_2020']
d1_test = ['d1_03_03_2020', 'd1_03_19_2020', 'd1_03_23_2020', 'd1_03_27_2020', 'd1_03_28_2020', 'd1_03_30_2020', 'd1_03_31_2020']

d2_val = ['d2_02_03_2021', 'd2_02_05_2021']
d2_test = ['d2_03_03_2020', 'd2_02_10_2021', 'd2_02_03_2021_2']


def get_splits(path='./dataset/labels.pkl', dataset='d1', split='train'):
    """Splits a dataset into training, validation, and testing sets.

    Args:
        path (str): Path to the dataset labels file in pickle format.
            Defaults to './dataset/labels.pkl'.
        dataset (str): Name of the dataset to split. Must be either 'd1' or 'd2'.
            Defaults to 'd1'.
        split (str): Type of split to return. Must be either None, 'train', 'val', or 'test'.
            If None, returns a dictionary containing all splits.
            Defaults to 'train'.

    Returns:
        DataFrame or Dict: If split is None, returns a dictionary containing all splits,
            where the keys are 'train', 'val', and 'test' and the values are DataFrames.
            Otherwise, returns the split specified by the split argument as a DataFrame.
    """
    assert dataset in ['d1', 'd2'], "dataset must be either 'd1' or 'd2'"
    assert split in [None, 'train', 'val', 'test'], "split must be in [None, 'train', 'val', 'test']"
    if dataset == 'd1':
        val_folders, test_folders = d1_val, d1_test
    else:
        val_folders, test_folders = d2_val, d2_test
    df = pd.read_pickle(path)
    df = df[df.img_folder.str.contains(dataset)]
    splits = {}
    splits['val'] = df[np.isin(df.img_folder, val_folders)]
    splits['test'] = df[np.isin(df.img_folder, test_folders)]
    splits['train'] = df[np.logical_not(np.isin(df.img_folder, val_folders + test_folders))]
    if split is None:
        return splits
    else:
        return splits[split]


def preprocess(path, xy, cfg, bbox_to_gt_func, split='train', return_xy=False):
    """Preprocesses an image and its associated bounding boxes for training or evaluation.

    Args:
        path (str): The path to the image file.
        xy (numpy.ndarray): The bounding box coordinates for the image, as a numpy array with shape (N, 4).
        cfg (Config): The configuration object for the training or evaluation run.
        bbox_to_gt_func (function): A function that converts bounding boxes to ground truth targets.
        split (str, optional): The dataset split being processed. Must be one of 'train', 'val', or 'test'. Defaults to 'train'.
        return_xy (bool, optional): Whether to return the bounding box coordinates as well as the image. Defaults to False.

    Returns:
        If `return_xy` is False, a tuple containing the preprocessed image and its ground truth targets.
        If `return_xy` is True, a tuple containing the preprocessed image and its bounding box coordinates.
    """
    path = path.numpy().decode('utf-8')
    xy = xy.numpy()

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # yolov4 tf convention
    img = img / 255.  # yolov4 tf convention

    if split == 'train' and np.random.uniform() < cfg.aug.overall_prob:
        transformed = False

        if cfg.aug.flip_lr_prob and np.random.uniform() < cfg.aug.flip_lr_prob:
            if not transformed:
                xy, img, M = transform(xy, img)
                transformed = True
            img, xy = flip(img, xy, direction='lr')

        if cfg.aug.flip_ud_prob and np.random.uniform() < cfg.aug.flip_ud_prob:
            if not transformed:
                xy, img, M = transform(xy, img)
                transformed = True
            img, xy = flip(img, xy, direction='ud')

        if cfg.aug.rot_prob and np.random.uniform() < cfg.aug.rot_prob:
            if not transformed:
                xy, img, M = transform(xy, img)
                transformed = True
            angles = np.arange(-180, 180, step=cfg.aug.rot_step)
            angle = angles[np.random.randint(len(angles))]
            img, xy = rotate(img, xy, angle, darts_only=True)

        if cfg.aug.rot_small_prob and np.random.uniform() < cfg.aug.rot_small_prob:
            angle = np.random.uniform(-cfg.aug.rot_small_max, cfg.aug.rot_small_max)
            img, xy = rotate(img, xy, angle, darts_only=False)  # rotate cal points too

        if cfg.aug.jitter_prob and np.random.uniform() < cfg.aug.jitter_prob:
            h, w = img.shape[:2]
            jitter = cfg.aug.jitter_max * w
            tx = np.random.uniform(-1, 1) * jitter
            ty = np.random.uniform(-1, 1) * jitter
            img, xy = translate(img, xy, tx, ty)

        if cfg.aug.warp_prob and np.random.uniform() < cfg.aug.warp_prob:
            if not transformed:
                xy, img, M = transform(xy, img)
            M_inv = np.linalg.inv(M)
            M_inv[0, 1:3] *= np.random.uniform(0, cfg.aug.warp_rho, 2)
            M_inv[1, [0, 2]] *= np.random.uniform(0, cfg.aug.warp_rho, 2)
            M_inv[2, 0:2] *= np.random.uniform(0, cfg.aug.warp_rho, 2)
            xy, img, _ = transform(xy, img, M=M_inv)

        else:
            if transformed:
                M_inv = np.linalg.inv(M)
                xy, img, _ = transform(xy, img, M=M_inv)

    if return_xy:
        return img, xy

    bboxes = get_bounding_boxes(xy, cfg.train.bbox_size)

    if split == 'train':
        # cutout augmentation
        if cfg.aug.cutout_prob and np.random.uniform() < cfg.aug.cutout_prob:
            img, bboxes = cut_out([np.expand_dims(img, axis=0), bboxes])
            img = img[0]

    gt = bbox_to_gt_func(bboxes)
    gt = [item.squeeze() for item in gt]
    return (img, *gt)


def align_board(img, xy):
    """Rotate and align the dartboard image to make it horizontal.

    Args:
        img (numpy.ndarray): The input image of the dartboard.
        xy (numpy.ndarray): An array of shape (20, 2) representing the 20 calibration points of the dartboard.

    Returns:
        Tuple of:
            - img (numpy.ndarray): The rotated image of the dartboard.
            - xy (numpy.ndarray): The updated calibration points after the rotation.

    The function first calculates the center of the dartboard by taking the mean of the first four calibration points. Then it calculates the angle between the center point and the top calibration point of the dartboard, and rotates the image and calibration points by this angle to make the dartboard horizontal.

    Note that the function performs rotation on all the calibration points (darts_only=False) and returns the updated calibration points.
    """
    center = np.mean(xy[:4, :2], axis=0)
    angle = 9 - np.arctan((center[0] - xy[0, 0]) / (center[1] - xy[0, 1])) / np.pi * 180
    img, xy = rotate(img, xy, angle, darts_only=False)
    return img, xy


def rotate(img, xy, angle, darts_only=True):
    """Rotates the given image and corresponding bounding boxes around their center.

    Args:
    - img (numpy.ndarray): Image data to be rotated.
    - xy (numpy.ndarray): The bounding boxes, where each row is a coordinate of a point in the format x, y, visibility.
    - angle (float): The angle of rotation in degrees.
    - darts_only (bool): Whether to rotate only the darts in the image (xy[4:] points) or all the bounding boxes (default=True).
    
    Returns:
    - A tuple containing:
        - img (numpy.ndarray): The rotated image.
        - xy (numpy.ndarray): The rotated bounding boxes.
    """
    h, w = img.shape[:2]
    center = np.mean(xy[:4, :2], axis=0)
    M = cv2.getRotationMatrix2D((center[0]*w, center[1]*h), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))

    vis = xy[:, 2:]
    xy = xy[:, :2]
    if darts_only:
        if xy.shape[0] > 4:
            xy_darts = xy[4:]
            xy_darts -= center
            xy_darts = np.matmul(M[:, :2], xy_darts.T).T
            xy_darts += center
            xy[4:] = xy_darts
    else:
        xy -= center
        xy = np.matmul(M[:, :2], xy.T).T
        xy += center
    xy = np.concatenate([xy, vis], axis=-1)
    return img, xy


def flip(img, xy, direction, darts_only=True):
    """Flip the input image and coordinates either left-right or up-down.

    Args:
        img (np.array): The input image as an array of shape `(h, w, c)`.
        xy (np.array): The coordinates of the dartboard as an array of shape `(n, 4)`.
            The first 4 coordinates are the corners of the dartboard and the rest are
            the darts' locations. Each row has the format `(x, y, visibility, presence)`
            where `visibility` is a binary value indicating if the dart is visible or not
            and `presence` is a binary value indicating if the dart is present or not.
        direction (str): The direction to flip the image and coordinates. Either 'lr'
            to flip left-right or 'ud' to flip up-down.
        darts_only (bool): Whether to flip only the darts' coordinates or the whole board.

    Returns:
        Tuple[np.array, np.array]: A tuple containing the flipped image as an array of shape
            `(h, w, c)` and the flipped coordinates of the dartboard as an array of shape `(n, 4)`.

    """
    if direction == 'lr':
        img = img[:, ::-1, :]  # flip left-right
        axis = 0
    else:
        img = img[::-1, :, :]  # flip up-down
        axis = 1
    center = np.mean(xy[:4, :2], axis=0)
    vis = xy[:, 2:]
    xy = xy[:, :2]
    if darts_only:
        if xy.shape[0] > 4:
            xy_darts = xy[4:]
            xy_darts -= center
            xy_darts[:, axis] = -xy_darts[:, axis]
            xy_darts += center
            xy[4:] = xy_darts
    else:
        xy -= center
        xy[:, axis] = -xy[:, axis]
        xy += center
    xy = np.concatenate([xy, vis], axis=-1)
    return img, xy


def translate(img, xy, tx, ty):
    """Translate the image and the dartboard coordinates.

    Args:
        img: numpy array of shape (height, width, 3), representing an image.
        xy: numpy array of shape (N, 4), representing the coordinates of the dartboard
            and darts on the image. The first 4 coordinates correspond to the four corners
            of the dartboard, and the remaining coordinates correspond to the darts.
        tx: float, representing the distance to translate the image horizontally.
        ty: float, representing the distance to translate the image vertically.

    Returns:
        A tuple containing:
            img: numpy array of shape (height, width, 3), the translated image.
            xy: numpy array of shape (N, 4), the translated dartboard coordinates.
    """
    h, w = img.shape[:2]
    M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    img = cv2.warpAffine(img, M, (w, h))
    xy[:, 0] += tx/w
    xy[:, 1] += ty/h
    return img, xy


def warp_perspective(img, xy, rho):
    """Apply a random perspective transform to an image and its corresponding points.

    Args:
        img (numpy.ndarray): The image to be transformed.
        xy (numpy.ndarray): The points to be transformed.
        rho (float): The maximum amount of distortion allowed.

    Returns:
        numpy.ndarray: The transformed image.
        numpy.ndarray: The transformed points.
    """
    patch_size = 128
    top_point = (32,32)
    left_point = (patch_size+32, 32)
    bottom_point = (patch_size+32, patch_size+32)
    right_point = (32, patch_size+32)
    four_points = [top_point, left_point, bottom_point, right_point]
    h, w = img.shape[:2]

    perturbed_four_points = [
        (p[0] + np.random.uniform(-rho, rho), p[1] + np.random.uniform(-rho, rho))
        for p in four_points]

    M = cv2.getPerspectiveTransform(
        np.float32(four_points),
        np.float32(perturbed_four_points))

    warped_image = cv2.warpPerspective(img, M, (w, h))

    vis = xy[:, 2:]
    xy = xy[:, :2]
    xy *= [[w, h]]

    xyz = np.concatenate((xy, np.ones((xy.shape[0], 1))), axis=-1)
    xyz = np.matmul(M, xyz.T).T
    xy = xyz[:, :2] / xyz[:, 2:]

    xy /= [[w, h]]
    xy = np.concatenate([xy, vis], axis=-1)

    return warped_image, xy


def get_bounding_boxes(xy, size):
    """
    Returns the bounding boxes for a given array of 2D coordinates and a size.
    
    Args:
        xy:     A numpy array of shape (N, 3) where N is the number of 2D coordinates.
                The third column indicates the visibility of the point.
        size:   An integer indicating the size of the bounding box.
    
    Returns:
        xywhc:  A numpy array of shape (M, 5) where M is the number of visible points in the input array.
                The first four columns contain the bounding box coordinates (x, y, width, height).
                The last column contains the class label of the point (1 to 4 for the first four points, 0 for others).
    """
    xy[((xy[:, 0] - size / 2 <= 0) |
        (xy[:, 0] + size / 2 >= 1) |
        (xy[:, 1] - size / 2 <= 0) |
        (xy[:, 1] + size / 2 >= 1)), -1] = 0
    xywhc = []
    for i, _xy in enumerate(xy):
        if i < 4:
            cls = i + 1
        else:
            cls = 0
        if _xy[-1]:  # is visible
            xywhc.append([_xy[0], _xy[1], size, size, cls])
    xywhc = np.array(xywhc)
    return xywhc


def set_shapes(img, gt1, gt2, gt3, input_size):
    """
    Sets the shapes of the input image and ground truth tensors.

    Args:
        img (tf.Tensor): Input image tensor.
        gt1 (tf.Tensor): Ground truth tensor at scale 1/8.
        gt2 (tf.Tensor): Ground truth tensor at scale 1/16.
        gt3 (tf.Tensor): Ground truth tensor at scale 1/32.
        input_size (int): Desired size of the input image.

    Returns:
        img (tf.Tensor): Input image tensor with the desired shape.
        gt1 (tf.Tensor): Ground truth tensor at scale 1/8 with the desired shape.
        gt2 (tf.Tensor): Ground truth tensor at scale 1/16 with the desired shape.
        gt3 (tf.Tensor): Ground truth tensor at scale 1/32 with the desired shape.
    """
    img.set_shape([input_size, input_size, 3])
    gt1.set_shape([input_size // 8, input_size // 8, 3, 10])
    gt2.set_shape([input_size // 16, input_size // 16, 3, 10])
    gt3.set_shape([input_size // 32, input_size // 32, 3, 10])
    return img, gt1, gt2, gt3


def set_shapes_tiny(img, gt1, gt2, input_size):
    """
    Resizes the given image and ground truth tensors to their expected shapes for a Tiny YOLOv3 network.

    Args:
        img: A tensor representing the input image with shape [H, W, C], where H, W are the height and width of the
            image in pixels, and C is the number of channels.
        gt1: A tensor representing the ground truth labels for the output feature map of size 1/16 of the input image.
            The tensor has shape [H/16, W/16, 3, 10], where H and W are the height and width of the input image in
            pixels, 3 is the number of anchor boxes at each cell in the feature map, and 10 is the number of elements
            in each label (4 for the bounding box coordinates, 1 for the objectness score, and 5 for the class
            probabilities).
        gt2: A tensor representing the ground truth labels for the output feature map of size 1/32 of the input image.
            The tensor has shape [H/32, W/32, 3, 10], where H and W are the height and width of the input image in
            pixels, 3 is the number of anchor boxes at each cell in the feature map, and 10 is the number of elements
            in each label (4 for the bounding box coordinates, 1 for the objectness score, and 5 for the class
            probabilities).
        input_size: An integer representing the expected input size of the image in pixels. This is the size that the
            image will be resized to in order to be fed into the Tiny YOLOv3 network.

    Returns:
        A tuple containing the resized image tensor and the ground truth tensors with their shapes set to the expected
        values for the Tiny YOLOv3 network.
    """
    img.set_shape([input_size, input_size, 3])
    gt1.set_shape([input_size // 16, input_size // 16, 3, 10])
    gt2.set_shape([input_size // 32, input_size // 32, 3, 10])
    return img, gt1, gt2


def load_tfds(
        cfg,
        bbox_to_gt_func,
        split='train',
        return_xy=False,
        batch_size=32,
        debug=False):
    """
    Loads a TensorFlow dataset from a given configuration.

    Args:
        cfg: A configuration object that contains settings for loading the dataset.
        bbox_to_gt_func: A function that maps bounding boxes to ground truth data.
        split: The split to use for loading data (e.g. 'train', 'val', or 'test').
        return_xy: Whether to return image data with bounding boxes in the output.
        batch_size: The batch size to use when loading data.
        debug: Whether to run in debug mode (slower, but useful for debugging).

    Returns:
        A TensorFlow dataset object containing image data and ground truth data.
    """

    data = get_splits(cfg.data.labels_path, cfg.data.dataset, split)
    img_path = osp.join(cfg.data.path, 'cropped_images', str(cfg.model.input_size))
    img_paths = [osp.join(img_path, folder, name) for (folder, name) in zip(data.img_folder, data.img_name)]

    xys = np.zeros((len(data), 7, 3))  # third column for visibility
    data.xy = data.xy.apply(np.array)
    for i, _xy in enumerate(data.xy):
        xys[i, :_xy.shape[0], :2] = _xy
        xys[i, :_xy.shape[0], 2] = 1
    xys = xys.astype(np.float32)

    if return_xy:
        dtypes = [tf.float32 for _ in range(2)]
    else:
        if cfg.model.tiny:
            dtypes = [tf.float32 for _ in range(3)]
        else:
            dtypes = [tf.float32 for _ in range(4)]

    AUTO = tf.data.experimental.AUTOTUNE if not debug else 1
    ds = tf.data.Dataset.from_tensor_slices((img_paths, xys))
    ds = ds.shuffle(10000).repeat()

    ds = ds.map(lambda path, xy:
                tf.py_function(
                    lambda path, xy: preprocess(path, xy, cfg, bbox_to_gt_func, split, return_xy),
                    [path, xy], dtypes),
                num_parallel_calls=AUTO)

    input_size = int(img_path.split('/')[-1])

    if not return_xy:
        if cfg.model.tiny:
            ds = ds.map(lambda img, gt1, gt2:
                        set_shapes_tiny(img, gt1, gt2, input_size),
                        num_parallel_calls=AUTO)
        else:
            ds = ds.map(lambda img, gt1, gt2, gt3:
                        set_shapes(img, gt1, gt2, gt3, input_size),
                        num_parallel_calls=AUTO)

    ds = ds.batch(batch_size).prefetch(AUTO)
    ds = data_generator(iter(ds), len(data), cfg.model.tiny) if not return_xy else ds
    return ds


class data_generator():
    """Wrap the tensorflow dataset in a generator so that we can combine
    gt into list because that's what the YOLOv4 loss function requires"""
    def __init__(self, tfds, n, tiny):
        self.tfds = tfds
        self.tiny = tiny
        self.n = n

    def __iter__(self):
        return self

    def __len__(self):
        return self.n

    def __next__(self):
        if self.tiny:
            img, gt1, gt2 = next(self.tfds)
            gt = [gt1, gt2]
        else:
            img, gt1, gt2, gt3 = next(self.tfds)
            gt = [gt1, gt2, gt3]
        return img, gt


if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.random.set_seed(0)
    np.random.seed(0)

    cfg = CN(new_allowed=True)
    cfg.merge_from_file('configs/aug_d2/tiny480_d2_20e_warp.yaml')

    from train import build_model

    yolo = build_model(cfg)

    yolo_dataset_object = yolo.load_dataset('dummy_dataset.txt', label_smoothing=0.)
    bbox_to_gt_func = yolo_dataset_object.bboxes_to_ground_truth

    ds = load_tfds(
        cfg,
        bbox_to_gt_func,
        split='train',
        return_xy=True,
        batch_size=1,
        debug=True)

    # for i, (img, (gt1, gt2)) in enumerate(ds):
    #     print(i, img.shape)
    #     print(gt1.shape, gt2.shape)
    #     img = (img.numpy()[0] * 255.).astype(np.uint8)[:, :, [2, 1, 0]]
    #     cv2.imshow('', img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    for img, xy in ds:
        img = img[0].numpy()
        xy = xy[0].numpy()

        img = (img * 255.).astype(np.uint8)
        xy = xy[xy[:, -1] == 1, :2]
        img = draw(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), xy, cfg, False, True)

        cv2.imshow('', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
