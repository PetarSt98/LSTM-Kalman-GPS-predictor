import numpy as np
import torch
import onnxruntime

from config import GlobalConfig


def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:, 2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T

    # reset z-coordinate
    out[:, 2] = xyz[:, 2]

    return out


def expand_waypoints_and_get_timestamps(waypoints_old, image_timestamp_old):
    waypoints = torch.squeeze(waypoints_old.detach().cpu(), 0).numpy()
    waypoints_extended = np.insert(waypoints, [0], [0, 0], axis=0)
    waypoints_tstamps = image_timestamp_old + np.arange(0, len(waypoints_extended) * 0.5, 0.5) * 1e9
    return waypoints_extended, waypoints_tstamps


def calculate_angle(point, point_offset):
    return np.arctan2(point_offset[1] - point[1], point_offset[0] - point[0])


def bind_input_and_output(_x, ort_session):
    _io_binding = ort_session.io_binding()

    if GlobalConfig.should_use_attention:
        x_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(to_numpy(_x['speed']), 'cuda', 0)
        _io_binding.bind_ortvalue_input('speed', x_ortvalue)

    x_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(to_numpy(_x['target_point']), 'cuda', 0)
    _io_binding.bind_ortvalue_input('target_point', x_ortvalue)

    x_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(to_numpy(_x['img']), 'cuda', 0)
    _io_binding.bind_ortvalue_input('img', x_ortvalue)

    if GlobalConfig.should_use_3_cams:
        x_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(to_numpy(_x['img2']), 'cuda', 0)
        _io_binding.bind_ortvalue_input('img2', x_ortvalue)

        x_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(to_numpy(_x['img3']), 'cuda', 0)
        _io_binding.bind_ortvalue_input('img3', x_ortvalue)

    x_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(to_numpy(_x['lidar']), 'cuda', 0)
    _io_binding.bind_ortvalue_input('lidar', x_ortvalue)

    output_names = ort_session.get_outputs()[0].name
    _io_binding.bind_output(output_names, 'cuda')

    return _io_binding


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
