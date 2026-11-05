import io
import urllib.request
import zipfile

import cv2
import numpy as np
import yaml


def calib(seq):
    url = "https://download.ifi.uzh.ch/rpg/DSEC/train/%s/%s_calibration.zip" % (seq, seq)
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=120).read()))
    n = [x for x in z.namelist() if x.endswith("cam_to_cam.yaml")][0]
    return yaml.safe_load(z.read(n))


def compute_remapping(c, mapping):
    fx, fy, cx, cy = c["intrinsics"]["camRect0"]["camera_matrix"]
    points = np.asarray(mapping, dtype=np.float64).reshape(-1, 2)
    rays = np.stack(((points[:, 0] - cx) / fx, (points[:, 1] - cy) / fy, np.ones(len(points))))
    rays = np.linalg.solve(c["extrinsics"]["R_rect0"], rays)
    rays = np.linalg.solve(np.asarray(c["extrinsics"]["T_10"])[:3, :3].T, rays)
    rays = np.linalg.solve(np.asarray(c["extrinsics"]["R_rect1"]).T, rays)
    fx, fy, cx, cy = c["intrinsics"]["camRect1"]["camera_matrix"]
    pixels = np.stack((fx * rays[0] / rays[2] + cx, fy * rays[1] / rays[2] + cy), axis=-1)
    return pixels.reshape(mapping.shape).astype(np.float32)


def remap_img(img, remap):
    return cv2.remap(img, remap, None, interpolation=cv2.INTER_CUBIC)
