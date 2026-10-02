"""The geometry descriptor, extracted from the sibling project so this deposit stands alone.

Everything below is copied verbatim, by line range, from
EngageNet-P1-Temporal/experiments/src/geom_extract.py: the constants of lines 27
to 41, build_models (44 to 61), _ratio (82 to 86), head_pose (89 to 107),
frame_descriptor (110 to 139) and temporal_stats (142 to 155). Nothing is
reworded or re-derived, because the geometry numbers reported in the manuscript
come out of these bodies.

What is left behind is the part that reads a corpus: sample_frames, clip_features,
extract_split and merge_shards, together with the sibling project's own data
module, which only those functions use. geom96.py in this deposit does its own
frame sampling and calls the five names below.

Requires cv2, numpy, onnxruntime and insightface. The two ONNX models are the
buffalo_l pack's det_10g.onnx for detection and 1k3d68.onnx for the 68-point
three-dimensional landmarks.
"""
from __future__ import annotations

import os

import cv2  # noqa: F401  (kept so the import surface matches the source module)
import numpy as np
import onnxruntime as ort


MODEL_DIR = os.path.expanduser("~/.insightface/models/buffalo_l")
DET_SIZE = (320, 320)
THREADS = 8
PROVIDERS = (["CUDAExecutionProvider", "CPUExecutionProvider"]
             if "CUDAExecutionProvider" in ort.get_available_providers()
             else ["CPUExecutionProvider"])

# 68-point layout
JAW = slice(0, 17)
BROW_L, BROW_R = slice(17, 22), slice(22, 27)
NOSE_BRIDGE, NOSE_BOT = slice(27, 31), slice(31, 36)
EYE_L, EYE_R = slice(36, 42), slice(42, 48)
MOUTH_OUT, MOUTH_IN = slice(48, 60), slice(60, 68)


def build_models():
    from insightface.model_zoo.landmark import Landmark
    from insightface.model_zoo.scrfd import SCRFD
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.inter_op_num_threads = 1
    # the detector emits four shape warnings per frame; at 16 frames a clip that
    # is a megabyte of log per hundred clips and a synchronous write each time
    so.log_severity_level = 3
    dpath = os.path.join(MODEL_DIR, "det_10g.onnx")
    lpath = os.path.join(MODEL_DIR, "1k3d68.onnx")
    det = SCRFD(model_file=dpath,
                session=ort.InferenceSession(dpath, so, providers=PROVIDERS))
    det.prepare(ctx_id=-1, input_size=DET_SIZE, det_thresh=0.5)
    lmk = Landmark(model_file=lpath,
                   session=ort.InferenceSession(lpath, so, providers=PROVIDERS))
    lmk.prepare(ctx_id=-1)
    return det, lmk


def _ratio(pts, a, b, c, d):
    """Vertical opening over horizontal width, scale-free."""
    v = np.linalg.norm(pts[a] - pts[b])
    h = np.linalg.norm(pts[c] - pts[d]) + 1e-6
    return v / h


def head_pose(pts3):
    """Yaw, pitch and roll from the 3D landmarks by Procrustes alignment to the
    clip-independent canonical shape, which is the mean of the frontal template
    directions rather than a fitted model."""
    p = pts3 - pts3.mean(0)
    scale = np.linalg.norm(p) + 1e-6
    p = p / scale
    # axes from the face itself: x from eye to eye, y from brow to chin
    x = p[EYE_R].mean(0) - p[EYE_L].mean(0)
    y = p[JAW][8] - 0.5 * (p[BROW_L].mean(0) + p[BROW_R].mean(0))
    x = x / (np.linalg.norm(x) + 1e-6)
    y = y - x * (y @ x)
    y = y / (np.linalg.norm(y) + 1e-6)
    z = np.cross(x, y)
    R = np.stack([x, y, z])
    yaw = np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0]))
    pitch = np.arctan2(R[2, 1], R[2, 2])
    roll = np.arctan2(R[1, 0], R[0, 0])
    return np.array([yaw, pitch, roll], dtype=np.float32)


def frame_descriptor(pts3, bbox, frame_shape):
    """Shape-normalised landmarks plus interpretable scalars for one frame."""
    p = pts3.astype(np.float32)
    inter_ocular = np.linalg.norm(p[EYE_R].mean(0) - p[EYE_L].mean(0)) + 1e-6
    centred = (p - p.mean(0)) / inter_ocular          # 68 x 3, scale free

    ear_l = _ratio(p, 37, 41, 36, 39)
    ear_l2 = _ratio(p, 38, 40, 36, 39)
    ear_r = _ratio(p, 43, 47, 42, 45)
    ear_r2 = _ratio(p, 44, 46, 42, 45)
    mar = _ratio(p, 51, 57, 48, 54)
    mar_in = _ratio(p, 62, 66, 60, 64)
    brow_l = np.linalg.norm(p[BROW_L].mean(0) - p[EYE_L].mean(0)) / inter_ocular
    brow_r = np.linalg.norm(p[BROW_R].mean(0) - p[EYE_R].mean(0)) / inter_ocular
    mouth_w = np.linalg.norm(p[54] - p[48]) / inter_ocular
    nose_chin = np.linalg.norm(p[30] - p[8]) / inter_ocular
    jaw_w = np.linalg.norm(p[16] - p[0]) / inter_ocular
    asym = abs(ear_l - ear_r) + abs(brow_l - brow_r)

    h, w = frame_shape[:2]
    bw = (bbox[2] - bbox[0]) / w
    bh = (bbox[3] - bbox[1]) / h
    cx = (bbox[0] + bbox[2]) / 2 / w
    cy = (bbox[1] + bbox[3]) / 2 / h

    pose = head_pose(p)
    scalars = np.array([ear_l, ear_l2, ear_r, ear_r2, mar, mar_in, brow_l, brow_r,
                        mouth_w, nose_chin, jaw_w, asym, bw, bh, cx, cy],
                       dtype=np.float32)
    return centred.reshape(-1), np.concatenate([scalars, pose])


def temporal_stats(seq):
    """Mean, spread, range and motion of a per-frame quantity."""
    a = np.asarray(seq, dtype=np.float32)
    if len(a) < 2:
        z = np.zeros_like(a[0])
        return np.concatenate([a[0], z, z, z, z, z])
    d = np.diff(a, axis=0)
    t = np.arange(len(a), dtype=np.float32)
    t = (t - t.mean()) / (t.std() + 1e-6)
    slope = (t[:, None] * (a - a.mean(0))).mean(0)
    return np.concatenate([a.mean(0), a.std(0), a.min(0), a.max(0),
                           np.abs(d).mean(0), slope])


