import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_boxes,
)
from utils.plots import colors
from utils.segment.general import process_mask
from utils.torch_utils import select_device


def mask_to_bbox(mask):

    mask = mask.astype(np.uint8)

    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())

    return [x1, y1, x2, y2]


def visualize_single_image(edge, im, pred, pred_masks, names, save_path, edge_path, conf_thres=0.25, alpha=0.4):

    if isinstance(edge, torch.Tensor):
        edge = edge.cpu().numpy()

    edge = edge * 255
    edge = edge.transpose(1, 2, 0)

    edge_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(edge_path), edge)

    if isinstance(im, torch.Tensor):
        im = im.cpu().numpy()

    if im.shape[0] == 3:
        im = im.transpose(1, 2, 0)

    im = (im * 255).astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    h, w = im.shape[:2]

    vis_img = im.copy()

    for i, (p, mask_item) in enumerate(zip(pred, pred_masks)):
        if p[4] < conf_thres:
            continue

        x1, y1, x2, y2, conf, cls = p[:6]

        cls = int(cls)

        label = f"{names[cls]} {conf:.2f}"

        color = colors(cls, True)
        color = (int(color[2]), int(color[1]), int(color[0]))

        mask_np = mask_item.cpu().numpy()

        if mask_np.shape != (h, w):
            mask_np = cv2.resize(mask_np, (w, h))

        mask_bin = mask_np > 0.5
        x1, y1, x2, y2 = mask_to_bbox(mask_bin)

        colored_mask = np.zeros_like(vis_img)
        colored_mask[mask_bin] = color

        vis_img = cv2.addWeighted(vis_img, 1.0, colored_mask, alpha, 0)

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

        cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), vis_img)


def load_images(source):

    imgs = []

    for ext in ["*.jpg", "*.png", "*.jpeg", "*.bmp"]:
        imgs.extend(list(Path(source).glob(ext)))

    return imgs


def run(weights, source, save_dir, imgsz=640, conf_thres=0.25, iou_thres=0.5, device="0"):

    device = select_device(device)

    model = DetectMultiBackend(weights, device=device)

    stride = model.stride
    imgsz = check_img_size(imgsz, s=stride)

    names = model.names

    model.eval()

    image_paths = load_images(source)

    print("Total images:", len(image_paths))

    for img_path in tqdm(image_paths):
        img0 = cv2.imread(str(img_path))

        img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (imgsz, imgsz))

        img = img.transpose(2, 0, 1)

        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device).float()

        img = img / 255.0

        img = img.unsqueeze(0)

        with torch.no_grad():
            out, pre_edge = model(img)

            preds, protos, _train_out = out

        preds = non_max_suppression(preds, conf_thres, iou_thres, max_det=300, nm=32)

        pred = preds[0]

        if len(pred) == 0:
            continue

        proto = protos[0]

        pred_masks = process_mask(proto, pred[:, 6:], pred[:, :4], img.shape[2:])

        predn = pred.clone()

        scale_boxes(img.shape[2:], predn[:, :4], img0.shape).round()

        save_path = Path(save_dir) / "visualization" / img_path.name
        edge_path = Path(save_dir) / "edge" / img_path.name

        visualize_single_image(pre_edge[0], img0, predn, pred_masks, names, save_path, edge_path)


def parse_opt():

    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="predict_results")

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.5)
    parser.add_argument("--device", default="0")

    return parser.parse_args()


def main():

    opt = parse_opt()

    run(**vars(opt))


if __name__ == "__main__":
    main()
