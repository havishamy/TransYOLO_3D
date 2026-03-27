# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Validate a trained YOLOv5 segment model on a segment dataset.

Usage:
    $ bash data/scripts/get_coco.sh --val --segments  # download COCO-segments val split (1G, 5000 images)
    $ python segment/val.py --weights yolov5s-seg.pt --data coco.yaml --img 640  # validate COCO-segments

Usage - formats:
    $ python segment/val.py --weights yolov5s-seg.pt                 # PyTorch
                                      yolov5s-seg.torchscript        # TorchScript
                                      yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                      yolov5s-seg_openvino_label     # OpenVINO
                                      yolov5s-seg.engine             # TensorRT
                                      yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                      yolov5s-seg_saved_model        # TensorFlow SavedModel
                                      yolov5s-seg.pb                 # TensorFlow GraphDef
                                      yolov5s-seg.tflite             # TensorFlow Lite
                                      yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                      yolov5s-seg_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import subprocess
import sys
from multiprocessing.pool import ThreadPool
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch.nn.functional as F

from models.common import DetectMultiBackend
from models.yolo import SegmentationModel
from utils.callbacks import Callbacks
from utils.general import (
    LOGGER,
    NUM_THREADS,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    coco80_to_coco91_class,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import ConfusionMatrix, box_iou
from utils.plots import output_to_target, plot_val_study, colors
from utils.segment.dataloaders import create_dataloader
from utils.segment.general import mask_iou, process_mask, process_mask_native, scale_image
from utils.segment.metrics import Metrics, ap_per_class_box_and_mask
from utils.segment.plots import plot_images_and_masks
from utils.torch_utils import de_parallel, select_device, smart_inference_mode


def save_one_txt(predn, save_conf, shape, file):
    """Saves detection results in txt format; includes class, xywh (normalized), optionally confidence if `save_conf` is
    True.
    """
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")


def save_one_json(predn, jdict, path, class_map, pred_masks):
    """Saves a JSON file with detection results including bounding boxes, category IDs, scores, and segmentation masks.

    Example JSON result: {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}.
    """
    from pycocotools.mask import encode

    def single_encode(x):
        """Encodes binary mask arrays into RLE (Run-Length Encoding) format for JSON serialization."""
        rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    pred_masks = np.transpose(pred_masks, (2, 0, 1))
    with ThreadPool(NUM_THREADS) as pool:
        rles = pool.map(single_encode, pred_masks)
    for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
        jdict.append(
            {
                "image_id": image_id,
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
                "segmentation": rles[i],
            }
        )


def process_batch(detections, labels, iouv, pred_masks=None, gt_masks=None, overlap=False, masks=False):
    """Return correct prediction matrix.

    Args:
        detections (array[N, 6]): x1, y1, x2, y2, conf, class
        labels (array[M, 5]): class, x1, y1, x2, y2

    Returns:
        correct (array[N, 10]), for 10 IoU levels.
    """
    if masks:
        if overlap:
            nl = len(labels)
            index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
            gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
            gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
        if gt_masks.shape[1:] != pred_masks.shape[1:]:
            gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
            gt_masks = gt_masks.gt_(0.5)
        iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
    else:  # boxes
        iou = box_iou(labels[:, 1:], detections[:, :4])

    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

def visualize_single_image(edge, im, pred, pred_masks, names, save_path,edge_path, conf_thres=0.25, alpha=0.4):
    """
    可视化单张图片的实例分割结果并保存
    修复：处理预测格式不一致和无效类别索引
    """
    if isinstance(edge, torch.Tensor):
        edge = edge.cpu().numpy().astype(int)
    edge*=255
    edge = edge.transpose(1, 2, 0)
    # Build Image
    edge_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(edge_path, edge)
    # 1. 图像预处理（保持不变）
    if isinstance(im, torch.Tensor):
        im = im.cpu().float().numpy()
    
    if im.shape[0] == 3:
        im = im.transpose(1, 2, 0)
    
    if im.max() <= 1.0:
        im = (im * 255).astype(np.uint8)
    else:
        im = im.astype(np.uint8)
    
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    h, w = im.shape[:2]
    
    if pred is None or len(pred) == 0:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), im)
        return
    
    # 过滤低置信度
    mask = pred[:, 4] > conf_thres
    pred = pred[mask]
    if len(pred) == 0:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), im)
        return
    
    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks[mask]
    else:
        m = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        pred_masks = pred_masks[m]
    
    vis_img = im.copy()
    
    # 2. 遍历每个实例进行绘制
    for i, (p, mask_item) in enumerate(zip(pred, pred_masks)):
        # ===== 关键修复：安全解包预测值 =====
        p_list = p.tolist() if isinstance(p, torch.Tensor) else p
        
        # 确保至少有6个元素 [x1, y1, x2, y2, conf, cls]
        if len(p_list) < 6:
            LOGGER.warning(f"Skipping invalid detection format in {save_path.name}: length={len(p_list)}")
            continue
            
        # 提取值（防止额外维度如跟踪ID）
        x1, y1, x2, y2 = float(p_list[0]), float(p_list[1]), float(p_list[2]), float(p_list[3])
        conf = float(p_list[4])
        cls = int(p_list[5])
        # ==========================================
        
        # 检查类别有效性
        if cls < 0 or cls >= len(names):
            LOGGER.debug(f"Skipping invalid class index {cls} in {save_path.name}")
            continue
        
        label = f'{names[cls]} {conf:.2f}'
        
        # 获取颜色
        color = colors(cls, True)
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))
        
        # 处理掩码
        if isinstance(mask_item, torch.Tensor):
            mask_np = mask_item.cpu().numpy()
        else:
            mask_np = np.array(mask_item)
            
        if mask_np.shape != (h, w):
            mask_np = cv2.resize(mask_np.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        mask_bin = (mask_np > 0.5).astype(np.uint8)
        
        if mask_bin.sum() == 0:
            continue
        
        # 绘制半透明掩码
        colored_mask = np.zeros_like(vis_img)
        colored_mask[mask_bin > 0] = color_bgr
        vis_img = cv2.addWeighted(vis_img, 1.0, colored_mask, alpha, 0)
        
        # 绘制边界框（转换为整数坐标）
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color_bgr, 2)
        
        # 绘制标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        label_y = max(y1 - 5, text_h + 5)
        cv2.rectangle(vis_img, 
                     (x1, label_y - text_h - 4), 
                     (x1 + text_w + 8, label_y), 
                     color_bgr, -1)
        
        cv2.putText(vis_img, label, (x1 + 4, label_y - 2), 
                   font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        # 绘制掩码轮廓
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(vis_img, contours, -1, color_bgr, 1)
    
    # 保存图像
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), vis_img)


@smart_inference_mode()
def run(
    data,
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    task="val",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    verbose=False,  # verbose output
    save_txt=False,  # save results to *.txt
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_json=False,  # save a COCO-JSON results file
    project=ROOT / "runs/val-seg",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    model=None,
    dataloader=None,
    save_dir=Path(""),
    plots=True,
    overlap=False,
    mask_downsample_ratio=1,
    compute_loss=None,
    callbacks=Callbacks(),
    save_vis=True,           # 新增：是否保存单张可视化
    vis_conf=0.25,            # 新增：可视化置信度阈值
    vis_alpha=0.4,            # 新增：掩码透明度
):
    """Validate a YOLOv5 segmentation model on specified dataset, producing metrics, plots, and optional JSON output."""
    if save_json:
        check_requirements("pycocotools>=2.0.6")
        process = process_mask_native  # more accurate
    else:
        process = process_mask  # faster

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != "cpu"  # half precision only supported on CUDA
        model.half() if half else model.float()
        nm = de_parallel(model).model[-1].nm  # number of masks
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        # 新增：创建可视化保存目录
        if save_vis:
            vis_dir = save_dir / "visualization"
            vis_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"{colorstr('bold', 'Visualization:')} saving to {vis_dir}")
        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        nm = de_parallel(model).model.model[-1].nm if isinstance(model, SegmentationModel) else 32  # number of masks
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != "cpu"
    is_coco = isinstance(data.get("val"), str) and data["val"].endswith(f"coco{os.sep}val2017.txt")  # COCO dataset
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, (
                f"{weights} ({ncm} classes) trained on different --data than what you passed ({nc} "
                f"classes). Pass correct combination of --weights and --data that are trained together."
            )
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad, rect = (0.0, False) if task == "speed" else (0.5, pt)  # square inference for benchmarks
        task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images
        dataloader = create_dataloader(
            data[task],
            imgsz,
            batch_size,
            stride,
            single_cls,
            pad=pad,
            rect=rect,
            workers=workers,
            prefix=colorstr(f"{task}: "),
            overlap_mask=overlap,
            mask_downsample_ratio=mask_downsample_ratio,
        )[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, "names") else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    
    # -------------------------- 新增：mIoU 统计初始化 --------------------------
    # 记录每个类别的 IoU 列表
    class_iou_dict = defaultdict(list)
    # 记录全局 IoU 列表（所有匹配实例）
    global_iou_list = []
    # -------------------------------------------------------------------------
    
    s = ("%22s" + "%11s" * 13) % (  # 新增 mAP75 / mIoU 列
        "Class",
        "Images",
        "Instances",
        "Box(P",
        "R",
        "mAP50",
        "mAP50-95)",
        "Mask(P",
        "R",
        "mAP50",
        "mAP75",       # 新增列：Mask mAP@0.75
        "mAP50-95)",
        "IoU",
        "mIoU",
    )
    dt = Profile(device=device), Profile(device=device), Profile(device=device)
    metrics = Metrics()
    loss = torch.zeros(5, device=device)
    jdict, stats = [], []
    # -------------------------- 新增：大中小目标尺寸分类统计 --------------------------
    stats_s, stats_m, stats_l = [], [], []  # Small / Medium / Large AP 统计
    AREA_THR_S = 32 ** 2   # 小目标上限: area < 1024 px²
    AREA_THR_L = 96 ** 2   # 大目标下限: area ≥ 9216 px²
    global_iou_list_s, global_iou_list_m, global_iou_list_l = [], [], []  # 分尺寸 mIoU
    # -------------------------------------------------------------------------
    # callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    for batch_i, (im, targets, paths, shapes, masks,edges) in enumerate(pbar):
        # callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
                masks = masks.to(device)
                edges = edges.to(device)
            masks = masks.float()
            edges = edges.float()
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            re = model(im) if compute_loss else model(im, augment=augment)
            out, pre_edge=re
            preds, protos, train_out  = out
        # Loss
        if compute_loss:
            loss += compute_loss((train_out, protos, pre_edge),targets, masks,edges)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with dt[2]:
            preds = non_max_suppression(
                preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det, nm=nm
            )

        # Metrics
        plot_masks = []  # masks for plotting
        for si, (pred, proto) in enumerate(zip(preds, protos)):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct_masks = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            correct_bboxes = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct_masks, correct_bboxes, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                    # 无预测时收集分尺寸目标（作为漏检 FN 计入 recall 分母）
                    if not overlap:
                        _tbox0 = xywh2xyxy(labels[:, 1:5].clone())
                        scale_boxes(im[si].shape[1:], _tbox0, shape, shapes[si][1])
                        _areas0 = (_tbox0[:, 2] - _tbox0[:, 0]) * (_tbox0[:, 3] - _tbox0[:, 1])
                        _cats0 = torch.zeros(nl, dtype=torch.long, device=device)
                        _cats0[_areas0 >= AREA_THR_S] = 1
                        _cats0[_areas0 >= AREA_THR_L] = 2
                        _eb = torch.zeros((0, niou), dtype=torch.bool, device=device)
                        _ef = torch.zeros(0, device=device)
                        for _ci0, _szst0 in enumerate([stats_s, stats_m, stats_l]):
                            _sf0 = _cats0 == _ci0
                            if _sf0.any():
                                _szst0.append((_eb.clone(), _eb.clone(), _ef.clone(), _ef.clone(), labels[_sf0, 0]))
                continue

            # Masks
            midx = [si] if overlap else targets[:, 0] == si
            gt_masks = masks[midx]
            pred_masks = process(proto, pred[:, 6:], pred[:, :4], shape=im[si].shape[1:])

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct_bboxes = process_batch(predn, labelsn, iouv)
                correct_masks = process_batch(predn, labelsn, iouv, pred_masks, gt_masks, overlap=overlap, masks=True)

                # ---- 新增：计算大中小目标尺寸类别（在 mIoU 段修改 gt_masks 之前）----
                if not overlap:
                    _areas = (tbox[:, 2] - tbox[:, 0]) * (tbox[:, 3] - tbox[:, 1])
                    size_cats = torch.zeros(nl, dtype=torch.long, device=device)
                    size_cats[_areas >= AREA_THR_S] = 1   # Medium
                    size_cats[_areas >= AREA_THR_L] = 2   # Large
                    for _cat_i, _sz_st in enumerate([stats_s, stats_m, stats_l]):
                        _sz_mask = size_cats == _cat_i
                        if _sz_mask.any():
                            _cb_sz = process_batch(predn, labelsn[_sz_mask], iouv)
                            _cm_sz = process_batch(
                                predn, labelsn[_sz_mask], iouv,
                                pred_masks, gt_masks[_sz_mask], overlap=False, masks=True
                            )
                        else:
                            _cb_sz = torch.zeros(npr, niou, dtype=torch.bool, device=device)
                            _cm_sz = torch.zeros(npr, niou, dtype=torch.bool, device=device)
                        _sz_st.append((
                            _cm_sz, _cb_sz, pred[:, 4], pred[:, 5],
                            labels[_sz_mask, 0] if _sz_mask.any() else torch.zeros(0, device=device)
                        ))
                else:
                    size_cats = None
                # ---------------------------------------------------------------

                # -------------------------- 新增：计算掩码 IoU --------------------------
                # 调整真实掩码尺寸与预测掩码一致
                if gt_masks.shape[1:] != pred_masks.shape[1:]:
                    gt_masks_resized = F.interpolate(
                        gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False
                    )[0]
                    gt_masks_resized = gt_masks_resized.gt_(0.5)
                else:
                    gt_masks_resized = gt_masks.gt_(0.5)
                
                # 计算掩码 IoU 矩阵 (nl, npr)
                iou_matrix = mask_iou(
                    gt_masks_resized.view(gt_masks_resized.shape[0], -1),
                    pred_masks.view(pred_masks.shape[0], -1)
                )
                
                # 匹配真实掩码和预测掩码（同类别 + 最大 IoU）
                correct_class = labels[:, 0:1] == pred[:, 5].unsqueeze(1).T  # (nl, npr)
                iou_matrix = iou_matrix * correct_class  # 不同类别 IoU 置 0
                
                # 对每个真实掩码，找到最佳匹配的预测掩码
                for gt_idx in range(nl):
                    gt_cls = int(labels[gt_idx, 0].item())
                    # 找到同类别中 IoU 最大的预测掩码
                    max_iou, pred_idx = iou_matrix[gt_idx].max(dim=0)
                    if max_iou > 0:  # 存在有效匹配
                        class_iou_dict[gt_cls].append(max_iou.item())
                        global_iou_list.append(max_iou.item())
                        # 新增：分尺寸 mIoU 统计
                        if size_cats is not None:
                            _sz = int(size_cats[gt_idx].item())
                            [global_iou_list_s, global_iou_list_m, global_iou_list_l][_sz].append(max_iou.item())
                # -------------------------------------------------------------------------
                
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct_masks, correct_bboxes, pred[:, 4], pred[:, 5], labels[:, 0]))  # (conf, pcls, tcls)
            # ============================================================
            # 新增：保存单张图片可视化 (核心修改部分)
            # ============================================================
            if save_vis:
                # 使用原始未归一化的图像 (im[si] 此时是 0-1 或 0-255 的 tensor)
                img_tensor = im[si]
                edge_vis = pre_edge[si]
                # 生成保存路径 (保持原文件名)
                save_name = Path(paths[si]).name
                vis_save_path = save_dir / "visualization" / save_name
                vis_edge_path = save_dir / "visualization_edge" / save_name
                # 调用可视化函数
                visualize_single_image(
                    edge_vis,
                    img_tensor,
                    pred,           # 原始预测 (包含所有检测)
                    pred_masks,     # 对应掩码
                    names,
                    vis_save_path,
                    vis_edge_path,
                    conf_thres=vis_conf,
                    alpha=vis_alpha
                )
            # ============================================================

            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
            if plots and batch_i < 3:
                plot_masks.append(pred_masks[:15])  # filter top 15 to plot

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / "labels" / f"{path.stem}.txt")
            if save_json:
                pred_masks = scale_image(
                    im[si].shape[1:], pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(), shape, shapes[si][1]
                )
                save_one_json(predn, jdict, path, class_map, pred_masks)  # append to COCO-JSON dictionary
            # callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        # Plot images
        if plots and batch_i < 3:
            if len(plot_masks):
                plot_masks = torch.cat(plot_masks, dim=0)
            plot_images_and_masks(im, targets, masks, edges, paths, save_dir / f"val_batch{batch_i}_labels.jpg",save_dir / f"val_batch{batch_i}_edge_labels.jpg", names)
            plot_images_and_masks(
                im,
                output_to_target(preds, max_det=15),
                plot_masks,
                pre_edge,
                paths,
                save_dir / f"val_batch{batch_i}_pred.jpg",
                save_dir / f"val_batch{batch_i}_edge_pred.jpg",
                names,
            )  # pred

        # callbacks.run('on_val_batch_end')

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        results = ap_per_class_box_and_mask(*stats, plot=plots, save_dir=save_dir, names=names)
        metrics.update(results)
    nt = np.bincount(stats[4].astype(int), minlength=nc)  # number of targets per class

    # -------------------------- 新增：计算 mIoU 指标 --------------------------
    # 计算各类别 IoU 均值
    class_miou_dict = {}
    for cls_id in range(nc):
        iou_list = class_iou_dict.get(cls_id, [])
        class_miou_dict[cls_id] = np.mean(iou_list) if iou_list else 0.0
    
    # 计算全局 mIoU
    global_miou = np.mean(global_iou_list) if global_iou_list else 0.0
    # 分尺寸 mIoU
    global_miou_s = np.mean(global_iou_list_s) if global_iou_list_s else 0.0
    global_miou_m = np.mean(global_iou_list_m) if global_iou_list_m else 0.0
    global_miou_l = np.mean(global_iou_list_l) if global_iou_list_l else 0.0

    # 计算大中小目标 AP 指标
    def _compute_size_metrics(sz_stats):
        """计算单个尺寸类别的 AP 指标，失败时返回 None。"""
        if not sz_stats:
            return None
        try:
            sz_np = [torch.cat(x, 0).cpu().numpy() for x in zip(*sz_stats)]
            if len(sz_np[4]) == 0:   # 无该尺寸目标
                return None
            sz_res = ap_per_class_box_and_mask(*sz_np, plot=False, save_dir=save_dir, names=names)
            sz_met = Metrics()
            sz_met.update(sz_res)
            return sz_met
        except Exception as e:
            LOGGER.warning(f"尺寸分类 AP 计算失败: {e}")
            return None

    met_s = _compute_size_metrics(stats_s)
    met_m = _compute_size_metrics(stats_m)
    met_l = _compute_size_metrics(stats_l)
    # -------------------------------------------------------------------------

    # Print results
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 11  # 新增 mAP75 / IoU / mIoU 列
    # 打印全局结果（含 mAP75 和 mIoU）
    _mr_all = metrics.mean_results()  # (mp_b, mr_b, map50_b, map_b, mp_m, mr_m, map50_m, map_m)
    _map75_all = metrics.metric_mask.map75
    LOGGER.info(pf % ("all", seen, nt.sum(), *_mr_all[:7], _map75_all, _mr_all[7], global_miou, global_miou))
    if nt.sum() == 0:
        LOGGER.warning(f"WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels")

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(metrics.ap_class_index):
            cls_iou = class_miou_dict.get(c, 0.0)
            _cr = metrics.class_result(i)  # (p_b, r_b, ap50_b, ap_b, p_m, r_m, ap50_m, ap_m)
            _ap75_cls = metrics.metric_mask.all_ap[i, 5] if len(metrics.metric_mask.all_ap) else 0.0
            LOGGER.info(pf % (names[c], seen, nt[c], *_cr[:7], _ap75_cls, _cr[7], cls_iou, global_miou))

    # Print speeds
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}" % t)
        # 打印 mIoU 汇总
        LOGGER.info(f"\n==================== mIoU Metrics ====================")
        LOGGER.info(f"Global mIoU: {global_miou:.4f}")
        for cls_id in range(nc):
            cls_name = names.get(cls_id, f"class_{cls_id}")
            cls_iou = class_miou_dict[cls_id]
            LOGGER.info(f"{cls_name} IoU: {cls_iou:.4f}")
        LOGGER.info(f"=======================================================")

        # ---- 新增：大中小目标分割指标汇总 ----
        LOGGER.info(f"\n====== 大中小目标分割指标 (COCO-style: S<32², 32²≤M<96², L≥96²) ======")
        hdr = (f"{'Size':<10} {'Box-P':>8} {'Box-R':>8} {'Box-mAP50':>10} {'Box-mAP50-95':>13}"
               f" {'Mask-P':>8} {'Mask-R':>8} {'Mask-mAP50':>11} {'Mask-mAP75':>11} {'Mask-mAP50-95':>13} {'Mask-mIoU':>10}")
        LOGGER.info(hdr)
        for _sl, _sm, _smiou in [('Small',  met_s, global_miou_s),
                                  ('Medium', met_m, global_miou_m),
                                  ('Large',  met_l, global_miou_l)]:
            if _sm is not None and len(_sm.metric_mask.all_ap):
                _mr = _sm.mean_results()  # (mp_b, mr_b, map50_b, map_b, mp_m, mr_m, map50_m, map_m)
                _map75_sz = _sm.metric_mask.map75
                LOGGER.info(
                    f"{_sl:<10} {_mr[0]:>8.3f} {_mr[1]:>8.3f} {_mr[2]:>10.3f} {_mr[3]:>13.3f}"
                    f" {_mr[4]:>8.3f} {_mr[5]:>8.3f} {_mr[6]:>11.3f} {_map75_sz:>11.3f} {_mr[7]:>13.3f} {_smiou:>10.4f}"
                )
            else:
                LOGGER.info(
                    f"{_sl:<10} {'N/A':>8} {'N/A':>8} {'N/A':>10} {'N/A':>13}"
                    f" {'N/A':>8} {'N/A':>8} {'N/A':>11} {'N/A':>11} {'N/A':>13} {_smiou:>10.4f}"
                )
        LOGGER.info(f"=============================================================================")
        # -----------------------------------------------------------------------
    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    # callbacks.run('on_val_end')

    mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask, map_mask = metrics.mean_results()

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ""  # weights
        anno_json = str(Path("../datasets/coco/annotations/instances_val2017.json"))  # annotations
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions
        LOGGER.info(f"\nEvaluating pycocotools mAP... saving {pred_json}...")
        with open(pred_json, "w") as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            results = []
            for eval in COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "segm"):
                if is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # img ID to evaluate
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                results.extend(eval.stats[:2])  # update results (mAP@0.5:0.95, mAP@0.5)
            map_bbox, map50_bbox, map_mask, map50_mask = results
        except Exception as e:
            LOGGER.info(f"pycocotools unable to run: {e}")

    # Return results (新增 mIoU 到返回值)
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    final_metric = mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask, map_mask, global_miou  # 新增 global_miou
    return (*final_metric, *(loss.cpu() / len(dataloader)).tolist()), metrics.get_maps(nc), t


def parse_opt():
    """Parses command line arguments for configuring YOLOv5 options like dataset path, weights, batch size, and
    inference settings.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/home/dsj/dataset/mydata/server/mydata.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default="/home/dsj/code/yolov5_modify/runs_edge_p2_sim/train-seg/exp/weights/best.pt", help="model path(s)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--task", default="test", help="train, val, test, speed or study")
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-json", action="store_true", help="save a COCO-JSON results file")
    parser.add_argument("--project", default="/home/dsj/code/yolov5_modify/runs_edge_p2_sim/test-seg", help="save results to project/name")
    parser.add_argument("--name", default="exp_claude", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")

    # 新增可视化参数
    parser.add_argument("--save-vis", default=True,action="store_true", help="save visualization for each image separately")
    parser.add_argument("--vis-conf", type=float, default=0.25, help="confidence threshold for visualization (default: 0.25)")
    parser.add_argument("--vis-alpha", type=float, default=0.4, help="mask transparency for visualization, 0-1 (default: 0.4)")

    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    # opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 tasks including training, validation, testing, speed, and study with configurable options."""
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))

    if opt.task in ("train", "val", "test"):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.warning(f"WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results")
        if opt.save_hybrid:
            LOGGER.warning("WARNING ⚠️ --save-hybrid returns high mAP from hybrid labels, not from predictions alone")
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != "cpu"  # FP16 for fastest results
        if opt.task == "speed":  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == "study":  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f"study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt"  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f"\nRunning {f} --imgsz {opt.imgsz}...")
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt="%10.4g")  # save
            subprocess.run(["zip", "-r", "study.zip", "study_*.txt"])
            plot_val_study(x=x)  # plot
        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)