import socket
import cv2
import numpy as np
import json
import torch

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.segment.general import process_mask, scale_image
from utils.torch_utils import select_device


class Config:
    MODEL_PATH = "/home/dsj/code/yolov5_modify/runs_edge_p2_sim/train-seg/exp/weights/best.pt"
    CONF_THRESH = 0.5
    IOU_THRESH = 0.4
    IMG_SIZE = 640
    DEVICE = "0" if torch.cuda.is_available() else "cpu"
    #DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class YOLOTCParser:

    def __init__(self, listen_ip, listen_port, config):

        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.config = config

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.listen_ip, self.listen_port))
        self.server_socket.listen(1)

        print(f"服务器启动，监听 {self.listen_ip}:{self.listen_port}")

        self.device = select_device(self.config.DEVICE)

        self.model = DetectMultiBackend(
            self.config.MODEL_PATH,
            device=self.device
        )

        self.stride = self.model.stride
        self.names = self.model.names
        self.imgsz = check_img_size(self.config.IMG_SIZE, s=self.stride)

        self.model.eval()

        print("YOLOv5-Seg 模型加载成功")

    # ===============================
    # 接收图像
    # ===============================

    def receive_image(self, client_socket):

        img_len_bytes = client_socket.recv(4)
        if not img_len_bytes:
            return None

        img_len = int.from_bytes(img_len_bytes, byteorder='big')

        img_bytes = b""
        while len(img_bytes) < img_len:

            chunk = client_socket.recv(
                min(4096, img_len - len(img_bytes))
            )

            if not chunk:
                return None

            img_bytes += chunk

        encode_img = np.frombuffer(img_bytes, dtype=np.uint8)

        image = cv2.imdecode(encode_img, cv2.IMREAD_COLOR)

        return image

    # ===============================
    # 发送结果
    # ===============================

    def send_result(self, client_socket, result):

        result_json = json.dumps(result, ensure_ascii=False)

        result_bytes = result_json.encode('utf-8')

        result_len = len(result_bytes)

        client_socket.sendall(
            result_len.to_bytes(4, byteorder='big')
        )

        client_socket.sendall(result_bytes)

    # ===============================
    # mask生成bbox
    # ===============================

    def mask_to_bbox(self, mask):

        mask = mask.astype(np.uint8)

        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            return None

        x1 = int(xs.min())
        x2 = int(xs.max())
        y1 = int(ys.min())
        y2 = int(ys.max())

        return [x1, y1, x2, y2]

    # ===============================
    # 推理
    # ===============================

    def inference(self, image):

        img0 = image.copy()

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #img = cv2.resize(img, (self.imgsz, self.imgsz))

        img = img.transpose(2, 0, 1)

        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device).float()

        img = img / 255.0

        img = img.unsqueeze(0)

        with torch.no_grad():

            out, pre_edge = self.model(img)

            preds, protos, train_out = out

        preds = non_max_suppression(
            preds,
            self.config.CONF_THRESH,
            self.config.IOU_THRESH,
            max_det=100,
            nm=32
        )

        pred = preds[0]

        if len(pred) == 0:
            return []

        proto = protos[0]

        pred_masks = process_mask(
            proto,
            pred[:, 6:],
            pred[:, :4],
            img.shape[2:]
        )

        pred_masks = scale_image(
            img.shape[2:],
            pred_masks.permute(1, 2, 0).cpu().numpy(),
            img0.shape[:2]
        )

        pred_masks = torch.from_numpy(pred_masks).permute(2, 0, 1)

        predn = pred.clone()

        scale_boxes(
            img.shape[2:],
            predn[:, :4],
            img0.shape
        ).round()

        detections = []

        for i in range(len(predn)):

            cls_id = int(predn[i, 5])

            cls_name = self.names[cls_id]

            conf = float(predn[i, 4])

            mask = pred_masks[i].cpu().numpy()

            mask_bin = (mask > 0.5).astype(np.uint8)

            bbox = self.mask_to_bbox(mask_bin)

            if bbox is None:
                continue

            mask_data = mask_bin.tolist()

            detections.append({

                "box": bbox,
                "cls_id": cls_id,
                "cls_name": cls_name,
                "conf": conf,
                "mask": mask_data

            })

        return detections

    # ===============================
    # 主循环
    # ===============================

    def run(self):

        while True:

            client_socket, client_addr = self.server_socket.accept()

            print(f"客户端连接: {client_addr}")

            try:

                while True:

                    image = self.receive_image(client_socket)

                    if image is None:
                        print("客户端断开")
                        break

                    detections = self.inference(image)

                    response = {

                        "success": True,
                        "detections": detections

                    }

                    self.send_result(client_socket, response)

            except Exception as e:

                print("错误:", e)

                error_result = {

                    "success": False,
                    "detections": [],
                    "error": str(e)

                }

                self.send_result(client_socket, error_result)

            finally:

                client_socket.close()


if __name__ == "__main__":

    config = Config()

    server = YOLOTCParser(
        listen_ip="0.0.0.0",
        listen_port=8866,
        config=config
    )

    server.run()