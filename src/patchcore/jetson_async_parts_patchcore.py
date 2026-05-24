import cv2
import torch
import numpy as np
import threading
import queue
from patchcore.memory_manager import MemoryBankManager
from patchcore.patchcore_extended import PatchCoreWithMemorySwitching

# ==== 共有キュー ====
rawframe_q = queue.Queue(maxsize=5)   # カメラ画像 (raw BGR)
infer_q = queue.Queue(maxsize=5)      # (画像,label)ペア

# ==== 1. カメラ画像取得スレッド ====
def camera_reader(cam_idx=0):
    cap = cv2.VideoCapture(cam_idx)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        if not rawframe_q.full():
            rawframe_q.put(frame)
        # 終了条件（例:ファイル終端等あればbreak）

# ==== 2. 部品分類スレッド ====
class PartClassifier:
    def __init__(self, model_path, labels):
        self.model = torch.jit.load(model_path).eval().cuda()
        self.labels = labels

    def predict(self, img):
        img = cv2.resize(img, (128, 128))
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).cuda() / 255
        with torch.no_grad():
            out = self.model(tensor)
        idx = out.argmax().item()
        return self.labels[idx]

def part_classifier_worker(classifier):
    while True:
        frame = rawframe_q.get()
        label = classifier.predict(frame)
        infer_q.put((frame, label))
        rawframe_q.task_done()

# ==== 3. PatchCore推論スレッド ====
class PatchCoreRTWrapper:
    def __init__(self, part_keys, models_dir, device="cuda:0"):
        self.device = torch.device(device)
        self.managers = {k: MemoryBankManager(f"{models_dir}/{k}") for k in part_keys}
        self.models = {}
        for part in part_keys:
            m = PatchCoreWithMemorySwitching(self.device, memory_manager=self.managers[part])
            m.load_from_path_with_manager(f"{models_dir}/{part}", self.device)
            self.models[part] = m

    def predict(self, part_key, image):
        model = self.models[part_key]
        image = cv2.resize(image, (224, 224))
        t = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0) / 255
        t = t.cuda()
        with torch.no_grad():
            scores, masks = model.predict(t)
        return scores, masks

def patchcore_worker(patchcore_wrapper):
    while True:
        frame, part = infer_q.get()
        if part not in patchcore_wrapper.models:
            print("Unknown part:", part)
            continue
        scores, masks = patchcore_wrapper.predict(part, frame)
        print(f"[Result] 部品:{part} 異常:{scores[0]:.3f}")
        # 結果をUIや保存など後段処理に回したい場合はここでqueueに流してもOK
        infer_q.task_done()

# ==== メインスレッドでスレッド起動 ====
if __name__ == "__main__":
    part_labels = ["A", "B", "C"]
    classifier = PartClassifier("models/part_classifier.pt", part_labels)
    patchcore = PatchCoreRTWrapper(part_labels, models_dir="models")
    cam_thread = threading.Thread(target=camera_reader, args=(0,), daemon=True)
    classify_thread = threading.Thread(target=part_classifier_worker, args=(classifier,), daemon=True)
    patchcore_thread = threading.Thread(target=patchcore_worker, args=(patchcore,), daemon=True)

    cam_thread.start()
    classify_thread.start()
    patchcore_thread.start()

    # メインはキーボード割込や例外で止まるまでsleepや監視専用
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("終了...")
