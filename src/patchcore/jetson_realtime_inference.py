import cv2
import torch
import numpy as np
from patchcore.memory_manager import MemoryBankManager
from patchcore.patchcore_extended import PatchCoreWithMemorySwitching

# 1. 軽量な部品分類モデル
class PartClassifier:
    def __init__(self, model_path, labels):
        self.model = torch.jit.load(model_path).eval().cuda()  # TorchScript推奨（速度最優先）
        self.labels = labels

    def predict(self, img):
        img = cv2.resize(img, (128, 128))
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).cuda() / 255
        with torch.no_grad():
            out = self.model(tensor)
        idx = out.argmax().item()
        return self.labels[idx]

# 2. PatchCore wrapper with dynamic memory switching
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
        # メモリバンクバージョン指定も可能
        # model.switch_memory_bank('v1.2')
        image = cv2.resize(image, (224, 224))
        t = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0) / 255
        t = t.cuda()
        with torch.no_grad():
            scores, masks = model.predict(t)
        return scores, masks

def camera_input_loop(cam_idx, classifier, patchcore_wrapper, camera_fps=30):
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print("カメラが起動できません")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        # ① 部品分類
        part = classifier.predict(frame)
        # ② メモリバンク切り替えてPatchCore推論
        scores, masks = patchcore_wrapper.predict(part, frame)
        print(f"部品:{part} 異常:{scores[0]:.3f}")

        # 可視化例
        if masks is not None:
            vis = (masks[0] * 255).astype(np.uint8)
            vis = cv2.resize(vis, (frame.shape[1], frame.shape[0]))
            cv2.imshow("anomaly_map", vis)
        if cv2.waitKey(1) == 27:  # ESCで終了
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # -- セットアップ
    part_labels = ["A", "B", "C"]  # 部品クラスに応じて列挙
    # TorchScript (.pt) 軽量分類モデルパス/labels
    part_cls = PartClassifier("models/part_classifier.pt", part_labels)
    patchcore = PatchCoreRTWrapper(part_labels, models_dir="models")
    camera_input_loop(0, part_cls, patchcore)