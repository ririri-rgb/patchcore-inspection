import Jetson.GPIO as GPIO
import time
import cv2
import torch
import numpy as np

# GPIOピン定義
INPUT_PIN = 17  # 物理スイッチ IN (BOARD番号の場合確認)
OUTPUT_PIN = 27 # 異常アラート OUT

# GPIOセットアップ
GPIO.setmode(GPIO.BCM)
GPIO.setup(INPUT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # プルアップ入力
GPIO.setup(OUTPUT_PIN, GPIO.OUT, initial=GPIO.LOW)        # 初期LOW

def patchcore_initialize():
    # PatchCore/部品分類/メモリバンク等の初期化処理
    # ... 省略（ここに前回までのコードを実装）
    return patchcore_model, part_classifier

def process_frame(frame, patchcore_model, part_classifier):
    # (部品分類＋PatchCore推論)
    label = part_classifier.predict(frame)
    scores, masks = patchcore_model.predict(label, frame)
    return scores[0]  # 1枚のみ

def main():
    patchcore_model, part_classifier = patchcore_initialize()
    cap = cv2.VideoCapture(0)
    try:
        while True:
            # ① スイッチONで撮影＋推論
            if GPIO.input(INPUT_PIN) == GPIO.LOW:  # ボタン押下時 (ONがLOWの配線の例)
                ret, frame = cap.read()
                if not ret:
                    continue
                score = process_frame(frame, patchcore_model, part_classifier)
                print(f"Score: {score:.3f}")
                
                # ② 異常ならGPIO出力ON、正常ならOFF
                if score > 0.5:
                    GPIO.output(OUTPUT_PIN, GPIO.HIGH)
                    print("!! 異常検知 !!")
                else:
                    GPIO.output(OUTPUT_PIN, GPIO.LOW)
                
                time.sleep(0.5)  # 指定秒数に1回推論
            else:
                # スイッチOFFのときはアラートも消す
                GPIO.output(OUTPUT_PIN, GPIO.LOW)
                time.sleep(0.1)
    finally:
        GPIO.cleanup()
        cap.release()

if __name__ == "__main__":
    main()