# 複数メモリバンク管理システム統合ガイド

本ガイドでは、PatchCoreを実運用環境で複数メモリバンク版を管理・切り替えする方法を説明します。

## 📋 目次

1. [概要](#概要)
2. [モデル構造](#モデル構造)
3. [訓練スクリプト統合](#訓練スクリプト統合)
4. [テスト環境での使用](#テスト環境での使用)
5. [本番環境での使用](#本番環境での使用)
6. [トラブルシューティング](#トラブルシューティング)

---

## 概要

### 複数メモリバンク管理とは

同一のニューラルネットワーク（backbone）に対して、異なるメモリバンク（訓練データの特徴量インデックス）を複数保持し、実行時に切り替える機能です。

### メリット

- **A/Bテスト**: 2つのメモリバンク版を本番環境で同時に試験
- **段階的更新**: 新しい訓練データで更新したメモリバンクをテスト後に本番適用
- **即座なロールバック**: 問題発生時に30秒以内で復旧
- **監査ログ**: すべての版変更を自動記録

---

## モデル構造

### 保存形式

```
models/mvtec_bottle/
├── patchcore_params.pkl          ← ネットワークパラメータ（全版で共用）
├── manifest.json                 ← 版管理メタデータ
└── versions/
    ├── v1.0_index.faiss         ← メモリバンク v1.0
    ├── v1.1_index.faiss         ← メモリバンク v1.1（現在使用中）
    ├── v1.2_index.faiss         ← メモリバンク v1.2
    └── v1.3_index.faiss         ← メモリバンク v1.3
```

### manifest.json の内容例

```json
{
  "active_version": "v1.1",
  "versions": {
    "v1.0": {
      "version_id": "v1.0",
      "faiss_path": "versions/v1.0_index.faiss",
      "metadata": {
        "auroc": 0.990,
        "training_date": "2024-01-15"
      },
      "created_at": "2024-01-15T10:30:00",
      "status": "archived"
    },
    "v1.1": {
      "version_id": "v1.1",
      "faiss_path": "versions/v1.1_index.faiss",
      "metadata": {
        "auroc": 0.995,
        "training_date": "2024-04-20"
      },
      "created_at": "2024-04-20T14:45:00",
      "status": "active",
      "rollback_labels": [
        {
          "label": "production_deployed",
          "created_at": "2024-08-15T09:00:00"
        }
      ]
    }
  }
}
```

---

## 訓練スクリプト統合

### ステップ1: 訓練実行（変更なし）

通常通りPatchCoreを訓練します：

```bash
python bin/run_patchcore.py \
  --gpu 0 --seed 0 --save_patchcore_model \
  --log_group IM224_WR50 \
  patch_core -b wideresnet50 -le layer2 -le layer3 \
  sampler -p 0.1 approx_greedy_coreset \
  dataset -d bottle mvtec /path/to/mvtec
```

### ステップ2: 訓練後にメモリバンク版を登録

```python
# bin/register_memory_bank.py（新規スクリプト）

import sys
from datetime import datetime
from patchcore.memory_manager import MemoryBankManager

def register_after_training(model_path, version_id, metadata=None):
    """訓練後にメモリバンク版を登録"""
    
    manager = MemoryBankManager(model_path)
    
    faiss_file = f"{model_path}/nnscorer_search_index.faiss"
    
    manager.register_memory_bank(
        version_id=version_id,
        faiss_path=faiss_file,
        metadata=metadata or {},
        set_as_active=False  # テスト環境のみで使用
    )
    
    print(f"✓ Memory bank {version_id} registered")

if __name__ == "__main__":
    model_path = sys.argv[1]  # 例: /path/to/models/mvtec_bottle
    version_id = sys.argv[2]   # 例: v1.3
    
    metadata = {
        "training_date": datetime.now().isoformat(),
        "training_images": 1000,
        "auroc": 0.993,
        "notes": "新しいデータ増幅手法を適用"
    }
    
    register_after_training(model_path, version_id, metadata)
```

**使用方法**:

```bash
python bin/register_memory_bank.py \
  /path/to/models/mvtec_bottle \
  v1.3
```

---

## テスト環境での使用

### A/Bテスト実施例

```python
# bin/ab_test_memory_banks.py（新規スクリプト）

import torch
import patchcore.metrics
from patchcore.patchcore_extended import PatchCoreWithMemorySwitching
from patchcore.memory_manager import MemoryBankManager

def run_ab_test(model_path, test_dataloader, versions):
    """2つのメモリバンク版を比較"""
    
    device = torch.device("cuda:0")
    manager = MemoryBankManager(model_path)
    model = PatchCoreWithMemorySwitching(device, memory_manager=manager)
    model.load_from_path_with_manager(model_path, device)
    
    results = {}
    
    for version_id in versions:
        print(f"\nTesting {version_id}...")
        
        # メモリバンク版を切り替え
        model.switch_memory_bank(version_id)
        
        # 推論実行
        scores, masks, labels_gt, masks_gt = model.predict(test_dataloader)
        
        # メトリクス計算
        auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
            scores, [l != "good" for l in labels_gt]
        )["auroc"]
        
        results[version_id] = {
            "auroc": auroc,
            "scores": scores,
            "masks": masks
        }
    
    # 結果表示
    print("\n" + "="*50)
    print("A/B Test Results")
    print("="*50)
    for version_id, result in results.items():
        print(f"{version_id}: AUROC = {result['auroc']:.4f}")
    
    # 勝者を決定
    winner = max(results.items(), key=lambda x: x[1]["auroc"])
    print(f"\n✓ Winner: {winner[0]} ({winner[1]['auroc']:.4f})")
    
    return results

if __name__ == "__main__":
    import sys
    from patchcore.datasets.mvtec import MVTecDataset, DatasetSplit
    
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    
    # テストデータローダー作成
    test_dataset = MVTecDataset(
        data_path,
        classname="bottle",
        imagesize=224,
        split=DatasetSplit.TEST
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=8
    )
    
    # A/Bテスト実行
    results = run_ab_test(model_path, test_dataloader, ["v1.1", "v1.3"])
```

**使用方法**:

```bash
python bin/ab_test_memory_banks.py \
  /path/to/models/mvtec_bottle \
  /path/to/mvtec
```

---

## 本番環境での使用

### 推論エンジン実装例

```python
# bin/inference_production.py（新規スクリプト）

import logging
import sys
import torch
from pathlib import Path

from patchcore.patchcore_extended import PatchCoreWithMemorySwitching
from patchcore.memory_manager import MemoryBankManager

LOGGER = logging.getLogger(__name__)

class ProductionInferenceEngine:
    """実運用向け推論エンジン"""
    
    def __init__(self, model_path, device, enable_monitoring=True):
        self.model_path = model_path
        self.device = device
        self.enable_monitoring = enable_monitoring
        
        # モデル初期化
        self.manager = MemoryBankManager(model_path)
        self.model = PatchCoreWithMemorySwitching(device, memory_manager=self.manager)
        self.model.load_from_path_with_manager(model_path, device)
        
        # 安全なロールバックポイント作成
        self.manager.create_rollback_point("production_started")
        
        LOGGER.info(
            f"Production engine initialized. "
            f"Active version: {self.manager.active_version_id}"
        )
    
    def predict(self, images):
        """推論実行"""
        scores, masks = self.model.predict(images)
        
        if self.enable_monitoring:
            self._log_inference_stats(scores)
        
        return scores, masks
    
    def switch_version(self, version_id):
        """メモリバンク版を切り替え"""
        try:
            self.model.switch_memory_bank(version_id)
            LOGGER.info(f"Switched to version: {version_id}")
        except Exception as e:
            LOGGER.error(f"Failed to switch version: {e}")
            self.emergency_rollback()
    
    def emergency_rollback(self):
        """緊急ロールバック"""
        LOGGER.critical("EMERGENCY ROLLBACK TRIGGERED")
        
        # 最後のロールバックポイントに復帰
        versions = self.manager.list_versions()
        stable_version = next(
            (v for v in versions if v["status"] == "rollback_point"),
            None
        )
        
        if stable_version:
            self.model.switch_memory_bank(stable_version["version_id"])
            LOGGER.info(f"Rolled back to {stable_version['version_id']}")
    
    def list_versions(self):
        """利用可能な版を一覧表示"""
        return self.model.list_available_memory_banks()
    
    def _log_inference_stats(self, scores):
        """推論統計を記録"""
        current_version = self.manager.active_version_id
        mean_score = scores.mean()
        anomaly_rate = (scores > 0.5).sum() / len(scores)
        
        LOGGER.info(
            f"Version: {current_version} | "
            f"Mean Score: {mean_score:.4f} | "
            f"Anomaly Rate: {anomaly_rate:.2%}"
        )

# 使用例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    model_path = "/path/to/models/mvtec_bottle"
    device = torch.device("cuda:0")
    
    engine = ProductionInferenceEngine(model_path, device)
    
    # インタラクティブなコマンドラインインタフェース
    while True:
        cmd = input("> ").strip()
        
        if cmd == "list":
            versions = engine.list_versions()
            for v in versions:
                mark = "→" if v["status"] == "active" else " "
                print(f"  {mark} {v['version_id']} ({v['status']})")
        
        elif cmd.startswith("switch "):
            version_id = cmd.split()[1]
            engine.switch_version(version_id)
        
        elif cmd == "rollback":
            engine.emergency_rollback()
        
        elif cmd == "quit":
            break
```

---

## トラブルシューティング

### Q1: ロールバックに時間がかかる

**原因**: FAISS インデックスが大きい場合、ディスク読み込みに時間がかかる

**対策**: メモリにインデックスをキャッシュする

```python
# キャッシング機能の追加例
class CachedMemoryBankManager(MemoryBankManager):
    def __init__(self, base_dir):
        super().__init__(base_dir)
        self._index_cache = {}
    
    def get_cached_index(self, version_id):
        if version_id not in self._index_cache:
            path = self.get_active_memory_bank_path()
            # FAISS インデックスをメモリにロード
            self._index_cache[version_id] = faiss.read_index(path)
        return self._index_cache[version_id]
```

### Q2: 複数プロセスで同時に版を切り替えると競合する

**原因**: manifest.json への同時アクセス

**対策**: ファイルロック機構を追加

```python
import fcntl

def _save_manifest(self):
    """ファイルロック付きでマニフェスト保存"""
    with open(self.manifest_file, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            json.dump(self.manifest, f, indent=2)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

### Q3: 異なるバックボーンモデル間でメモリバンクを共有できるか

**答**: できません。

メモリバンク内の特徴量は特定のネットワークアーキテクチャに固有です。

異なるバックボーン間での共有は、特徴量の意味が異なるため精度が低下します。

### Q4: メモリバンクのサイズはどのくらい？

| 訓練データ | メモリバンクサイズ | 備考 |
|----------|-----------------|------|
| 1,000枚 | ~80MB | 小規模 |
| 10,000枚 | ~800MB | 標準 |
| 100,000枚 | ~8GB | 大規模 |
| 1,000,000枚 | ~80GB | 超大規模 |

複数版を保持する場合は、ストレージに余裕を確保してください。

---

## チェックリスト

本番環境へのデプロイ前に確認:

- [ ] テスト環境でA/Bテスト完了
- [ ] ロールバックポイント作成済み
- [ ] manifest.json が正しく更新されている
- [ ] 複数メモリバンク版が正しくロードできる
- [ ] 緊急ロールバック手順を理解している
- [ ] ストレージに十分な空き容量がある
- [ ] ログ監視が設定されている

---

## 参考リンク

- `src/patchcore/memory_manager.py` - メモリバンク管理実装
- `src/patchcore/patchcore_extended.py` - 拡張PatchCore実装
- `examples/memory_bank_examples.py` - 実践例
