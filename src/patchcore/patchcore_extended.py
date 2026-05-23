"""PatchCoreの拡張版：複数メモリバンク切り替え対応"""

import logging
import os
import pickle

import numpy as np
import torch

import patchcore.common
import patchcore.patchcore
from patchcore.memory_manager import MemoryBankManager

LOGGER = logging.getLogger(__name__)


class PatchCoreWithMemorySwitching(patchcore.patchcore.PatchCore):
    """複数メモリバンク版を切り替え可能なPatchCore"""

    def __init__(self, device, memory_manager: MemoryBankManager = None):
        """
        Args:
            device: PyTorchデバイス
            memory_manager: MemoryBankManagerインスタンス（オプション）
        """
        super().__init__(device)
        self.memory_manager = memory_manager

    def load_from_path_with_manager(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN = None,
        prepend: str = "",
    ) -> None:
        """
        メモリバンク管理機能付きでモデルをロード

        Args:
            load_path: モデルディレクトリ
            device: PyTorchデバイス
            nn_method: 最寄り近傍探索の方法
            prepend: ファイル名プレフィックス
        """
        # ネットワークパラメータをロード
        LOGGER.info("Loading and initializing PatchCore with memory switching.")

        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)

        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]

        if nn_method is None:
            nn_method = patchcore.common.FaissNN(False, 4)

        self.load(**patchcore_params, device=device, nn_method=nn_method)

        # メモリバンク管理を初期化
        if self.memory_manager is None:
            self.memory_manager = MemoryBankManager(load_path)

        # アクティブなメモリバンクをロード
        active_memory_path = self.memory_manager.get_active_memory_bank_path()
        self.anomaly_scorer.load(os.path.dirname(active_memory_path), prepend)

        LOGGER.info(
            f"Loaded model from {load_path} with active version: "
            f"{self.memory_manager.active_version_id}"
        )

    def switch_memory_bank(self, version_id: str) -> None:
        """メモリバンク版を切り替え

        Args:
            version_id: 切り替え先の版ID
        """
        if self.memory_manager is None:
            raise RuntimeError("Memory manager not initialized")

        # 版をアクティブ化
        self.memory_manager.activate_version(version_id)

        # メモリバンクをロード
        memory_path = self.memory_manager.get_active_memory_bank_path()
        memory_dir = os.path.dirname(memory_path)

        # 新しいnn_methodを作成してロード
        nn_method = patchcore.common.FaissNN(
            self.anomaly_scorer.nn_method.on_gpu,
            self.anomaly_scorer.nn_method.num_workers if hasattr(
                self.anomaly_scorer.nn_method, "num_workers"
            ) else 4,
        )

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=self.anomaly_scorer.n_nearest_neighbours,
            nn_method=nn_method,
        )
        self.anomaly_scorer.load(memory_dir)

        LOGGER.info(f"Switched to memory bank version: {version_id}")

    def predict(self, data):
        """アクティブなメモリバンクで予測"""
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def list_available_memory_banks(self) -> list:
        """利用可能なメモリバンク版の一覧を取得"""
        if self.memory_manager is None:
            return []
        return self.memory_manager.list_versions()

    def get_current_memory_bank_info(self) -> dict:
        """現在のメモリバンク情報を取得"""
        if self.memory_manager is None:
            raise RuntimeError("Memory manager not initialized")
        return self.memory_manager.get_version_info(
            self.memory_manager.active_version_id
        )

    def compare_memory_banks(self, version_id_1: str, version_id_2: str) -> dict:
        """2つのメモリバンク版を比較"""
        if self.memory_manager is None:
            raise RuntimeError("Memory manager not initialized")
        return self.memory_manager.compare_versions(version_id_1, version_id_2)

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")
