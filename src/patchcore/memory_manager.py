"""複数メモリバンク管理と版管理システム"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import patchcore.common

LOGGER = logging.getLogger(__name__)


class MemoryBankVersion:
    """単一のメモリバンク版を表現"""

    def __init__(
        self,
        version_id: str,
        faiss_path: str,
        metadata: Optional[Dict] = None,
        created_at: Optional[str] = None,
        status: str = "active",
    ):
        self.version_id = version_id  # e.g., "v1.0", "v1.1"
        self.faiss_path = faiss_path
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.now().isoformat()
        self.status = status  # "active", "archived", "rollback_point"

    def to_dict(self) -> Dict:
        return {
            "version_id": self.version_id,
            "faiss_path": self.faiss_path,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "status": self.status,
        }

    @staticmethod
    def from_dict(data: Dict) -> "MemoryBankVersion":
        return MemoryBankVersion(**data)


class MemoryBankManager:
    """複数メモリバンク版の一元管理"""

    def __init__(self, base_dir: str):
        """
        Args:
            base_dir: メモリバンク管理ディレクトリ
                     例: /models/bottle/
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # versions/配下にメモリバンクを保存
        self.versions_dir = self.base_dir / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

        # マニフェスト（版管理情報）
        self.manifest_file = self.base_dir / "manifest.json"
        self.manifest = self._load_manifest()

        # 現在使用中の版
        self.active_version_id = self._get_active_version()

    def register_memory_bank(
        self,
        version_id: str,
        faiss_path: str,
        metadata: Optional[Dict] = None,
        set_as_active: bool = False,
    ) -> str:
        """
        新しいメモリバンク版を登録

        Args:
            version_id: 版ID（e.g., "v1.0", "v1.1"）
            faiss_path: FAISSインデックスのパス
            metadata: メタデータ（精度、訓練日時など）
            set_as_active: 登録後すぐに有効化するか

        Returns:
            保存先パス
        """
        if version_id in self.manifest["versions"]:
            raise ValueError(f"Version '{version_id}' already exists")

        # メモリバンクをversions/に複製
        dest_faiss_path = self.versions_dir / f"{version_id}_index.faiss"
        shutil.copy(faiss_path, dest_faiss_path)

        version = MemoryBankVersion(
            version_id=version_id,
            faiss_path=str(dest_faiss_path),
            metadata=metadata,
        )

        self.manifest["versions"][version_id] = version.to_dict()

        if set_as_active:
            self.activate_version(version_id)

        self._save_manifest()

        LOGGER.info(
            f"Registered memory bank version: {version_id} at {dest_faiss_path}"
        )
        return str(dest_faiss_path)

    def activate_version(self, version_id: str) -> None:
        """メモリバンク版を有効化"""
        if version_id not in self.manifest["versions"]:
            raise ValueError(f"Version '{version_id}' not found")

        # 前のバージョンはアーカイブ
        if self.active_version_id:
            old_version = self.manifest["versions"][self.active_version_id]
            old_version["status"] = "archived"

        # 新しいバージョンをアクティブ化
        new_version = self.manifest["versions"][version_id]
        new_version["status"] = "active"

        self.manifest["active_version"] = version_id
        self.active_version_id = version_id

        self._save_manifest()
        LOGGER.info(f"Activated memory bank version: {version_id}")

    def rollback_to_version(self, version_id: str) -> None:
        """特定の版にロールバック"""
        if version_id not in self.manifest["versions"]:
            raise ValueError(f"Version '{version_id}' not found")

        version_data = self.manifest["versions"][version_id]
        if not os.path.exists(version_data["faiss_path"]):
            raise FileNotFoundError(
                f"FAISS file not found: {version_data['faiss_path']}"
            )

        self.activate_version(version_id)
        LOGGER.warning(f"Rolled back to version: {version_id}")

    def get_active_memory_bank_path(self) -> str:
        """現在のメモリバンクパスを取得"""
        if not self.active_version_id:
            raise RuntimeError("No active version set")

        version_data = self.manifest["versions"][self.active_version_id]
        return version_data["faiss_path"]

    def list_versions(self) -> List[Dict]:
        """登録済み版の一覧を返す"""
        return [
            {
                "version_id": vid,
                "status": data["status"],
                "created_at": data["created_at"],
                "metadata": data.get("metadata", {}),
            }
            for vid, data in sorted(self.manifest["versions"].items())
        ]

    def get_version_info(self, version_id: str) -> Dict:
        """特定の版の詳細情報を取得"""
        if version_id not in self.manifest["versions"]:
            raise ValueError(f"Version '{version_id}' not found")
        return self.manifest["versions"][version_id]

    def compare_versions(self, version_id_1: str, version_id_2: str) -> Dict:
        """2つの版を比較"""
        if version_id_1 not in self.manifest["versions"]:
            raise ValueError(f"Version '{version_id_1}' not found")
        if version_id_2 not in self.manifest["versions"]:
            raise ValueError(f"Version '{version_id_2}' not found")

        v1 = self.manifest["versions"][version_id_1]
        v2 = self.manifest["versions"][version_id_2]

        return {
            "version_1": {
                "id": version_id_1,
                "created_at": v1["created_at"],
                "metadata": v1.get("metadata", {}),
            },
            "version_2": {
                "id": version_id_2,
                "created_at": v2["created_at"],
                "metadata": v2.get("metadata", {}),
            },
        }

    def create_rollback_point(self, label: str = "") -> str:
        """現在のアクティブ版をロールバックポイントに設定"""
        if not self.active_version_id:
            raise RuntimeError("No active version set")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rollback_label = label or f"rollback_{timestamp}"

        version_data = self.manifest["versions"][self.active_version_id]
        version_data["status"] = "rollback_point"
        if "rollback_labels" not in version_data:
            version_data["rollback_labels"] = []
        version_data["rollback_labels"].append(
            {"label": rollback_label, "created_at": datetime.now().isoformat()}
        )

        self._save_manifest()
        LOGGER.info(f"Created rollback point: {rollback_label}")
        return rollback_label

    def delete_version(self, version_id: str, force: bool = False) -> None:
        """版を削除（アクティブ版は削除不可）"""
        if version_id == self.active_version_id and not force:
            raise ValueError("Cannot delete active version. Set force=True to override.")

        if version_id not in self.manifest["versions"]:
            raise ValueError(f"Version '{version_id}' not found")

        version_data = self.manifest["versions"][version_id]
        faiss_path = version_data["faiss_path"]

        # ファイルを削除
        if os.path.exists(faiss_path):
            os.remove(faiss_path)

        # マニフェストから削除
        del self.manifest["versions"][version_id]
        self._save_manifest()

        LOGGER.info(f"Deleted version: {version_id}")

    def export_version(self, version_id: str, export_dir: str) -> str:
        """版をエクスポート（バックアップ用）"""
        if version_id not in self.manifest["versions"]:
            raise ValueError(f"Version '{version_id}' not found")

        version_data = self.manifest["versions"][version_id]
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        src_faiss = version_data["faiss_path"]
        dest_faiss = export_dir / f"{version_id}_index.faiss"

        shutil.copy(src_faiss, dest_faiss)

        # メタデータもエクスポート
        metadata_file = export_dir / f"{version_id}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(version_data, f, indent=2)

        LOGGER.info(f"Exported version {version_id} to {export_dir}")
        return str(dest_faiss)

    def _get_active_version(self) -> Optional[str]:
        """アクティブな版を取得"""
        return self.manifest.get("active_version")

    def _load_manifest(self) -> Dict:
        """マニフェストを読み込む"""
        if self.manifest_file.exists():
            with open(self.manifest_file, "r") as f:
                return json.load(f)
        return {"versions": {}, "active_version": None}

    def _save_manifest(self) -> None:
        """マニフェストを保存"""
        with open(self.manifest_file, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def __repr__(self) -> str:
        return (
            f"MemoryBankManager(base_dir={self.base_dir}, "
            f"active_version={self.active_version_id})"
        )
