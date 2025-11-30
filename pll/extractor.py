import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    import open3d as o3d
except ImportError:
    o3d = None

from mask3d import InstanceSegmentation, get_model, prepare_data_tensor


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = ROOT_DIR / "Model" / "area1_scannet_pretrained.ckpt"


class Mask3DFeatureExtractor(InstanceSegmentation):
    """继承Mask3D实例分割网络，用于导出f_q和f_point"""

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: torch.device):
        base_model = get_model(checkpoint_path)
        base_model.__class__ = cls
        base_model.to(device)
        base_model.eval()
        base_model.checkpoint_path = checkpoint_path
        return base_model

    @torch.inference_mode()
    def extract_features(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, List[torch.Tensor]]:
        """执行一次前向，输出查询特征与点级特征"""
        data, features, _, _ = prepare_data_tensor(
            [points.to(device)],
            [colors.to(device)],
            device=device,
        )
        outputs = self(
            data,
            raw_coordinates=features[:, -3:],
        )
        f_q = [feat.detach().cpu() for feat in outputs["queries_embeddings"]]
        f_point = [
            feat.detach().cpu() for feat in outputs["backbone_features"]
        ]
        coords = outputs.get("sampled_coords", None)
        if coords is not None:
            coords = [c.detach().cpu() for c in coords]
        return {"f_q": f_q, "f_point": f_point, "coords": coords}

    def pretty_print(
        self,
        features: Dict[str, List[torch.Tensor]],
        tag: str = "sample",
        preview: int = 2,
    ):
        """打印特征维度与若干行内容"""
        for idx, (fq, fp) in enumerate(
            zip(features["f_q"], features["f_point"])
        ):
            print(f"[{tag}] 样本{idx}: f_q形状={tuple(fq.shape)}, f_point形状={tuple(fp.shape)}")
            fq_rows = min(preview, fq.shape[0])
            fp_rows = min(preview, fp.shape[0])
            print(f"  f_q前{fq_rows}行：\n{fq[:fq_rows]}")
            print(f"  f_point前{fp_rows}行：\n{fp[:fp_rows]}")
            if features["coords"]:
                coord = features["coords"][idx]
                print(
                    f"  采样坐标形状={tuple(coord.shape)}, 前{preview}行：\n{coord[:preview]}"
                )


def load_from_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    points = data["points"]
    colors = data.get("colors")
    if colors is None:
        colors = np.ones_like(points)
    return points, colors


def load_from_pth(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = torch.load(path)
    if isinstance(data, dict):
        points = data.get("points") or data.get("xyz")
        colors = data.get("colors") or data.get("rgb")
    else:
        points, colors = data
    if colors is None:
        colors = torch.ones_like(points)
    return points.numpy(), colors.numpy()


def load_from_ply(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if o3d is None:
        raise ImportError("读取PLY需要open3d，请先安装该依赖")
    mesh = o3d.io.read_point_cloud(str(path))
    points = np.asarray(mesh.points, dtype=np.float32)
    if mesh.has_colors():
        colors = np.asarray(mesh.colors, dtype=np.float32)
    else:
        colors = np.ones_like(points)
    return points, colors


def load_point_cloud(
    path: Path,
    max_points: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """统一读取点云并裁剪数量"""
    if path.suffix == ".npz":
        points, colors = load_from_npz(path)
    elif path.suffix in {".pth", ".pt"}:
        points, colors = load_from_pth(path)
    elif path.suffix == ".ply":
        points, colors = load_from_ply(path)
    else:
        raise ValueError(f"暂不支持的文件格式: {path.suffix}")

    if points.shape[0] > max_points:
        ids = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[ids]
        colors = colors[ids]

    if colors.max() > 1.0:
        colors = colors / 255.0

    points_t = torch.from_numpy(points).float().unsqueeze(0)
    colors_t = torch.from_numpy(colors).float().unsqueeze(0)
    return points_t, colors_t


def random_point_cloud(num_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """无文件时生成随机点云，方便联调"""
    points = torch.rand(1, num_points, 3)
    colors = torch.ones_like(points) * 0.5
    return points, colors


def parse_args():
    parser = argparse.ArgumentParser(description="Mask3D特征提取脚本")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="S3DIS权重路径",
    )
    parser.add_argument(
        "--pointcloud",
        type=str,
        default="",
        help="包含points/colors的npz|pth|ply文件",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=80000,
        help="最多采样多少个点参与提取",
    )
    parser.add_argument(
        "--random_points",
        type=int,
        default=4096,
        help="未提供文件时生成的随机点数",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_path = os.path.abspath(args.checkpoint)

    if args.pointcloud:
        pc_path = Path(args.pointcloud).expanduser()
        if not pc_path.exists():
            raise FileNotFoundError(f"找不到点云文件: {pc_path}")
        points, colors = load_point_cloud(pc_path, args.max_points)
        tag = pc_path.stem
    else:
        points, colors = random_point_cloud(args.random_points)
        tag = "random"

    extractor = Mask3DFeatureExtractor.from_checkpoint(
        checkpoint_path, device
    )
    features = extractor.extract_features(points, colors, device)
    extractor.pretty_print(features, tag=tag)


if __name__ == "__main__":
    main()

