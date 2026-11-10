"""Fine-tune the final FaceNet layers on the local, labeled student crops."""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "data" / "dataset"
DATA_DIR = ROOT / "data"
WEIGHTS_PATH = DATA_DIR / "finetuned_facenet.pt"
REPORT_PATH = DATA_DIR / "finetune_report.json"
SEED = 173
MAX_EPOCHS = 15
PATIENCE = 4


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


class FaceDataset(Dataset):
    def __init__(self, samples: list[Sample], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            face = image.convert("RGB").resize((160, 160), Image.Resampling.BILINEAR)
        return self.transform(face), sample.label


def split_dataset() -> tuple[list[Sample], list[Sample], list[str]]:
    random.seed(SEED)
    class_dirs = [
        path for path in DATASET_DIR.iterdir()
        if path.is_dir() and path.name.isdigit()
    ] if DATASET_DIR.exists() else []
    class_dirs.sort(key=lambda path: int(path.name))
    if len(class_dirs) < 2:
        raise ValueError("Fine-tuning needs at least two enrolled students.")

    train_samples: list[Sample] = []
    validation_samples: list[Sample] = []
    student_ids: list[str] = []
    for label, class_dir in enumerate(class_dirs):
        images = sorted(class_dir.glob("*.jpg"))
        if len(images) < 4:
            raise ValueError(
                f"Student folder {class_dir.name} has {len(images)} images; add at least 4 per student."
            )
        validation_count = max(1, min(len(images) - 2, round(len(images) * 0.2)))
        # Keep a per-student tail holdout so adjacent captured frames are less likely to leak across splits.
        train_samples.extend(Sample(path, label) for path in images[:-validation_count])
        validation_samples.extend(Sample(path, label) for path in images[-validation_count:])
        student_ids.append(class_dir.name)
    return train_samples, validation_samples, student_ids


def main() -> int:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    try:
        train_samples, validation_samples, student_ids = split_dataset()
    except (OSError, ValueError) as error:
        print(f"Dataset setup error: {error}", flush=True)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading pretrained VGGFace2 FaceNet weights on {device}…", flush=True)
    try:
        model = InceptionResnetV1(pretrained="vggface2", classify=False).to(device)
    except Exception as error:
        print(f"Could not load pretrained FaceNet weights: {error}", flush=True)
        return 3

    # Keep early general-purpose face features stable; adapt the final residual blocks and projection.
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.repeat_3, model.block8, model.last_linear, model.last_bn):
        for parameter in module.parameters():
            parameter.requires_grad = True

    classifier = nn.Linear(512, len(student_ids)).to(device)
    nn.init.normal_(classifier.weight, mean=0.0, std=0.01)
    nn.init.zeros_(classifier.bias)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=(127.5 / 255,) * 3, std=(128 / 255,) * 3),
    ])
    transform_validation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(127.5 / 255,) * 3, std=(128 / 255,) * 3),
    ])

    train_dataset = FaceDataset(train_samples, transform_train)
    validation_dataset = FaceDataset(validation_samples, transform_validation)
    label_counts = Counter(sample.label for sample in train_samples)
    sampling_weights = [1.0 / label_counts[sample.label] for sample in train_samples]
    sampler = WeightedRandomSampler(sampling_weights, num_samples=len(sampling_weights), replacement=True)
    batch_size = min(16, len(train_samples))
    drop_last = len(train_samples) > batch_size and len(train_samples) % batch_size == 1
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=drop_last
    )
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=0)

    backbone_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": 1e-5},
            {"params": classifier.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_accuracy = -1.0
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    completed_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        classifier.train()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                module.eval()
        total_loss = 0.0
        for faces, labels in train_loader:
            faces, labels = faces.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            embeddings = model(faces)
            logits = classifier(embeddings) * 16.0
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(backbone_parameters, max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach().item())

        model.eval()
        classifier.eval()
        correct = total = 0
        validation_loss = 0.0
        with torch.inference_mode():
            for faces, labels in validation_loader:
                faces, labels = faces.to(device), labels.to(device)
                logits = classifier(model(faces)) * 16.0
                validation_loss += float(criterion(logits, labels).item()) * len(labels)
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                total += len(labels)
        accuracy = correct / max(total, 1)
        average_val_loss = validation_loss / max(total, 1)
        completed_epochs = epoch
        print(
            f"Epoch {epoch}/{MAX_EPOCHS}: train_loss={total_loss / max(len(train_loader), 1):.4f} "
            f"val_loss={average_val_loss:.4f} val_accuracy={accuracy:.1%}",
            flush=True,
        )

        if accuracy > best_accuracy or (accuracy == best_accuracy and average_val_loss < best_loss):
            best_accuracy, best_loss = accuracy, average_val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
                if not key.startswith("logits.")
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print(f"Early stopping after {epoch} epochs.", flush=True)
                break

    if best_state is None:
        print("Training ended without a usable checkpoint.", flush=True)
        return 4

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = WEIGHTS_PATH.with_suffix(".pt.tmp")
    torch.save(best_state, temporary)
    temporary.replace(WEIGHTS_PATH)
    report = {
        "device": str(device),
        "backbone": "InceptionResnetV1 pretrained on VGGFace2",
        "fine_tuned_layers": ["repeat_3", "block8", "last_linear", "last_bn"],
        "students": len(student_ids),
        "train_images": len(train_samples),
        "validation_images": len(validation_samples),
        "validation_accuracy": best_accuracy,
        "validation_loss": best_loss,
        "epochs_completed": completed_epochs,
        "student_ids": student_ids,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved fine-tuned weights to {WEIGHTS_PATH}", flush=True)
    print(f"Best held-out validation accuracy: {best_accuracy:.1%}", flush=True)
    print("This validation split is small-data monitoring, not a deployment accuracy guarantee.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Fine-tuning interrupted; the last completed weights remain intact.", file=sys.stderr, flush=True)
        raise SystemExit(130)
