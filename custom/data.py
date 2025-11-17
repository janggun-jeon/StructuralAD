#custom.data
from pathlib import Path
from collections.abc import Sequence
from pandas import DataFrame
from torchvision.transforms.v2 import Transform

import torch
import gc

from anomalib.data.utils import Split, validate_path, LabelName, read_image
from anomalib.data.base.dataset import AnomalibDataset
from anomalib.data.image.mvtec import MVTecDataset, MVTec
from anomalib import TaskType

IMG_EXTENSIONS = (".png", ".PNG")

def make_custom_dataset(
    root: str | Path,
    split: str | Split | None = None,
    extensions: Sequence[str] | None = None,
) -> DataFrame:

    if extensions is None:
        extensions = IMG_EXTENSIONS

    root = validate_path(root)
    samples_list = [(str(root),) + f.parts[-3:] for f in root.glob(r"**/*") if f.suffix in extensions]
    if not samples_list:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    samples = DataFrame(samples_list, columns=["path", "split", "label", "image_path"])

    samples["image_path"] = samples.path + "/" + samples.split + "/" + samples.label + "/" + samples.image_path

    samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)

    mask_samples = samples.loc[samples.split == "ground_truth"].sort_values(
        by="image_path",
        ignore_index=True,
    )
    samples = samples[samples.split != "ground_truth"].sort_values(
        by="image_path",
        ignore_index=True,
    )

    samples["mask_path"] = ""
    if not mask_samples.empty:
        samples.loc[
            (samples.split == "test") & (samples.label_index == LabelName.ABNORMAL),
            "mask_path",
        ] = mask_samples.image_path.to_numpy()

    samples["mask_path"] = samples["mask_path"].apply(
        lambda path: path if Path(path).exists() else None
    )

    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples

class CustomAnomalibDataset(AnomalibDataset):
    def __getitem__(self, index: int) -> dict[str, str | torch.Tensor]:
        image_path = self.samples.iloc[index].image_path
        mask_path = self.samples.iloc[index].mask_path
        label_index = self.samples.iloc[index].label_index

        image = read_image(image_path, as_tensor=True)
        item = {"image_path": image_path, "label": label_index}

        if self.task == TaskType.CLASSIFICATION:
            if self.transform:
                item["image"] = self.transform(image)
            else:
                image
            del image
        elif self.task in {TaskType.DETECTION, TaskType.SEGMENTATION}:
            mask = (
                Mask(torch.zeros(image.shape[-2:])).to(torch.uint8)
                if label_index == LabelName.NORMAL
                else read_mask(mask_path, as_tensor=True)
            )
            item["image"], item["mask"] = self.transform(image, mask) if self.transform else (image, mask)

            if self.task == TaskType.DETECTION:
                # create boxes from masks for detection task
                boxes, _ = masks_to_boxes(item["mask"])
                item["boxes"] = boxes[0]
        else:
            msg = f"Unknown task type: {self.task}"
            raise ValueError(msg)
        
        gc.collect()
        
        return item

class BallDataset(CustomAnomalibDataset):
    def __init__(
        self,
        task: TaskType,
        root: Path | str = "./datasets/BALL",
        category: str = "ball",
        transform: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(task=task, transform=transform)
        
        self.root_category = Path(root) / Path(category)
        self.category = category
        self.split = split
        self.samples = make_custom_dataset(self.root_category, split=self.split, extensions=IMG_EXTENSIONS)
        
class Ball(MVTec):
    def __init__(self, root: Path | str = "./datasets/BALL", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = TaskType.CLASSIFICATION
        self.root = Path(root)
        self.category = kwargs.get('category', 'ball')
        self.num_workers = 0

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = BallDataset(
            task=TaskType.CLASSIFICATION,
            transform=self.train_transform,
            split=Split.TRAIN,
            root=self.root,
            category=self.category,
        )
        self.test_data = BallDataset(
            task=TaskType.CLASSIFICATION,
            transform=self.eval_transform,
            split=Split.TEST,
            root=self.root,
            category=self.category,
        )
        
class MVTec(MVTec):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = "./datasets/MVTec"
        self.category = "bottle"
        self.num_workers = 0