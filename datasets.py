from abc import abstractmethod
import os
import yaml
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning as L

import torchvision.transforms as T
import torch


# def simple_sharpen(img, amount=1.0):
#     """
#     img: (H, W, C) normalizada em [0,1]
#     sharpening leve sem LAB, sem unsharp mask.
#     """
#     import cv2

#     kernel = np.array([
#         [0,    -1,     0],
#         [-1,  5+amount, -1],
#         [0,    -1,     0]
#     ], dtype=np.float32)

#     out = np.zeros_like(img)
#     for c in range(img.shape[2]):
#         out[:, :, c] = cv2.filter2D(img[:, :, c], -1, kernel)
#     out = np.clip(out, 0, 1)
#     return out


class BaseDataset(Dataset):
    def __init__(
        self,
        data_dir,
        target_modality,
        source_modality,
        stage,
        image_size,
        norm=True,
        padding=True
    ):
        self.data_dir = data_dir
        self.target_modality= target_modality
        self.source_modality = source_modality
        self.stage = stage
        self.image_size = image_size
        self.norm = norm
        self.padding = padding
        self.original_shape = None

    @abstractmethod
    def _load_data(self, contrast):
        pass

    def _pad_data(self, data):
        H, W = data.shape[-2:]

        pad_top = (self.image_size - H) // 2
        pad_bottom = self.image_size - H - pad_top
        pad_left = (self.image_size - W) // 2
        pad_right = self.image_size - W - pad_left

        if data.ndim == 4:
            return np.pad(data, ((0, 0),(0, 0),
                                 (pad_top, pad_bottom),
                                 (pad_left, pad_right)))
        else:
            return np.pad(data, ((0, 0),
                                 (pad_top, pad_bottom),
                                 (pad_left, pad_right)))

    def _normalize(self, data):
        """Normalização simples RGB [-1,1]"""
        if data.shape[1] == 3:
            if data.max() > 1.0:
                data = data / 255.0

            normalized = np.zeros_like(data)
            for c in range(3):
                ch = data[:, c]
                mn, mx = ch.min(), ch.max()
                if mx > mn:
                    normalized[:, c] = 2 * ((ch - mn)/(mx - mn)) - 1
                else:
                    normalized[:, c] = ch
            return normalized
        else:
            return (data - 0.5) / 0.5


class NumpyDataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        target_modality,
        source_modality,
        stage,
        image_size,
        norm=True,
        padding=True,
    ):
        super().__init__(
            data_dir,
            target_modality,
            source_modality,
            stage,
            image_size,
            norm,
            padding
        )


        self.target = self._load_data(self.target_modality)
        self.source = self._load_data(self.source_modality)

        # garante mesmo número de amostras
        min_len = min(len(self.target), len(self.source))
        self.target = self.target[:min_len]
        self.source = self.source[:min_len]

        self.original_shape = self.target.shape[-2:]

        self.transform = self._build_transforms()

        # if self.enhance_texture:
        #     print("[INFO] Aplicando sharpen simples nas imagens...")
        #     self.target = self._apply_sharpen(self.target)
        #     self.source = self._apply_sharpen(self.source)

        self.subject_ids = self._load_subject_ids('subject_ids.yaml')

        if self.padding:
            self.target = self._pad_data(self.target)
            self.source = self._pad_data(self.source)

        if self.norm:
            self.target = self._normalize(self.target)
            self.source = self._normalize(self.source)

    def _build_transforms(self):
        """Data augmentation somente para stage=train."""
        if self.stage != "train":
            return None

        return T.Compose([
            T.ToTensor(),  # converte para torch antes de transformar
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.1),
            T.RandomRotation(10),
            T.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
        ])


    #####################################################################
    # NOVO: Replace do unsharp-mask/LAB → sharpen RGB simples
    #####################################################################
    # def _apply_sharpen(self, images):
    #     out = np.zeros_like(images)
    #     for i, img in enumerate(images):
    #         # (C,H,W) → (H,W,C)
    #         hwc = np.transpose(img, (1,2,0))
    #         mn, mx = hwc.min(), hwc.max()

    #         if mx > 1.0:  # 0–255
    #             hwc = hwc / 255.0
    #         elif mn < 0:  # [-1,1]
    #             hwc = (hwc + 1) / 2

    #         sharpened = simple_sharpen(hwc, amount=self.texture_amount)
    #         out[i] = np.transpose(sharpened, (2,0,1))

    #     return out.astype(np.float32)

    def _load_data(self, contrast):
        data_dir = os.path.join(self.data_dir, contrast, self.stage)
        files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        data = []
        for file in files:
            img = np.load(os.path.join(data_dir, file))

            if img.ndim == 3 and img.shape[-1] == 3:
                img = np.transpose(img, (2,0,1))
            elif img.ndim == 3 and img.shape[-1] == 1:
                img = img.squeeze(-1)
                img = np.expand_dims(img, 0)
            elif img.ndim == 2:
                img = np.expand_dims(img, 0)
            elif img.ndim == 3 and img.shape[0] in [1,3]:
                pass
            else:
                raise ValueError(f"Formato inválido {img.shape}")

            data.append(img)

        return np.array(data).astype(np.float32)

    def _load_subject_ids(self, filename):
        path = os.path.join(self.data_dir, filename)
        if os.path.exists(path):
            with open(path,'r') as f:
                return np.array(yaml.load(f, Loader=yaml.FullLoader))
        return None

    def __len__(self):
        return len(self.source)

    def __getitem__(self, i):
        tgt = self.target[i]     # numpy (C,H,W)
        src = self.source[i]     # numpy (C,H,W)

        # (C,H,W) → (H,W,C) para seguir padrão do torchvision
        tgt = np.transpose(tgt, (1,2,0))
        src = np.transpose(src, (1,2,0))

        if self.transform is not None:
            # aplica augmentation
            tgt = self.transform(tgt)
            src = self.transform(src)
        else:
            # conversão simples para tensor
            tgt = torch.from_numpy(np.transpose(tgt, (2,0,1))).float()
            src = torch.from_numpy(np.transpose(src, (2,0,1))).float()

        return tgt, src, i


#####################################################################
# DATAMODULE
#####################################################################
class DataModule(L.LightningDataModule):
    def __init__(
        self, 
        dataset_dir,
        source_modality,
        target_modality,
        dataset_class,
        image_size,
        padding,
        norm,
        train_batch_size=1,
        val_batch_size=1,
        test_batch_size=1,
        num_workers=1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_dir = dataset_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.source_modality = source_modality
        self.target_modality = target_modality
        self.image_size = image_size
        self.padding = padding
        self.norm = norm
        self.num_workers = num_workers

        self.dataset_class = globals()[dataset_class]

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = self.dataset_class(
                data_dir=self.dataset_dir,
                target_modality=self.target_modality,
                source_modality=self.source_modality,
                stage='train',
                image_size=self.image_size,
                padding=self.padding,
                norm=self.norm,
            )

            self.val_dataset = self.dataset_class(
                data_dir=self.dataset_dir,
                target_modality=self.target_modality,
                source_modality=self.source_modality,
                stage='val',
                image_size=self.image_size,
                padding=self.padding,
                norm=self.norm,
            )

        if stage == "test":
            self.test_dataset = self.dataset_class(
                data_dir=self.dataset_dir,
                target_modality=self.target_modality,
                source_modality=self.source_modality,
                stage='test',
                image_size=self.image_size,
                padding=self.padding,
                norm=self.norm,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
