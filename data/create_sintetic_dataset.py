import os
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm
import cv2




def split_images_to_npy(
    data_root,
    output_root,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    img_extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tif"),
    target_size=(256, 256)
):
    random.seed(seed)

    data_root = Path(data_root)
    output_root = Path(output_root)

    classes = [d for d in os.listdir(data_root) if (data_root / d).is_dir()]
    print(f"Classes encontradas: {classes}")

    for cls in classes:
        class_dir = data_root / cls
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(img_extensions)]

        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        # Criar estrutura de diretórios
        for split in ["train", "val", "test"]:
            (output_root / cls / split).mkdir(parents=True, exist_ok=True)

        print(f"[OK] {cls}: {n_train} train | {n_val} val | {n_test} test")

        idx = 0  # reset index for each class

        # Função para salvar
        def process_and_save(img_list, split):
            nonlocal idx

            for img_name in tqdm(img_list, desc=f"{cls} - {split}"):

                img_path = class_dir / img_name
                # MODIFICADO: Ler como RGB em vez de grayscale
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

                if img is None:
                    print(f"[AVISO] Erro ao ler {img_path}")
                    continue

                # Resize padronizado
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                
                # Converter BGR para RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0  # normalize
                out_path = output_root / cls / split / f"{idx}.npy"
                idx += 1

                # Salvar como (H, W, 3) para RGB
                np.save(out_path, img)

        process_and_save(train_imgs, "train")
        process_and_save(val_imgs, "val")
        process_and_save(test_imgs, "test")



def main():
    data_root = "/mnt/dados/Translation_Model/SelfRDB/dataset/animal_faces"
    output_root = "/mnt/dados/Translation_Model/SelfRDB/dataset/ANIMAL_FACES_NPY"

    split_images_to_npy(
        data_root=data_root,
        output_root=output_root,
        train_ratio=0.8,
        val_ratio=0.10,
        test_ratio=0.10,
        target_size=(256, 256)  # <-- padroniza tudo
    )


if __name__ == "__main__":
    main()
