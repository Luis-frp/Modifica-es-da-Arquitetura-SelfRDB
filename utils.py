import os
import warnings
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
import matplotlib.pyplot as plt


def save_image_pair(x0, x0_pred, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    n_image = min(4, x0.shape[0])
    fig, axes = plt.subplots(nrows=2, ncols=n_image, figsize=(n_image*2, 4))

    if n_image == 1:
        axes = axes[..., None]

    for i in range(n_image):
        # Converter de (C, H, W) para (H, W, C)
        img_x0 = x0[i].permute(1, 2, 0).cpu().numpy()
        img_pred = x0_pred[i].permute(1, 2, 0).cpu().numpy()
        
        # Se for RGB (3 canais), não usar cmap='gray'
        if img_x0.shape[-1] == 3:
            # Clip para [0,1] se necessário
            img_x0 = np.clip((img_x0 + 1) / 2, 0, 1)
            img_pred = np.clip((img_pred + 1) / 2, 0, 1)
            axes[0, i].imshow(img_x0)
            axes[1, i].imshow(img_pred)
        else:
            # Grayscale
            axes[0, i].imshow(img_x0.squeeze(), cmap='gray')
            axes[1, i].imshow(img_pred.squeeze(), cmap='gray')
        
        axes[0, i].axis('off')
        axes[1, i].axis('off')

    plt.tight_layout(pad=0.1)
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()


def save_eval_images(
    source_images,
    target_images,
    pred_images,
    psnrs,
    ssims,
    save_path
):
    h, w = 30, 30
    zoom_region = [100-w, 100+w, 100-h, 100+h]
    zoom_size = [0, -0.4, 1, 0.47]

    # Squeeze channel dimension apenas se for grayscale (1 canal)
    if source_images.shape[1] == 1:
        source_images = source_images.squeeze()
        target_images = target_images.squeeze()
        pred_images = pred_images.squeeze()

    # If images between [-1, 1], scale to [0, 1]
    if np.nanmin(source_images) < -0.1:
        source_images = ((source_images + 1) / 2).clip(0, 1)

    if np.nanmin(target_images) < -0.1:
        target_images = ((target_images + 1) / 2).clip(0, 1)

    if np.nanmin(pred_images) < -0.1:
        pred_images = ((pred_images + 1) / 2).clip(0, 1)
    
    plt.style.use('dark_background')

    for i in range(len(source_images)):
        fig, ax = plt.subplots(1, 3, figsize=(12*1.5,8*1.5))
        
        # Detectar se é RGB ou grayscale
        is_rgb = source_images[i].ndim == 3 and source_images[i].shape[0] == 3
        
        if is_rgb:
            # Para RGB, transpor de (C, H, W) para (H, W, C)
            src_img = np.transpose(source_images[i], (1, 2, 0))
            tgt_img = np.transpose(target_images[i], (1, 2, 0))
            pred_img = np.transpose(pred_images[i], (1, 2, 0))
            
            # REMOVIDO: mean_norm que distorcia a imagem
            ax[0].imshow(src_img)
            ax[1].imshow(tgt_img)
            ax[2].imshow(pred_img)
        else:
            # Para grayscale, usar zoom e a normalização original
            ax_zoomed(ax[0], mean_norm(source_images[i]), zoom_region, zoom_size)
            ax_zoomed(ax[1], mean_norm(target_images[i]), zoom_region, zoom_size)
            ax_zoomed(ax[2], mean_norm(pred_images[i]), zoom_region, zoom_size)
        
        ax[0].set_title('Source')
        ax[1].set_title('Target')
        ax[2].set_title(f'PSNR: {psnrs[i]:.2f}\nSSIM: {ssims[i]:.2f}')

        # Save figure
        path = os.path.join(save_path, 'sample_images', f'slice_{i}.png')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)


def save_preds(preds, path):
    if not isinstance(preds, np.ndarray):
        preds = np.array(preds)

    # Normalize predictions
    preds = ((preds + 1) / 2).clip(0, 1)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, preds)


def to_norm(x):
    x = x/2
    x = x + 0.5
    return x.clip(0, 1)

def norm_01(x):
    return (x - x.min(axis=(-1,-2), keepdims=True))/(x.max(axis=(-1,-2), keepdims=True) - x.min(axis=(-1,-2), keepdims=True))


def mean_norm(x):
    """
    Normalização baseada na média.
    Para RGB de endoscopia: preservar estrutura de cor
    """
    # Para RGB, não usar abs() que destrói contraste
    if x.ndim >= 3 and (x.shape[0] == 3 or x.shape[-1] == 3):
        # Normalizar por canal para preservar cores
        mean_val = x.mean(axis=(-1,-2), keepdims=True)
        epsilon = 1e-8
        result = x / (mean_val + epsilon)
    else:
        # Para grayscale, manter comportamento original
        x = np.abs(x)
        mean_val = x.mean(axis=(-1,-2), keepdims=True)
        epsilon = 1e-8
        result = x / (mean_val + epsilon)
    
    # Replace any NaN or Inf values
    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
    return result


def apply_mask_and_norm(x, mask, norm_func):
    x = x*mask
    x = norm_func(x)
    return x


def center_crop(x, crop):
    h, w = x.shape[-2:]
    x = x[..., h//2-crop[0]//2:h//2+crop[0]//2, w//2-crop[1]//2:w//2+crop[1]//2]
    return x


def ax_zoomed(
    ax,
    im,
    zoom_region,
    zoom_size,
    zoom_edge_color='yellow'
):
    ax.imshow(np.flip(im, axis=0), origin='lower', cmap='gray')
    x1, x2, y1, y2 = zoom_region
    axins = ax.inset_axes(
        zoom_size,
        xlim=(x1, x2), ylim=(y1, y2))
    
    axins.imshow(np.flip(im, axis=0), cmap='gray')

    # Add border to zoomed region
    for spine in axins.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(2)
    
    # Remove inset axes ticks/labels
    axins.set_xticks([])
    axins.set_yticks([])
    
    ax.indicate_inset_zoom(axins, edgecolor=zoom_edge_color, linewidth=3)
    ax.axis('off')


def compute_metrics(
    gt_images,
    pred_images, 
    mask=None,
    norm='mean',
    subject_ids=None,
    report_path=None
):
    """ Compute PSNR and SSIM between gt_images and pred_images.
    
    Args:
        gt_images (torch.Tensor): Ground truth images.
        pred_images (torch.Tensor): Predicted images.
        mask (torch.Tensor): Mask to apply to images.
        crop (tuple): Center crop images to (Height, Width).
        norm (str): Normalization method. Options: 'mean', '01'.
        subject_ids (list): List of subject IDs for each slice.

    Returns:
        dict: Dictionary containing PSNR and SSIM values.
    
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        # If torch tensor, convert to numpy
        if isinstance(gt_images, torch.Tensor):
            gt_images = gt_images.cpu().numpy()

        if isinstance(pred_images, torch.Tensor):
            pred_images = pred_images.cpu().numpy()

        # Se imagens tiverem 4 dimensões com 1 canal, fazer squeeze
        # Se tiverem 3 canais (RGB), manter as 4 dimensões
        if gt_images.ndim == 4 and gt_images.shape[1] == 1:
            gt_images = gt_images.squeeze(axis=1)
        if pred_images.ndim == 4 and pred_images.shape[1] == 1:
            pred_images = pred_images.squeeze(axis=1)

        # If images 2-dimensional, add batch dimension
        gt_images = gt_images[None, ...] if gt_images.ndim == 2 else gt_images
        pred_images = pred_images[None, ...] if pred_images.ndim == 2 else pred_images

        assert gt_images.shape == pred_images.shape, \
            "Ground truth and predicted images must have the same shape"
        
        # Compute psnr and ssim
        psnr_values = []
        ssim_values = []

        # Normalize function
        if norm == 'mean':
            norm_func = mean_norm
        elif norm == '01':
            norm_func = norm_01

        # If images between [-1, 1], scale to [0, 1]
        if np.nanmin(gt_images) < -0.1:
            gt_images = ((gt_images + 1) / 2).clip(0, 1)

        if np.nanmin(pred_images) < -0.1:
            pred_images = ((pred_images + 1) / 2).clip(0, 1)

        # Apply mask and normalize
        if mask is not None:
            # Crop to mask shape
            gt_images = center_crop(gt_images, mask.shape[-2:])
            pred_images = center_crop(pred_images, mask.shape[-2:])

            gt_images = apply_mask_and_norm(gt_images, mask, norm_func)
            pred_images = apply_mask_and_norm(pred_images, mask, norm_func)
        # else:
        #     # REMOVIDO: Normalização dupla que causava métricas incorretas
        #     # As imagens já estão normalizadas para [0, 1]
        #     gt_images = norm_func(gt_images)
        #     pred_images = norm_func(pred_images)

        # Compute psnr and ssim
        for gt, pred in zip(gt_images, pred_images):
            # Para RGB (C, H, W), transpor para (H, W, C)
            if gt.ndim == 3 and gt.shape[0] == 3:
                gt = np.transpose(gt, (1, 2, 0))
                pred = np.transpose(pred, (1, 2, 0))
                # SSIM para RGB precisa de channel_axis
                psnr_value = psnr(gt, pred, data_range=gt.max())
                ssim_value = ssim(gt, pred, data_range=gt.max(), channel_axis=2)*100
            else:
                # Grayscale
                gt = gt.squeeze()
                pred = pred.squeeze()
                psnr_value = psnr(gt, pred, data_range=gt.max())
                ssim_value = ssim(gt, pred, data_range=gt.max())*100
            
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)

        # Convert list to numpy array
        psnr_values = np.asarray(psnr_values)
        ssim_values = np.asarray(ssim_values)

        # Compute subject reports
        subject_reports = {}
        if subject_ids is not None:
            for i in np.unique(subject_ids):
                idx = np.where(subject_ids == i)[0]
                subject_report = {
                    'psnrs': psnr_values[idx],
                    'ssims': ssim_values[idx],
                    'psnr_mean': np.nanmean(psnr_values[idx]),
                    'ssim_mean': np.nanmean(ssim_values[idx]),
                    'psnr_std': np.nanstd(psnr_values[idx]),
                    'ssim_std': np.nanstd(ssim_values[idx])
                }
                subject_reports[i] = subject_report
            
        # Compute mean and std values
        if subject_ids is not None:
            psnr_mean = np.nanmean([report['psnr_mean'] for report in subject_reports.values()])
            ssim_mean = np.nanmean([report['ssim_mean'] for report in subject_reports.values()])

            psnr_std = np.nanstd([report['psnr_mean'] for report in subject_reports.values()])
            ssim_std = np.nanstd([report['ssim_mean'] for report in subject_reports.values()])
        else:
            psnr_mean = np.nanmean(psnr_values)
            ssim_mean = np.nanmean(ssim_values)

            psnr_std = np.nanstd(psnr_values)
            ssim_std = np.nanstd(ssim_values)
        
        if report_path is not None:
            with open(report_path, 'w') as f:
                f.write(f'PSNR: {psnr_mean:.2f} ± {psnr_std:.2f}\n')
                f.write(f'SSIM: {ssim_mean:.2f} ± {ssim_std:.2f}\n')
                f.write('\n')

                if subject_ids is not None:
                    for subject_id, report in subject_reports.items():
                        f.write(f'Subject {subject_id}\n')
                        f.write(f'PSNR: {report["psnr_mean"]:.2f} ± {report["psnr_std"]:.2f}\n')
                        f.write(f'SSIM: {report["ssim_mean"]:.2f} ± {report["ssim_std"]:.2f}\n')
                        f.write('\n')         

        res = {
            'psnr_mean': psnr_mean,
            'ssim_mean': ssim_mean,
            'psnr_std': psnr_std,
            'ssim_std': ssim_std,
            'psnrs': psnr_values,
            'ssims': ssim_values,
            'subject_reports': subject_reports
        }

        return res
