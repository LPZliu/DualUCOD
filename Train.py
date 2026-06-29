import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from lib.new_dino_network import Network
from utils.data_val import get_loader, test_dataset
from utils.utils import clip_gradient, get_coef, cal_ual
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from torch import optim
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ==================== Loss Functions ====================
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()

# ==================== DWT Module ====================
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        ll = x1 + x2 + x3 + x4
        lh = -x1 + x2 - x3 + x4
        hl = -x1 - x2 + x3 + x4
        hh = x1 - x2 - x3 + x4
        return ll, lh, hl, hh

def pad_to_even(tensor):
    _, _, h, w = tensor.shape
    pad_h = (2 - h % 2) % 2
    pad_w = (2 - w % 2) % 2
    if pad_h == 0 and pad_w == 0:
        return tensor
    padder = nn.ConstantPad2d((0, pad_w, 0, pad_h), 0)
    return padder(tensor)

# ==================== Batch-Compatible Wavelet Enhancement ====================
def wavelet_enhance_batch(img_tensor, gain=1.5):
    """
    Apply wavelet enhancement to a batch of RGB images [B, 3, H, W] on GPU.
    Returns enhanced image tensor of same shape.
    """
    device = img_tensor.device
    B, C, H, W = img_tensor.shape
    assert C == 3, "Input must be RGB"

    # Convert to YCbCr in PyTorch (approximate)
    transform = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ], device=device, dtype=img_tensor.dtype)

    img_flat = img_tensor.permute(0, 2, 3, 1).reshape(-1, 3)
    ycbcr_flat = torch.matmul(img_flat, transform.T)
    ycbcr = ycbcr_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)

    y = ycbcr[:, 0:1, :, :]
    cb = ycbcr[:, 1:2, :, :]
    cr = ycbcr[:, 2:3, :, :]

    y_padded = pad_to_even(y)
    dwt = DWT().to(device)
    ll, lh, hl, hh = dwt(y_padded)
    enhanced_y = ll + gain * (lh + hl + hh)

    enhanced_y = F.interpolate(enhanced_y, size=(H, W), mode='bilinear', align_corners=False)
    enhanced_ycbcr = torch.cat([enhanced_y, cb, cr], dim=1)

    inv_transform = torch.tensor([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ], device=device, dtype=img_tensor.dtype)

    enhanced_flat = enhanced_ycbcr.permute(0, 2, 3, 1).reshape(-1, 3)
    rgb_flat = torch.matmul(enhanced_flat, inv_transform.T)
    enhanced_rgb = rgb_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)
    enhanced_rgb = torch.clamp(enhanced_rgb, 0.0, 1.0)
    return enhanced_rgb

# ==================== Training Function ====================
def train(train_loader, model, optimizer, epoch, save_path, writer, device):
    global step, total_step, opt
    model.train()
    loss_all = 0
    epoch_step = 0

    for i, (images, init_gts, _, names) in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images = images.to(device)
        refined_gts = init_gts.to(device)

        with torch.no_grad():
            aug_images = wavelet_enhance_batch(images)

        preds, preds2, con_loss, att_mask = model(images, 'train', aug_images)

        ual_coef = get_coef(iter_percentage=i / total_step, method='cos')
        ual_loss = cal_ual(seg_logits=preds[4], seg_gts=refined_gts)
        ual_loss *= ual_coef

        loss_init = (structure_loss(preds[0], refined_gts) * 0.0625 +
                     structure_loss(preds[1], refined_gts) * 0.125 +
                     structure_loss(preds[2], refined_gts) * 0.25 +
                     structure_loss(preds[3], refined_gts) * 0.5)
        loss_pre = (structure_loss(preds[5], 1 - refined_gts) * 0.0625 +
                    structure_loss(preds[6], 1 - refined_gts) * 0.125 +
                    structure_loss(preds[7], 1 - refined_gts) * 0.25 +
                    structure_loss(preds[8], 1 - refined_gts) * 0.5)

        loss_init_aug = (structure_loss(preds2[0], refined_gts) * 0.0625 +
                         structure_loss(preds2[1], refined_gts) * 0.125 +
                         structure_loss(preds2[2], refined_gts) * 0.25 +
                         structure_loss(preds2[3], refined_gts) * 0.5)
        loss_pre_aug = (structure_loss(preds2[5], 1 - refined_gts) * 0.0625 +
                        structure_loss(preds2[6], 1 - refined_gts) * 0.125 +
                        structure_loss(preds2[7], 1 - refined_gts) * 0.25 +
                        structure_loss(preds2[8], 1 - refined_gts) * 0.5)

        loss_final = structure_loss(preds[4], refined_gts)
        loss_final_aug = structure_loss(preds2[4], refined_gts)
        loss = loss_init + loss_final + 2 * ual_loss + loss_pre + loss_init_aug + loss_pre_aug + loss_final_aug

        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        step += 1
        epoch_step += 1
        loss_all += loss.item()

        if i % 20 == 0 or i == total_step or i == 1:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.format(
                datetime.now(), epoch, opt.epoch, i, total_step, loss.item()))
            logging.info('[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.format(
                epoch, opt.epoch, i, total_step, loss.item()))
            writer.add_scalars('Loss_Statistics',
                               {'Loss_init': loss_init.item(),
                                'Loss_final': loss_final.item(),
                                'Con_loss': con_loss.item(),
                                'Loss_total': loss.item()},
                               global_step=step)

    loss_all /= epoch_step
    logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
    writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
    if epoch % 80 == 0:
        # Save safely for DP or non-DP
        state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        torch.save(state_dict, save_path + f'Net_epoch_{epoch}.pth')

# ==================== Validation Function ====================
def val(test_loader, model, epoch, save_path, writer):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name, _ = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            result, _, _, _ = model(image, 'test')

            res = F.interpolate(result[4], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.mean(np.abs(res - gt))

        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', mae, global_step=epoch)
        print('Epoch: {}, MAE: {:.4f}, bestMAE: {:.4f}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            # Save best model safely
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, save_path + 'Net_epoch_best.pth')
            print(f'Save best model at epoch {epoch}.')

        logging.info('[Val Info]: Epoch:{} MAE:{:.4f} bestEpoch:{} bestMAE:{:.4f}'.format(epoch, mae, best_epoch, best_mae))

# ==================== Main ====================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--trainsize', type=int, default=476)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--decay_epoch', type=int, default=5)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--gpu_id', type=str, default='0, 1')
    parser.add_argument('--train_root', type=str, default='./data/TrainDataset/')
    parser.add_argument('--val_root', type=str, default='./data/TestDataset/COD10K/')
    parser.add_argument('--save_path', type=str, default='./checkpoint/')
    opt = parser.parse_args()

    # Parse GPU IDs correctly
    gpu_ids = [int(x) for x in opt.gpu_id.split(',')]
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    print(f'USE GPU {opt.gpu_id}')
    cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model on device
    model = Network(channels=64).to(device)

    # Wrap with DataParallel if multiple GPUs
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(len(gpu_ids))))
        print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")

    # Load pretrained model if specified
    if opt.load:
        state_dict = torch.load(opt.load, map_location=device)
        if isinstance(model, torch.nn.DataParallel):
            # Load directly if model is DP
            model.load_state_dict(state_dict)
        else:
            # Remove 'module.' prefix if loading DP-trained checkpoint on single GPU
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        print('Loaded model from', opt.load)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    save_path = opt.save_path
    os.makedirs(save_path, exist_ok=True)

    train_loader = get_loader(
        image_root=opt.train_root + 'Imgs/',
        gt_root=opt.train_root + 'PseudoMask_Enhanced/',
        edge_root=opt.train_root + 'Edge/',
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        num_workers=8
    )
    val_loader = test_dataset(
        image_root=opt.val_root + 'Imgs/',
        gt_root=opt.val_root + 'GT/',
        testsize=opt.trainsize
    )
    total_step = len(train_loader)

    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train with Self-Refined PseudoLabelPool (Student-only)")

    step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_mae = 1.0
    best_epoch = 0

    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=1e-6)

    print("Start training...")
    for epoch in range(1, opt.epoch + 1):
        cur_lr = cosine_schedule.get_last_lr()[0]
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        logging.info(f'>>> current lr: {cur_lr}')

        train(train_loader, model, optimizer, epoch, save_path, writer, device)
        val(val_loader, model, epoch, save_path, writer)

        cosine_schedule.step()

    writer.close()