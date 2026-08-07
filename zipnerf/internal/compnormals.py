import sys
sys.path.append('pretrained/surface_normal_uncertainty')
from models.NNET import NNET
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

def load_checkpoint(fpath, model):
    ckpt = torch.load(fpath, map_location='cpu')['model']

    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)
    return model

class args:
    architecture = 'GN'
    sampling_ratio=0.4
    importance_ratio=0.7
    input_height=480
    input_width=640

device = torch.device('cuda:0')
# load checkpoint
checkpoint = 'pretrained/surface_normal_uncertainty/checkpoints/nyu.pt'
model = NNET(args).to(device)
model = load_checkpoint(checkpoint, model)
model.eval()

# def kappa_to_alpha(pred_kappa):
#     alpha = ((2 * pred_kappa) / ((pred_kappa ** 2.0) + 1)) \
#             + ((np.exp(- pred_kappa * np.pi) * np.pi) / (1 + np.exp(- pred_kappa * np.pi)))
#     alpha = np.degrees(alpha)
#     return alpha

# pred_alpha = kappa_to_alpha(pred_kappa)
def getNormals(imgpth):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = Image.fromarray(imgpth)
    orgsz = img.size
    img = img.convert("RGB").resize(size=(args.input_width, args.input_height), resample=Image.BILINEAR)
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = normalize(img)

    img = img.cuda().unsqueeze(0)
    with torch.no_grad():
        norm_out_list, _, _ = model(img)
    norm_out = norm_out_list[-1]

    pred_norm = norm_out[:, :3, :, :]
    pred_kappa = norm_out[:, 3:, :, :]
    pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()      # (B, H, W, 3)
    pred_kappa = pred_kappa.cpu().permute(0, 2, 3, 1).numpy()
    pred_norm = np.array(pred_norm).astype(np.float32)
    pred_norm_rgb = ((pred_norm + 1) * 0.5) * 255
    pred_norm_rgb = np.clip(pred_norm_rgb, a_min=0, a_max=255)
    pred_norm_rgb = pred_norm_rgb.astype(np.uint8)
    pred_norm_rgb = Image.fromarray(pred_norm_rgb[0]).resize(size=(orgsz[0], orgsz[1]), resample=Image.BILINEAR)

    # def kappa_to_alpha(pred_kappa):
    #     alpha = ((2 * pred_kappa) / ((pred_kappa ** 2.0) + 1)) \
    #             + ((np.exp(- pred_kappa * np.pi) * np.pi) / (1 + np.exp(- pred_kappa * np.pi)))
    #     alpha = np.degrees(alpha)
    #     return alpha
    # pred_alpha = kappa_to_alpha(pred_kappa)
    # pred_alpha = np.clip(pred_alpha, a_min=0, a_max=60)
    # pred_alpha = pred_alpha.astype(np.uint8)
    # pred_alpha = Image.fromarray(pred_alpha[0,:,:,0]).resize(size=(orgsz[0], orgsz[1]), resample=Image.BILINEAR)
    # return np.array(pred_norm_rgb), np.array(pred_alpha)
    return np.array(pred_norm_rgb)

