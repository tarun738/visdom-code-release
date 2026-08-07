import collections
import functools

import torch.optim
from internal import camera_utils
from internal import configs
from internal import datasets
from internal import image
from internal import math
from internal import models
from internal import ref_utils
from internal import stepfun
from internal import utils
import numpy as np
from torch.utils._pytree import tree_map, tree_flatten
from torch_scatter import segment_coo


class GradientScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, colors, sigmas, ray_dist):
        ctx.save_for_backward(ray_dist)
        return colors, sigmas

    @staticmethod
    def backward(ctx, grad_output_colors, grad_output_sigmas):
        (ray_dist,) = ctx.saved_tensors
        scaling = torch.square(ray_dist).clamp(0, 1)
        return grad_output_colors * scaling[..., None], grad_output_sigmas * scaling, None


def tree_reduce(fn, tree, initializer=0):
    return functools.reduce(fn, tree_flatten(tree)[0], initializer)


def tree_sum(tree):
    return tree_reduce(lambda x, y: x + y, tree, initializer=0)


def tree_norm_sq(tree):
    return tree_sum(tree_map(lambda x: torch.sum(x ** 2), tree))


def tree_norm(tree):
    return torch.sqrt(tree_norm_sq(tree))


def tree_abs_max(tree):
    return tree_reduce(
        lambda x, y: max(x, torch.abs(y).max().item()), tree, initializer=0)


def tree_len(tree):
    return tree_sum(tree_map(lambda z: np.prod(z.shape), tree))


def summarize_tree(tree, fn, ancestry=(), max_depth=3):
    """Flatten 'tree' while 'fn'-ing values and formatting keys like/this."""
    stats = {}
    for k, v in tree.items():
        name = ancestry + (k,)
        stats['/'.join(name)] = fn(v)
        if hasattr(v, 'items') and len(ancestry) < (max_depth - 1):
            stats.update(summarize_tree(v, fn, ancestry=name, max_depth=max_depth))
    return stats


def compute_data_loss(batch, renderings, config):
    """Computes data loss terms for RGB, normal, and depth outputs."""
    data_losses = []
    mask_losses = []
    depth_losses = []
    normal_losses = []
    stats = collections.defaultdict(lambda: [])

    # lossmult can be used to apply a weight to each ray in the batch.
    # For example: masking out rays, applying the Bayer mosaic mask, upweighting
    # rays from lower resolution images and so on.
    lossmult = batch['lossmult']
    lossmult = torch.broadcast_to(lossmult, batch['rgb'][..., :3].shape)
    if config.disable_multiscale_loss:
        lossmult = torch.ones_like(lossmult)
    mask_seen =  batch['rgb'].mean(-1) > -1
    lossmult = lossmult[mask_seen,:3]

    for rendering in renderings:
        resid_sq = ((rendering['rgb'][mask_seen,:3] - batch['rgb'][mask_seen, :3]) ** 2)
        denom = lossmult.sum().clamp_min(1)
        stats['mses'].append(((lossmult * resid_sq).sum() / denom).item())

        if config.data_loss_type == 'mse':
            # Mean-squared error (L2) loss.
            data_loss = resid_sq
        elif config.data_loss_type == 'charb':
            # Charbonnier loss.
            data_loss = torch.sqrt(resid_sq + config.charb_padding ** 2)
        elif config.data_loss_type == 'rawnerf':
            # Clip raw values against 1 to match sensor overexposure behavior.
            rgb_render_clip = rendering['rgb'].clamp_max(1)
            resid_sq_clip = (rgb_render_clip - batch['rgb'][..., :3]) ** 2
            # Scale by gradient of log tonemapping curve.
            scaling_grad = 1. / (1e-3 + rgb_render_clip.detach())
            # Reweighted L2 loss.
            data_loss = resid_sq_clip * scaling_grad ** 2
        else:
            assert False
        data_losses.append((lossmult * data_loss).sum() / denom)
        if config.compute_mask_loss:
            loss_bce = torch.nn.BCELoss()
            maskLossMask = mask_seen | (batch['mask'].squeeze(-1) == 0.0)
            gtmsk = torch.nan_to_num(torch.clamp(batch['mask'][maskLossMask,:].squeeze(),1e-3,1-1e-3))
            renderedmask = torch.nan_to_num(torch.clamp(rendering['acc'][maskLossMask].squeeze(),1e-3,1-1e-3))
            mask_loss = 0.1*loss_bce(renderedmask, gtmsk)
            # mask_loss = -0.1 * ((1 - gtmsk) * torch.log(1 - renderedmask + 1e-5)).mean()
            mask_losses.append(mask_loss)
            stats['masks'].append(mask_loss.item())
        # mask_loss = (lossmult * (rendering['acc'] - batch['masks']) ** 2).sum() / denom
        # data_losses.append(mask_loss)
        if config.compute_disp_metrics:
            # Using mean to compute disparity, but other distance statistics can
            # be used instead.
            disp = 1 / (1 + rendering['distance_mean'])
            stats['disparity_mses'].append(((disp - batch['disps']) ** 2).mean().item())

        if config.compute_normal_metrics:
            if 'normals_pred' in rendering:
                weights = torch.tensor([1.0]).cuda() #batch['mask']
                normalized_normals_gt = ref_utils.l2_normalize(batch['normals'][mask_seen,:3])
                normalized_normals = ref_utils.l2_normalize(rendering['normals_pred'][mask_seen,:3])
                normal_mae = (1e-2)*ref_utils.compute_weighted_mae(weights, normalized_normals,
                                                            normalized_normals_gt)*torch.pi/180.0
                # import pdb; pdb.set_trace()
                normal_losses.append( (1e-2)*(weights*(normalized_normals_gt-normalized_normals)**2).mean())
                # normal_losses.append( normal_mae )
            else:
                # If normals are not computed, set MAE to NaN.
                normal_mae = torch.nan
            stats['normal_maes'].append(normal_mae.item())
        if config.compute_depth_tvl_reg:
            # Compute the TVL1 regularization term on the depth map.
            depth = rendering['depth']
            mask_sanity = torch.nan_to_num(torch.clamp(batch['mask'].squeeze(-1),1e-3,1-1e-3))*mask_seen
            # if not len(mask_sanity.shape) == len(depth.shape):
            #     mask_sanity = mask_sanity[None,...]
            depth_mask =  (mask_sanity[:, :, 1:] * mask_sanity[:, :, :-1])
            depth_grad = torch.abs((depth[:, :, 1:]-depth[:, :, :-1])*depth_mask).mean()
            depth_mask = (mask_sanity[:, 1:, :] * mask_sanity[:, :-1, :])
            depth_grad += torch.abs((depth[:, 1:, :]-depth[:, :-1, :])*depth_mask).mean()
            depth_losses.append((1e-2)*depth_grad.item())
        

    
    loss = (
            ((config.data_coarse_loss_mult * sum(data_losses[:-1]) + config.data_loss_mult * data_losses[-1]) if config.compute_color_loss else 0.0)
            + ((config.data_coarse_loss_mult * sum(mask_losses[:-1]) + config.data_loss_mult * mask_losses[-1]) if config.compute_mask_loss else 0.0)
            + ((config.data_coarse_loss_mult * sum(depth_losses[:-1]) + config.data_loss_mult * depth_losses[-1]) if config.compute_depth_tvl_reg else 0.0)
            + ((config.data_coarse_loss_mult * sum(normal_losses[:-1]) + config.data_loss_mult * normal_losses[-1]) if config.compute_normal_metrics else 0.0)
            )

    stats = {k: np.nan_to_num(np.array(stats[k])) for k in stats}
    return loss, stats


def interlevel_loss(ray_history, config):
    """Computes the interlevel loss defined in mip-NeRF 360."""
    # Stop the gradient from the interlevel loss onto the NeRF MLP.
    last_ray_results = ray_history[-1]
    c = last_ray_results['sdist'].detach()
    w = last_ray_results['weights'].detach()
    loss_interlevel = 0.
    for ray_results in ray_history[:-1]:
        cp = ray_results['sdist']
        wp = ray_results['weights']
        loss_interlevel += stepfun.lossfun_outer(c, w, cp, wp).mean()
    return config.interlevel_loss_mult * loss_interlevel


def anti_interlevel_loss(ray_history, config):
    """Computes the interlevel loss defined in mip-NeRF 360."""
    last_ray_results = ray_history[-1]
    c = last_ray_results['sdist'].detach()
    w = last_ray_results['weights'].detach()
    w_normalize = w / (c[..., 1:] - c[..., :-1]+1e-5)
    loss_anti_interlevel = 0.
    for i, ray_results in enumerate(ray_history[:-1]):
        cp = ray_results['sdist']
        wp = ray_results['weights']
        c_, w_ = stepfun.blur_stepfun(c, w_normalize, config.pulse_width[i])

        # piecewise linear pdf to piecewise quadratic cdf
        area = 0.5 * (w_[..., 1:] + w_[..., :-1]) * (c_[..., 1:] - c_[..., :-1])

        cdf = torch.cat([torch.zeros_like(area[..., :1]), torch.cumsum(area, dim=-1)], dim=-1)

        # query piecewise quadratic interpolation
        cdf_interp = math.sorted_interp_quad(cp, c_, w_, cdf)
        # difference between adjacent interpolated values
        w_s = torch.diff(cdf_interp, dim=-1)

        loss_anti_interlevel += ((w_s - wp).clamp_min(0) ** 2 / (wp + 1e-5)).mean()
        # import pdb; pdb.set_trace()
    return config.anti_interlevel_loss_mult * loss_anti_interlevel


def distortion_loss(ray_history, config):
    """Computes the distortion loss regularizer defined in mip-NeRF 360."""
    last_ray_results = ray_history[-1]
    c = last_ray_results['sdist']
    w = last_ray_results['weights']
    loss = stepfun.lossfun_distortion(c, w).mean()
    return config.distortion_loss_mult * loss


def orientation_loss(batch, model, ray_history, config):
    """Computes the orientation loss regularizer defined in ref-NeRF."""
    total_loss = 0.
    for i, ray_results in enumerate(ray_history):
        w = ray_results['weights']
        n = ray_results[config.orientation_loss_target]
        if n is None:
            raise ValueError('Normals cannot be None if orientation loss is on.')
        # Negate viewdirs to represent normalized vectors from point to camera.
        v = -1. * batch['viewdirs']
        n_dot_v = (n * v[..., None, :]).sum(dim=-1)
        loss = (w * n_dot_v.clamp_min(0) ** 2).sum(dim=-1).mean()
        if i < model.num_levels - 1:
            total_loss += config.orientation_coarse_loss_mult * loss
        else:
            total_loss += config.orientation_loss_mult * loss
    return total_loss


def hash_decay_loss(ray_history, config):
    total_loss = 0.
    for i, ray_results in enumerate(ray_history):
        total_loss += config.hash_decay_mults * ray_results['loss_hash_decay']
    return total_loss


def opacity_loss(renderings, config):
    total_loss = 0.
    for i, rendering in enumerate(renderings):
        o = rendering['acc']
        total_loss += config.opacity_loss_mult * (-o * torch.log(o + 1e-5)).mean()
    return total_loss


def predicted_normal_loss(model, ray_history, config):
    """Computes the predicted normal supervision loss defined in ref-NeRF."""
    total_loss = 0.
    for i, ray_results in enumerate(ray_history):
        w = ray_results['weights']
        n = ray_results['normals']
        n_pred = ray_results['normals_pred']
        if n is None or n_pred is None:
            raise ValueError(
                'Predicted normals and gradient normals cannot be None if '
                'predicted normal loss is on.')
        loss = torch.mean((w * (1.0 - torch.sum(n * n_pred, dim=-1))).sum(dim=-1))
        if i < model.num_levels - 1:
            total_loss += config.predicted_normal_coarse_loss_mult * loss
        else:
            total_loss += config.predicted_normal_loss_mult * loss
    return total_loss


def clip_gradients(model, accelerator, config):
    """Clips gradients of MLP based on norm and max value."""
    if config.grad_max_norm > 0 and accelerator.sync_gradients:
        accelerator.clip_grad_norm_(model.parameters(), config.grad_max_norm)

    if config.grad_max_val > 0 and accelerator.sync_gradients:
        accelerator.clip_grad_value_(model.parameters(), config.grad_max_val)

    for param in model.parameters():
        if param.grad is not None:
            param.grad.nan_to_num_()


def create_optimizer(config: configs.Config, model):
    """Creates optax optimizer for model training."""
    adam_kwargs = {
        'betas': [config.adam_beta1, config.adam_beta2],
        'eps': config.adam_eps,
    }
    lr_kwargs = {
        'max_steps': config.max_steps,
        'lr_delay_steps': config.lr_delay_steps,
        'lr_delay_mult': config.lr_delay_mult,
    }

    lr_fn_main = lambda step: math.learning_rate_decay(
        step,
        lr_init=config.lr_init,
        lr_final=config.lr_final,
        **lr_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_init, **adam_kwargs)

    return optimizer, lr_fn_main
