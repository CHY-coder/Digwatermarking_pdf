import torch
import utils as utils
from torch.nn import functional as F

def add_noise2(batch, args, global_step):
    N, C, H, W = batch.shape

    global_step = torch.tensor(global_step).float()
    ramp_fn = lambda ramp: torch.min(global_step / ramp, torch.tensor(1.0))

    # resizing, translation, scaling, rotation
    img = utils.apply_transformations(batch, args, ramp_fn, H, W)
    
    # perspective
    pers_rate = torch.rand([]) * ramp_fn(args.rnd_perspec_ramp) * args.rnd_perspec
    img = utils.perspective_img(img, W, pers_rate)
    
    # blur
    probs = [0.25, 0.25]  # Probability for gauss and line blur types
    N_blur = 7
    sigrange_gauss = [1.0, 3.0]
    sigrange_line = [0.25, 1.0]
    wmin_line = 3
    kernel = utils.random_blur_kernel(probs, N_blur, sigrange_gauss, sigrange_line, wmin_line)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # 增加一个维度，用于通道数，现在形状是 [1, N_blur, N_blur]
    img = F.conv2d(img, kernel.to(img.device), padding=N_blur // 2)
    img = torch.clamp(img, 0, 1)
    
    # noise
    rnd_noise = torch.rand([]) * ramp_fn(args.rnd_noise_ramp) * args.rnd_noise
    img = utils.noise(img, rnd_noise)
    
    # color manipulation
    img = utils.color_manipulation(img, args, ramp_fn)

    return img
