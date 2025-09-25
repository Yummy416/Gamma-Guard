import torch
import torch.nn.functional as F

def low_pass_filter(x, cutoff_freq=0.1):
    """
    对输入的 Tensor 进行低通滤波。
    :param x: 输入的 Tensor（形状为 [batch_size, sequence_length, hidden_dim]）。
    :param cutoff_freq: 截止频率。
    :return: 滤波后的 Tensor。
    """
    freq_domain = torch.fft.fft(x, dim=-1)
    n_freq = freq_domain.size(-1)
    mask = torch.ones_like(freq_domain)
    mask[..., int(cutoff_freq * n_freq):] = 0
    filtered_freq_domain = freq_domain * mask
    filtered_x = torch.fft.ifft(filtered_freq_domain, dim=-1).real
    return filtered_x

def gaussian_filter(x, sigma=1.0, alpha=0.5):
    """
    对输入的嵌入表示进行高斯滤波。
    :param x: 输入的嵌入表示（PyTorch Tensor，形状为 [batch_size, sequence_length, embedding_dim]）。
    :param sigma: 高斯核的标准差。
    :return: 滤波后的嵌入表示（形状为 [batch_size, sequence_length, embedding_dim]）。
    """
    ori_type = x.dtype
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    kernel_size = int(2 * sigma * 2) + 1
    kernel = torch.exp(-torch.arange(-kernel_size // 2, kernel_size // 2 + 1).float() ** 2 / (2 * sigma ** 2)).to(x.device)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)
    kernel = kernel.repeat(x.size(2), 1, 1)
    x = x.transpose(1, 2)
    filtered_x = F.conv1d(x, kernel, padding='same', groups=x.size(1))

    filtered_x = filtered_x.transpose(1, 2)
    x = x.transpose(1, 2)
    diff = torch.sigmoid(filtered_x - x)
    weight = alpha * diff
    adjusted_x = x * (1 - weight) + filtered_x * weight
    adjusted_x = adjusted_x.to(ori_type)
    return filtered_x

if __name__ == '__main__':
    pass