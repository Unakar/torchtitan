# Lipschitz-transformer
在/home/t2vg-a100-G2-1/a_xietian/dev/SpectralNorm-GPT/dev/repos/lipschitz-transformers里，

论文可以阅读https://arxiv.org/abs/2507.13338，理解其大意， 在/home/t2vg-a100-G2-1/a_xietian/dev/SpectralNorm-GPT/dev/instructions/spectral_control_gpt5.md有一份gpt5生成的简介(可能未必正确)

代码部分有如下可关注的点
1. 在lipschitz-transformers/modula/atom.py里用jax定义了一系列函数(但由于不是pytorch，不可以直接使用)。请完整阅读此py文件，对其内容做到心中有数
def _orthogonalize(M):
    """Orthogonalize a single matrix, always bfloat16. Credit for coefficients to @YouJiacheng and @leloykun."""
def _hard_cap(M):
    """Apply min(1, x) approximately to the singular values of a single matrix. Credit: Franz Cesista."""
def _soft_cap(M, alpha):
    """Apply min(1, x) approximately to the singular values of a single matrix."""
def _pure_svd(M, w_max=1):
    """Apply min(w_max, x) exactly to the singular values of a single matrix."""
def _power_iterate(A, key, num_iters=16):
    """Power iterate to find the principal singular value and vectors of M."""
def _spectral_hammer(M, key, w_max=1):
    """Set the largest singular value of M to w_max."""
def _spectral_weight_decay(M, key, spectral_wd=0.1):
    """Decay the largest singular value of M by 1 - wd."""
def _spectral_normalize(M, key):
    """Normalize the singular values of M to 1."""
def soft_cap_coupling(w_max, wd, max_update_norm):
    """Calculates the strength for soft cap that bounds singular values at w_max.""" 

2. 类似的，在主代码dev/repos/lipschitz-transformers/nanogpt/train_spectral_cap.py的L37-206
和dev/repos/lipschitz-transformers/nanogpt/train_spectral_normalize.py的L127和L218
用pytorch实现了类似的函数，可以参考使用

3. 我希望我们的代码开发主要在/home/t2vg-a100-G2-1/a_xietian/dev/SpectralNorm-GPT/torchtitan/torchtitan下面，利用torchtitan,首先把上面12两点里需要用到的各种函数可以写成一个spectral_utils.py备用