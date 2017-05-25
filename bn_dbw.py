import numpy as np
import torch
from torch.autograd import Variable


def to_tensor(x):
    if isinstance(x, Variable):
        x = x.data
    assert torch.is_tensor(x)
    return x


def sum_(tensor, axis):
    """
    Perform sum over multiple axes using numpy
    Return result as a torch tensor
    """
    res = np.sum(tensor.numpy(), axis=axis)
    return torch.from_numpy(res)


def fun(gxhat, xhat, std):
    N, _, D1, D2 = xhat.size()
    K = N * D1 * D2
    gx = 1. / (K * std)
    gx *= (K * gxhat - view_expand_as(sum_(gxhat, axis=(0, 2, 3)), xhat) -
           xhat * view_expand_as(sum_(xhat * gxhat, axis=(0, 2, 3)), xhat))
    return gx


def view_expand_as(tensor, x):
    view_sizes = [1 for _ in range(x.dim())]
    view_sizes[1] = -1
    res = tensor.view(*view_sizes).expand_as(x)
    return res


class BNpy:

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):

        self.weight = torch.randn(num_features)
        self.bias = torch.randn(num_features)

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

    def forward(self, i):

        i = to_tensor(i)

        ii = i.transpose(1, 3).contiguous().view(-1, i.size(1))
        mean = ii.mean(0)
        # computing inverse of Bessel correction
        bessel_inv = np.sqrt((len(ii) - 1.) / len(ii))
        std = ii.std(0) * bessel_inv

        # save for backward (clones might be unnecessary)
        self.i = i.clone()
        self._save_mean = mean.clone()
        self._save_std = std.clone()

        # pre-compute before expanding
        alpha = self.weight / (std + self.eps)
        beta = self.bias - mean * alpha

        # expand to correct size
        alpha = view_expand_as(alpha, i)
        beta = view_expand_as(beta, i)

        # compute output
        o = alpha * i + beta

        return o

    def backward(self, g_o):

        g_o = to_tensor(g_o)

        # recompute xhat
        alpha = 1. / (self._save_std + self.eps)
        beta = - self._save_mean * alpha
        alpha = view_expand_as(alpha, self.i)
        beta = view_expand_as(beta, self.i)
        xhat = alpha * self.i + beta

        # intermediate partial derivatives
        g_xhat = g_o * view_expand_as(self.weight, self.i)

        # final partial derivatives
        g_i = fun(g_xhat, xhat, view_expand_as(self._save_std, self.i))
        g_w = sum_(xhat * g_o, axis=(0, 2, 3))
        g_b = sum_(g_o, axis=(0, 2, 3))

        # save for double backward
        self.g_o = g_o.clone()
        self.g_xhat = g_xhat.clone()

        return g_i, g_w, g_b

    def backwardbackward(self, gg_x, gg_w):

        # recompute xhat
        alpha = 1. / (self._save_std + self.eps)
        beta = - self._save_mean * alpha
        alpha = view_expand_as(alpha, self.i)
        beta = view_expand_as(beta, self.i)
        xhat = alpha * self.i + beta

        # gg_xhat = d(L_tilde) / d(g_xhat)
        gg_xhat = fun(gg_x, xhat, view_expand_as(self._save_std, self.i))
        # gg_o = d(L_tilde) / d(g_o)
        gg_o = gg_xhat * view_expand_as(self.weight, self.i)

        # g_xhat = d(L_tilde) / d(xhat)
        # NB: different from self.g_xhat = d(L) / d(xhat)
        N, _, D1, D2 = self.i.size()
        K = view_expand_as(1. / (N * D1 * D2 * self._save_std), self.i)
        g_xhat = -K * (gg_x * view_expand_as(sum_(xhat * self.g_xhat, axis=(0, 2, 3)), self.i) +
                       xhat * view_expand_as(sum_(gg_x * self.g_xhat, axis=(0, 2, 3)), self.i))

        # g_i = d(L_tilde) / d(i)
        g_i = fun(g_xhat, xhat, view_expand_as(self._save_std, self.i))
        # g_w = d(L_tilde) / d(w)
        g_w = sum_(gg_xhat * self.g_o, axis=(0, 2, 3))

        return gg_o, g_i, g_w
