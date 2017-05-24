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


def fun(gxhat, xhat, var):
    N, _, D1, D2 = xhat.size()
    gx = 1. / (N * D1 * D2)
    gx *= (gxhat - sum_(gxhat, axis=(0, 2, 3)).view(1, -1, 1, 1).expand_as(xhat) -
           xhat * sum_(xhat * gxhat, axis=(0, 2, 3)).view(1, -1, 1, 1).expand_as(xhat))
    return gx


class BNpy:

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):

        self.weight = torch.randn(num_features)
        self.bias = torch.randn(num_features)

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

    def forward(self, x):

        x = to_tensor(x)

        xx = x.transpose(1, 3).contiguous().view(-1, x.size(1))
        mean = xx.mean(0)
        # computing inverse of Bessel correction
        bessel_inv = np.sqrt((len(xx) - 1.) / len(xx))
        std = xx.std(0)

        # save for backward
        self.x = x.clone()
        self._save_mean = mean.clone()
        self._save_std = std.clone()

        # pre-compute before expanding
        alpha = self.weight / (std * bessel_inv + self.eps)
        beta = self.bias - mean * alpha

        # expand to correct size
        alpha = alpha.view(1, -1, 1, 1).expand_as(x)
        beta = beta.view(1, -1, 1, 1).expand_as(x)

        # compute output
        output = alpha * x + beta

        return output

    def backward(self, g_y):

        g_y = to_tensor(g_y)
        # save for double backward
        self.g_y = g_y.clone()

        # rexpand save_mean
        self._save_mean = self._save_mean.view(1, -1, 1, 1).expand_as(g_y)
        self._save_std = self._save_std.view(1, -1, 1, 1).expand_as(g_y)

        # recompute xhat
        xhat = (self.x - self._save_mean) / self._save_std

        # intermediate partial derivatives
        g_xhat = g_y * self.weight.view(1, -1, 1, 1).expand_as(self.x)

        # final partial derivatives
        g_x = fun(g_xhat, xhat, self._save_std ** 2)
        g_weight = sum_(xhat * g_y, axis=(0, 2, 3))
        g_bias = sum_(g_y, axis=(0, 2, 3))

        return g_x, g_weight, g_bias

    def backwardbackward(self, gg_x, gg_weight, gg_bias):

        # recompute xhat
        xhat = (self.x - self._save_mean) / self._save_std

        gg_xhat = fun(gg_x, xhat, self._save_std ** 2)
        gg_o = gg_xhat * self.weight.view(1, -1, 1, 1).expand_as(self.x)

        g_x = gg_weight.view(1, -1, 1, 1).expand_as(self.x) * self.g_x
        g_weight = sum_(gg_xhat * self.g_y)

        return gg_o, g_x, g_weight
