import numpy as np
import torch
import torch.nn as nn
import unittest

from torch.autograd import Variable, gradcheck
from bn_autograd import BatchNorm2d
from bn_dbw import BNpy


def to_numpy(tensor):
    if isinstance(tensor, Variable):
        tensor = tensor.data
    if torch.is_tensor(tensor):
        tensor = tensor.numpy()
    if not hasattr(tensor, '__len__'):
        tensor = np.array([tensor])
    assert isinstance(tensor, np.ndarray)
    tensor = tensor.squeeze()
    return tensor


def assert_all_close(tensor_1, tensor_2, rtol=1e-4, atol=1e-4):
    tensor_1 = to_numpy(tensor_1)
    tensor_2 = to_numpy(tensor_2)
    np.testing.assert_allclose(tensor_1, tensor_2, rtol=rtol, atol=atol)


def batchnorm_2d_py(x, weight, bias, eps):
    """ perform standard on batch normalization on x
    """

    x = to_numpy(x)
    weight = to_numpy(weight)
    bias = to_numpy(bias)

    # estimation
    mean = np.mean(x, axis=(0, 2, 3))
    inv_std = 1. / (np.std(x, axis=(0, 2, 3)) + eps)

    # reshape
    mean = mean[None, :, None, None]
    inv_std = inv_std[None, :, None, None]
    weight = weight[None, :, None, None]
    bias = bias[None, :, None, None]

    # center and normalize
    out = (x - mean) * inv_std
    # adjust with learnable parameters
    out = out * weight + bias

    return out


class TestBN2d(unittest.TestCase):

    def setUp(self):

        self.inp_size = (5, 6, 7, 8)
        self.inp = torch.randn(self.inp_size)
        self.weight = torch.randn(self.inp_size[1])
        self.bias = torch.randn(self.inp_size[1])
        self.eps = 1e-5

        self.bn_ref = nn.BatchNorm2d(self.inp_size[1])
        self.bn_tst = BatchNorm2d(self.inp_size[1])
        self.bn_py = BNpy(self.inp_size[1])

        self.bn_py.weight.set_(self.weight)
        self.bn_ref.weight.data.set_(self.weight)
        self.bn_tst.weight.data.set_(self.weight)

        self.bn_py.bias.set_(self.bias)
        self.bn_ref.bias.data.set_(self.bias)
        self.bn_tst.bias.data.set_(self.bias)

    def forward(self):

        out_ref = self.bn_ref(Variable(self.inp))
        out_tst = self.bn_tst(Variable(self.inp))

        rm_ref = self.bn_ref.running_mean
        rm_tst = self.bn_tst.running_mean

        rv_ref = self.bn_ref.running_var
        rv_tst = self.bn_tst.running_var

        assert_all_close(out_ref, out_tst)
        assert_all_close(rm_ref, rm_tst)
        assert_all_close(rv_ref, rv_tst)

        if self.bn_ref.training:
            out_py = self.bn_py.forward(self.inp)
            assert_all_close(out_ref, out_py)

    def backward(self):

        inp_ref = Variable(self.inp, requires_grad=True)
        inp_tst = Variable(self.inp, requires_grad=True)

        self.bn_ref(inp_ref).sum().backward()
        self.bn_tst(inp_tst).sum().backward()

        grad_ref = inp_ref.grad.clone()
        grad_tst = inp_tst.grad.clone()

        assert_all_close(grad_ref, grad_tst)

        if self.bn_ref.training:
            self.bn_py.forward(self.inp)
            grad_py, _, _ = self.bn_py.backward(self.inp)
            assert_all_close(grad_ref, grad_py)

    def test_forward_train(self):
        self.bn_ref.train()
        self.bn_tst.train()

        self.forward()

    def test_forward_eval(self):
        self.bn_ref.eval()
        self.bn_tst.eval()

        self.forward()

    def test_backward_train(self):
        self.bn_ref.train()
        self.bn_tst.train()

        self.backward()

    def test_backward_eval(self):
        self.bn_ref.eval()
        self.bn_tst.eval()

        self.backward()
