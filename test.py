import numpy as np
import torch
import torch.nn as nn
import unittest

from torch.autograd import Variable, gradcheck, grad
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

        torch.manual_seed(1234)
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
        grad_output = torch.randn(self.inp.size())

        self.bn_ref(inp_ref).backward(grad_output)
        self.bn_tst(inp_tst).backward(grad_output)

        grad_i_ref = inp_ref.grad.clone()
        grad_i_tst = inp_tst.grad.clone()

        grad_w_ref = self.bn_ref.weight.grad.clone()
        grad_w_tst = self.bn_tst.weight.grad.clone()

        grad_b_ref = self.bn_ref.bias.grad.clone()
        grad_b_tst = self.bn_tst.bias.grad.clone()

        assert_all_close(grad_i_ref, grad_i_tst)
        assert_all_close(grad_w_ref, grad_w_tst)
        assert_all_close(grad_b_ref, grad_b_tst)

        if self.bn_ref.training:
            self.bn_py.forward(self.inp)
            grad_i_py, grad_w_py, grad_b_py = self.bn_py.backward(grad_output)
            assert_all_close(grad_w_ref, grad_w_py)
            assert_all_close(grad_b_ref, grad_b_py)
            assert_all_close(grad_i_ref, grad_i_py)

    def backwardbackward(self):

        g_o = torch.randn(self.inp.size())
        gg_i = torch.randn(self.inp.size())
        gg_w = torch.randn(self.weight.size())

        inp_tst = Variable(self.inp, requires_grad=True)
        self.bn_py.forward(self.inp)
        self.bn_py.backward(g_o)
        gg_o_py, g_i_py, g_w_py = self.bn_py.backwardbackward(gg_i, gg_w)

        out_tst = self.bn_tst(inp_tst)
        w_tst = self.bn_tst.weight
        g_o = Variable(g_o, requires_grad=True)
        g_w_sym, g_i_sym = grad(out_tst, (w_tst, inp_tst,), (g_o,),
                                retain_graph=True, create_graph=True)
        gg_o_tst, g_i_tst, g_w_tst = grad((g_i_sym, g_w_sym),
                                          (g_o, inp_tst, w_tst),
                                          (gg_i, gg_w))

        # assert_all_close(gg_o_tst, gg_o_py)
        assert_all_close(g_i_tst, g_i_py)
        assert_all_close(g_w_tst, g_w_py)

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

    def test_backwardbackward_train(self):
        self.bn_ref.train()

        self.backwardbackward()
