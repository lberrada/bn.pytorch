import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable


class BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):

        super(BatchNorm2d, self).__init__(num_features, eps,
                                          momentum, affine)

    def forward(self, x):
        """
        Re-implementation of BN: should get identical forward and
        backward passes as torch implementation
        """
        self._check_input_dim(x)
        if self.training:
            xx = x.transpose(1, 3).contiguous().view(-1, x.size(1))
            mean = xx.mean(0)
            # computing inverse of Bessel correction
            bessel_inv = np.sqrt((len(xx) - 1.) / len(xx))
            std = xx.std(0)

            # pre-compute before expanding
            alpha = self.weight / (std * bessel_inv + self.eps)
            beta = self.bias - mean * alpha

            # expand to correct size
            alpha = alpha.view(1, -1, 1, 1).expand_as(x)
            beta = beta.view(1, -1, 1, 1).expand_as(x)

            # compute output
            output = alpha * x + beta

            # running mean and runnning average (including Bessel correction)
            self.running_mean *= (1. - self.momentum)
            self.running_mean += self.momentum * mean.data
            self.running_var *= (1. - self.momentum)
            self.running_var += self.momentum * std.data ** 2

        else:
            # get function
            f = torch._C._functions.BatchNorm(self.running_mean,
                                              self.running_var,
                                              self.training,
                                              self.momentum,
                                              self.eps,
                                              torch.backends.cudnn.enabled)
            # apply function to input
            output = f(x, self.weight, self.bias)
        return output
