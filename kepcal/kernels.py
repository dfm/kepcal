# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

import theano
import theano.tensor as T

__all__ = []


class Kernel(object):

    def __init__(self, *args):
        self.parameter_vector = theano.shared(np.array(args, dtype=float),
                                              name="params")

class Matern32Kernel(Kernel):

    def get_value(self, t):
        amp = T.exp(self.parameter_vector[0])
        tau = T.exp(self.parameter_vector[1])
        r = T.sqrt(3.0 * t**2) / tau
        return amp * (1.0 + r) * T.exp(-r)
