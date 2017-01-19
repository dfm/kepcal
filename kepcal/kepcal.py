# -*- coding: utf-8 -*-

from __future__ import division, print_function

import tqdm
import numpy as np

import theano
import theano.tensor as T
import theano.sandbox.linalg as sT

from .adam import adam

__all__ = []


class KepCal(object):
    """
    Y should have the shape ``nstars, ntimes``

    """

    def __init__(self, times, seasons, Y, kernel=None):
        self.Y = np.array(np.atleast_2d(Y))
        self.nstars, self.ntimes = self.Y.shape
        if len(times) != self.ntimes or len(seasons) != self.ntimes:
            raise ValueError("dimension mismatch")
        self.times = np.atleast_1d(times)
        self.seasons = np.atleast_1d(seasons).astype(int)
        self.nseasons = len(set(seasons))
        self.kernel = kernel

        # Normalize
        self.Y /= np.median(Y, axis=1)[:, None]

    def initialize(self):
        F = np.ones(self.nstars)
        C = np.median(self.Y, axis=0)
        Z = np.empty((self.nstars, self.nseasons))
        for s in range(self.nseasons):
            Z[:, s] = np.median((self.Y / C)[:, self.seasons == s], axis=1)
        F_pred = self.Y / (Z[:, self.seasons] * C)
        lnV = np.log(np.median((F_pred - 1.0)**2, axis=1))
        lnS = -4 + np.zeros_like(C)

        self.Z = theano.shared(Z, name="Z1")
        self.C = theano.shared(C, name="C")
        self.F = theano.shared(F, name="F")
        self.lnV = theano.shared(lnV, name="lnV")
        self.lnS = theano.shared(lnS, name="lnS")
        self.jitter = theano.shared(-5.0, name="jitter")

    def get_training_function(self, **adam_args):
        t_in = T.dvector("t")
        Y_in = T.dmatrix("Y")

        z = self.Z[:, self.seasons]
        Y_var = (z * self.C[None, :])**2 * T.exp(self.lnV)[:, None]
        Y_var += z**2 * T.exp(self.lnS)[None, :]
        Y_var += T.exp(2*self.jitter)

        # Compute the model and the likelihood of the data
        model = z * self.C[None, :]
        resid = Y_in - model
        cost = T.sum(resid[1:]**2 / Y_var[1:] + T.log(Y_var[1:]))

        # Apply the GP
        if self.kernel is not None:
            white_noise = z[0]**2 * T.exp(self.lnS)
            white_noise += T.exp(2*self.jitter)
            K = self.kernel.get_value(t_in[:, None] - t_in[None, :])
            # K += white_noise * T.identity_like(K)
            K += T.nlinalg.alloc_diag(white_noise)
            alpha = T.dot(sT.matrix_inverse(K), resid[0])
            cost += T.dot(resid[0], alpha) + T.log(sT.det(K))
        else:
            cost += T.sum(resid[0]**2 / Y_var[0] + T.log(Y_var[0]))

        # Hyperpriors
        cost += T.sum((self.C - 1)**2 / 1e-4)
        cost += T.sum((self.Z - 1)**2 / 1e-4)

        # The parameters
        params = [self.C, self.Z, self.lnV, self.lnS, self.jitter]
        inputs = [Y_in]
        input_values = [self.Y]
        if self.kernel is not None:
            params.append(self.kernel.parameter_vector)
            inputs.append(t_in)
            input_values.append(self.times)

        updates = adam(cost, params, **adam_args)
        return theano.function(inputs, cost, updates=updates), input_values

    def train(self, **kwargs):
        niter = kwargs.pop("niter", 5000)
        train_fn, args = self.get_training_function(**kwargs)
        for i in tqdm.tqdm(range(niter), total=niter):
            cost = train_fn(*args)
        return cost

    def calibrate(self):
        Y_in = T.dmatrix("Y")
        z = self.Z[:, self.seasons]
        trend = z * self.C[None, :]
        cal_flux = Y_in / trend
        cal_ferr = T.sqrt(T.exp(2*self.jitter)+z**2*T.exp(self.lnS)[None, :])
        cal_ferr /= trend
        return theano.function([Y_in], (cal_flux, cal_ferr))(self.Y)
