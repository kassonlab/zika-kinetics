#!/usr/bin/python
# optimize kinetic model
# Code 2018 by Peter Kasson

import json
import numpy
import scipy.optimize
from scipy.io import loadmat

from likelihood import *

class ModelOptimizer(object):
  """Optimizer for kinetic models."""
  def __init__(self, nstates, timelength, pinned=[]):
    self.startvals = numpy.zeros((nstates, 1))
    self.startvals[0] = 1
    self.nstates = nstates
    self.pinned = pinned
    self.model = Model(timelength, self.startvals)
    testmat = numpy.ones((nstates, nstates))
    testmat[pinned] = 0
    self.unpinned_idx = numpy.nonzero(testmat)
    self.optmethod = 'CG'
    self.dat = []

  def load_data(self, dat_filename):
    """Load experimental reference data."""
    self.dat = loadmat(dat_filename)
    # leave this to be subclassed

  def mean_nll(self, rate_constants):
    print 'Define in subclass!'

  def optimize(self, start_constants=None):
    """Optimize parameters given data.
    args: None
    rets:
      optimized_rates
      mean_nll
    """
    if not start_constants:
      start_constants = numpy.ones(len(self.unpinned_idx[0]))
    res = scipy.optimize.minimize(self.mean_nll, start_constants,
                                  method=self.optmethod)
    return (res.x, res.fun)

class pHModelOptimizer(ModelOptimizer):
  def load_data(self, dat_filename):
    """Load and parse reference data.
    args:  dat_filename:  name of mat file.
    """
    # data format is JSON, key is pH, value is CDFData
    dat = json.load(open(dat_filename))
    self.pH = dat.keys()
    self.dat = [make_expdat(x) for x in dat.values()]

  def __init__(self, nstates, timelength, pinned, pH_dep):
    """Constructor.
    args:
      nstates:  number of states
      timelength:  max time of CDF in sec
      pinned:  list of rates (tuples) that are pinned to 0
      pH_dep:  list of rates that are pH-dependent.
    """
    ModelOptimizer.__init__(self, nstates, timelength, pinned)
    self.pH_dep = pH_dep
    self.pH = []

  def make_pHdep(self, rate_constants, pH):
    """Multiply pH-dependent rate constants by pH."""
    pH_rates = rate_constants.copy()
    for idx in self.pH_dep:
      pH_rates[idx] *= pH
    return pH_rates

  def mean_nll(self, rate_constants):
    """Calculate mean NLL across models."""
    # construct rate matrix
    rate_matrix = numpy.zeros((self.nstates, self.nstates))
    rate_matrix[self.unpinned_idx] = rate_constants
    return numpy.sum([self.model.calc_nll(self.make_pHdep(rate_matrix, pH), dat)
                      / (dat.num_fused + dat.num_not_fused)
                      for (pH, dat) in zip(self.pH, self.dat)])
