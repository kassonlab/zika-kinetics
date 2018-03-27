#!/usr/bin/python
# calculate model and likelihood
# Code 2018 by Peter Kasson

import numpy

class exp_data(object):
  """Experimental data."""
  __slots__ = ('counts_by_time', 'num_fused', 'num_not_fused',
               'unique_wait_times', 'measured_state')
  def __init__(self):
    self.counts_by_time = []
    self.num_fused = 0
    self.num_not_fused = 0
    self.unique_wait_times = []
    self.measured_state = 0

def make_expdat(dat):
  """Helper function to compile exp_data.
    args:  dat:  dict of data objects.
    rets:  expdat:  exp_data object.
  """

class Model(object):
  """Kinetic model."""
  def __init__(self, duration, start_vals):
    """Constructor.
    args:
      duration: time in sec to run model
      start_vals: vector of starting vals
    """
    self.duration = duration
    self.start_vals = start_vals

  def propagate(self, rate_constants):
    """Propagate kinetic model.
    args:
      rate_constants: matrix of rate constants
    rets:
      conc_vals: matrix of concentrations at each time
    """

  def calc_pdf_single(self, rate_constants, query_state):
    """Calculate PDF for model.
    args:
      rate_constants:  matrix of rate constants
      query_state:  which state to calculate PDF for
    rets:
      pdf_val:  PDF for query_state
    """

  def log_pdf_by_time(self, rate_constants, query_state, time_vals):
    """Calculate log PDF for each time value in vector.
    args:
      rate_constants:  matrix of rate constants
      query_state:  which state to calculate PDF for
      time_vals:  values at which to calculate PDF
    rets:
      pdf_val:  PDF values
      prob_notfused: probability not fused
    """

  def prob_obs(self, model_pdf, observations):
    """Calculation probability of observations given model.
    args:
      model_pdf: probability distribution function of model
      observations: vector of experimental observation times
    rets:
      log_prob:  log likelihood of observations given model
    """

  def calc_nll(self, rate_constants, dat):
    """Calculate negative log likelihood for a single model.
    args:
      rate_constants:  matrix of rate constants
      exp_data: dict with experimental data
    rets:
      nll:  negative log likelihood
    """
    (logvals, prob_notfused) = self.log_pdf_by_time(rate_constants,
                                                    dat.measured_state,
                                                    dat.unique_wait_times)
    nll = numpy.sum(logvals * dat.counts_by_time)
    # need to make sure finite
    nll += numpy.log(prob_notfused) * dat.num_not_fused
    return nll
