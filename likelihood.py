#!/usr/bin/python
# calculate model and likelihood
# Code 2018 by Peter Kasson

import numpy

class exp_data(object):
  """Experimental data."""
  __slots__ = ('counts_by_time', 'num_fused', 'num_not_fused',
               'unique_wait_times', 'measured_state', 'pdf')
  def __init__(self):
    self.counts_by_time = []
    self.num_fused = 0
    self.num_not_fused = 0
    self.unique_wait_times = []
    self.measured_state = 0
    self.pdf = []

def make_expdat(dat, time_bin=1):
  """Helper function to compile exp_data.
    args:  dat:  dict of data objects.
    rets:  expdat:  exp_data object.
  """
  res = exp_data()
  # dat should include EfficiencyBefore, optionally EffMax,
  # TimesToFusion (previously SortedpHtoFList)
  res.num_fused = len(dat['TimesToFusion'])
  res.num_not_fused = max(0, round((1-dat['Efficiency']) * res.num_fused))
  res.measured_state = int(dat['FusionState']) - 1  # make 0-indexed
  binned_times = numpy.round(dat['TimesToFusion'] * time_bin) / time_bin
  res.unique_wait_times, res.counts_by_time = numpy.unique(binned_times,
                                                           return_counts=True)
  # cumprob = numpy.cumsum(res.counts_by_time).astype(numpy.double)
  # now need PDF.  Do we normalize cumprob?
  # or do we just use the counts by time?
  return res

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
    self.dt = 1
    self.normalized = True  # sum(start_vals) = 1

  def propagate(self, rate_constants):
    """Propagate kinetic model.
    args:
      rate_constants: matrix of rate constants
    rets:
      conc_vals: matrix of concentrations at each time
    """
    # row-normalize matrix
    # need to deal with sum > 1
    for i in range(len(rate_constants)):
      rate_constants[i, i] = 0
      if sum(rate_constants[i, :]) > 1:
        rate_constants[i, :] /= sum(rate_constants[i, :])
      rate_constants[i, i] = 1 - sum(rate_constants[i, :])
    conc_vals = numpy.zeros((len(self.start_vals), self.duration+1))
    conc_vals[:, 0] = self.start_vals
    for i in range(self.duration):
      conc_vals[:, i+1] = numpy.matmul(conc_vals[:, i], rate_constants)
    return conc_vals

  def pdf_by_time(self, rate_constants, query_state, time_idx):
    """Calculate PDF for each time value in vector.
    args:
      rate_constants:  matrix of rate constants
      query_state:  which state to calculate PDF for
      time_idx:  index values at which to calculate PDF
    rets:
      pdf_val:  PDF values
      prob_notfused: probability not fused
    """
    model_conc = self.propagate(rate_constants)
    # now get PDF at time_vals for query_state
    # following gets full PDF; need to figure out if can get specific
    model_pdf = numpy.gradient(model_conc[query_state, :], self.dt)
    return (model_pdf[time_idx.astype(int)], 1-model_conc[query_state, -1])

  def calc_nll(self, rate_constants, dat):
    """Calculate negative log likelihood for a single model.
    args:
      rate_constants:  matrix of rate constants
      exp_data: experimental data
    rets:
      nll:  negative log likelihood
    """
    # for the moment, hardcoding dt = 1; otherwise would want to do
    # dat.unique_wait_times/self.dt +1 below
    (pdf_vals, prob_notfused) = self.pdf_by_time(rate_constants,
                                                 dat.measured_state,
                                                 dat.unique_wait_times+1)
    safe_idx = numpy.nonzero(pdf_vals)
    nll = numpy.sum(pdf_vals[safe_idx] * dat.counts_by_time[safe_idx])
    # need to make sure finite
    if prob_notfused > 0:
      nll += numpy.log(prob_notfused) * dat.num_not_fused
    print nll
    return nll
