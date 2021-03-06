#!/usr/bin/python
# calculate model and likelihood
# Code 2018 by Peter Kasson

import numpy
from scipy.integrate import odeint
from scipy.linalg import expm

class exp_data(object):
  """Experimental data."""
  __slots__ = ('counts_by_time', 'num_fused', 'num_not_fused',
               'unique_wait_times', 'measured_state', 'pdf', 'eq_time',
               'eq_pH')
  def __init__(self):
    self.counts_by_time = []
    self.num_fused = 0
    self.num_not_fused = 0
    self.unique_wait_times = []
    self.measured_state = 0
    self.pdf = []
    self.eq_time = 300
    self.eq_pH = 7.4

def make_expdat(dat, time_bin=1):
  """Helper function to compile exp_data.
    args:  dat:  dict of data objects.
    rets:  expdat:  exp_data object.
  """
  res = exp_data()
  # dat should include EfficiencyBefore, optionally EffMax,
  # TimesToFusion (previously SortedpHtoFList)
  res.num_fused = len(dat['TimesToFusion'])
  res.num_not_fused = (round(res.num_fused / dat['EfficiencyCorrected'])
                       if numpy.isfinite(dat['EfficiencyCorrected']) else 1)
  res.measured_state = int(dat['FusionState']) - 1  # make 0-indexed
  binned_times = numpy.round(dat['TimesToFusion'] * time_bin) / time_bin
  res.unique_wait_times, res.counts_by_time = numpy.unique(binned_times,
                                                           return_counts=True)
  # cumprob = numpy.cumsum(res.counts_by_time).astype(numpy.double)
  # now need PDF.  Do we normalize cumprob?
  # or do we just use the counts by time?
  res.eq_time = 300  # hard-code
  res.eq_pH = 7.4
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

  def propagate(self, rate_constants, eq_rates, eq_time):
    """Propagate kinetic model.
    args:
      rate_constants: matrix of rate constants
    rets:
      conc_vals: matrix of concentrations at each time
    """
    for i in range(len(self.start_vals)):
      rate_constants[i, i] = 0
      eq_rates[i, i] = 0
      rate_constants[i, i] = -1 * sum(rate_constants[i, :])
      eq_rates[i, i] = -1 * sum(eq_rates[i, :])
    if False:
      return odeint(lambda v, _t: numpy.matmul(v, rate_constants),
                    self.start_vals, range(self.duration+1)).transpose()
    # alternate:
    update = expm(rate_constants)
    conc_vals = numpy.zeros((len(self.start_vals), self.duration+1))
    conc_vals[:, 0] = numpy.matmul(self.start_vals, expm(eq_rates * eq_time))
    for i in range(self.duration):
      conc_vals[:, i+1] = numpy.matmul(conc_vals[:, i], update)
    return conc_vals

  def pdf_by_time(self, rate_constants, eq_rates, eq_time,
                  query_state, time_idx, eq_corr=True):
    """Calculate PDF for each time value in vector.
    args:
      rate_constants:  matrix of rate constants
      eq_constants:  rate constants for equilibration
      eq_time:  length of equilibration
      query_state:  which state to calculate PDF for
      time_idx:  index values at which to calculate PDF
      eq_corr:  subtract out fused during equilibration
    rets:
      pdf_val:  PDF values
      prob_notfused: probability not fused
    """
    model_conc = self.propagate(rate_constants, eq_rates, eq_time)
    # now get PDF at time_vals for query_state
    # following gets full PDF; need to figure out if can get specific
    model_pdf = numpy.gradient(model_conc[query_state, :], self.dt)
    # threshold small PDF values to zero
    model_pdf[numpy.abs(model_pdf) < 1e-10] = 0
    # subtract out anything that fused during equilibration
    sub_val = model_conc[query_state,0] if eq_corr else 0
    return (model_pdf[time_idx.astype(int)],
            1 - (model_conc[query_state, -1] - sub_val)
            / (numpy.sum(model_conc[:, -1]) - sub_val))

  def calc_nll(self, rate_constants, eq_rates, dat, eq_corr=True):
    """Calculate negative log likelihood for a single model.
    args:
      rate_constants:  matrix of rate constants
      eq_rates:  rate matrix for equilibration
      exp_data: experimental data
      eq_corr:  subtract out fused during equilibration
    rets:
      nll:  negative log likelihood
    """
    # for the moment, hardcoding dt = 1; otherwise would want to do
    # dat.unique_wait_times/self.dt +1 below
    (pdf_vals, prob_notfused) = self.pdf_by_time(rate_constants,
                                                 eq_rates, dat.eq_time,
                                                 dat.measured_state,
                                                 dat.unique_wait_times+1,
                                                 eq_corr)
    logvals = numpy.log(pdf_vals)
    nll = -1 * numpy.sum(numpy.array([x if numpy.isfinite(x) else -20
                                 for x in logvals]) * dat.counts_by_time)
    # nll = numpy.sum(numpy.log(pdf_vals[safe_idx])
    #                 * dat.counts_by_time[safe_idx])
    # need to make sure finite
    if prob_notfused > 0 and numpy.isfinite(prob_notfused):
      nll -= numpy.log(prob_notfused) * dat.num_not_fused
    else:
      nll += 20 * dat.num_not_fused
    # print nll
    if not numpy.isfinite(nll):
      import pdb; pdb.set_trace()
    return nll
