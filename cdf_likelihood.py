#!/usr/bin/python
# Code 2019 by Peter Kasson

import numpy
import scipy.stats

from likelihood import exp_data, Model

def ecdf(event_list, t_max, dt=1):
  """Computes empirical CDF
  Args:
    event_list: list of times
    t_max: maximum time
    dt: timestep -- ignored for the moment
  Rets:
    cdfvals: list of CDF values for 1...t_max.
  """
  (histvals, _) = numpy.histogram(event_list,
                                  numpy.linspace(0, t_max, 1 + t_max),
                                  density=True)
  return numpy.cumsum(histvals)

def make_expcdf(dat, dt=1, FusionState=4, t_max=500):
  """Replacement for make_expdat to make exp_data with CDF."""
  res = exp_data()
  # dat should include EfficiencyBefore, optionally EffMax,
  # TimesToFusion (previously SortedpHtoFList)
  res.num_fused = len(dat['fusiontimes'])
  res.num_not_fused = (round(res.num_fused / dat['efficiency'])
                       if numpy.isfinite(dat['efficiency']) else 1)
  res.measured_state = FusionState - 1  # make 0-indexed
  binned_times = numpy.round(dat['fusiontimes'] * dt) / dt
  res.unique_wait_times, res.counts_by_time = numpy.unique(binned_times,
                                                           return_counts=True)
  # non-normalized CDF
  res.cdf = ecdf(dat['TimesToFusion'], t_max, dt) * dat['EfficiencyCorrected']
  res.eq_time = 0
  res.eq_T = 310
  return res

class CDFModel(Model):
  """Kinetic model, evaluate with CDFs."""

  def model_cross_entropy(self, rate_constants, target_cdf, query_state,
                          eq_rates=None, eq_time=0):
    """Compute cross-entropy for model and target distribution.
    Args:
      rate_constants:
      target_cdf:
      query_state: which state is fusion product
      eq_rates: equilibration rates, skip if None
      eq_time: equilibration time
    Rets:
      cross-entropy.
    """
    if not numpy.any(eq_rates):
      eq_rates = numpy.eye(len(rate_constants))
      eq_time = 0
    try:
      model_conc = self.propagate(rate_constants, eq_rates, eq_time)
      timevals = model_conc[query_state, 1:] - model_conc[query_state, 0]
      rval = (scipy.stats.entropy(timevals)
              + scipy.stats.entropy(timevals, target_cdf))
    except Exception as e:
      print e
      import pdb; pdb.set_trace()
    if not numpy.isfinite(rval):
      import pdb; pdb.set_trace()
    return rval

  def model_mse(self, rate_constants, target_cdf, query_state,
                eq_rates=None, eq_time=0):
    """Compute MSE between model and target CDFs
    Args:
      rate_constants:
      target_cdf:
      query_state: which state is fusion product
      eq_rates: equilibration rates, skip if None
      eq_time: equilibration time
    Rets:
      mse: mean squared error
    """
    if not numpy.any(eq_rates):
      eq_rates = numpy.eye(len(rate_constants))
      eq_time = 0
    try:
      model_conc = self.propagate(rate_constants, eq_rates, eq_time)
      timevals = model_conc[query_state, 1:] - model_conc[query_state, 0]
      mse = numpy.sum([x**2 if numpy.isfinite(x) else 1
                       for x in timevals - target_cdf])
    except Exception as e:
      print e
      import pdb; pdb.set_trace()
    return mse

  def get_quantidx(self, vals, quants=numpy.linspace(0, 1, 51)):
    """Compute idxarr s.t. vals[idxarr[i]] >= quants[i], minimizing idxarr.
    Args:
      vals: values to use
      quants: quantiles to compute
    Rets:
      idxarr: array of indices.
    """
    return numpy.searchsorted(vals, quants)

  def quantile_err(self, rate_constants, target_cdf, query_state,
                   eq_rates=None, eq_time=0):
    """Compute MSE between model and target CDFs
    Args:
      rate_constants:
      target_cdf:
      query_state: which state is fusion product
      eq_rates: equilibration rates, skip if None
      eq_time: equilibration time
    Rets:
      mse: mean squared error
    """
    if not numpy.any(eq_rates):
      eq_rates = numpy.eye(len(rate_constants))
      eq_time = 0
    try:
      model_conc = self.propagate(rate_constants, eq_rates, eq_time)
      timevals = model_conc[query_state, 1:] - model_conc[query_state, 0]
      time_quants = self.get_quantidx(timevals)
      target_quants = self.get_quantidx(target_cdf)
      mse = numpy.sum([x**2  for x in time_quants - target_quants])
    except Exception as e:
      print e
      import pdb; pdb.set_trace()
    return mse
