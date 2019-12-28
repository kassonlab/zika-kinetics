# tracking Colab notebook
# code 2019 by Peter Kasson

import json
import glob
import sys

import gflags
import numpy
import scipy.io

import optimize as opt
from cdf_likelihood import *

kBdivh = 2.084e10
neg_kB = -1.987e-3
dS_boost = (2.946 - 1) # multiply by log(kBdivh)/2 div 10
dH_boost = (8839/1.1)

class TempModelOptimizer(opt.ModelOptimizer):
  def load_data(self, dat_filename):
    """Load and parse reference data.
    args:  dat_filename:  path to mat files.
    """
    self.T = []
    self.dat = []
    for datafile in glob.glob('%s/*.mat' % dat_filename):
      dat = scipy.io.loadmat(datafile)
      self.T.append(dat['tval'] + 273)
      self.dat.append(make_expdat(dat, self.timelength))

  def __init__(self, nstates, timelength, pinned, T_dep):
    """Constructor.
    args:
      nstates:  number of states
      timelength:  max time of CDF in sec
      pinned:  list of rates (tuples) that are pinned to 0
      T_dep:  list of rates that are temperature-dependent.
    """
    opt.ModelOptimizer.__init__(self, nstates, timelength, pinned)
    self.model = CDFModel(timelength, self.startvals)
    self.T_dep = T_dep
    self.T = []
    self.timelength = timelength

  def make_Tdep(self, dS_mat, dH_mat, T):
    """Ajust temperature-dependent rate constants by T."""
    T_rates = dS_mat.copy()
    # need two parameters per rate constant
    # unless we really want k = 1/(betah) exp(-dH/RT)exp(dS/R)
    # dH_mat and dS_mat in reduced units of R
    for idx in self.T_dep:
      T_rates[tuple(idx-1)] = (T * kBdivh *
                               numpy.exp(-1 * dH_mat[tuple(idx-1)] / T)
                               * numpy.exp(dS_mat[tuple(idx-1)]))
    return T_rates

  def mean_nll(self, rate_tuple):
    """Calculate mean NLL across models.
    Args:
      rate_tuple is a multidimensional array where [0] is dS and [1] is dH.
    """
    # construct rate matrix
    dS_mat = numpy.zeros((self.nstates, self.nstates))
    rlen = len(self.unpinned_idx[0])
    dS_mat[self.unpinned_idx] = dS_boost +rate_tuple[:rlen]
    dH_mat = numpy.zeros((self.nstates, self.nstates))
    dH_mat[self.unpinned_idx] = dH_boost + dH_boost/10 * rate_tuple[rlen:] # multiply by log(kBdivh)*300/2
    return numpy.sum([self.model.model_cross_entropy(self.make_Tdep(dS_mat, dH_mat, T),
                                                     dat.cdf, dat.measured_state,
                                                     self.make_Tdep(dS_mat, dH_mat, dat.eq_T),
                                                     dat.eq_time)
                      for (T, dat) in zip(self.T, self.dat)])

class RestrictedTOptimizer(TempModelOptimizer):
  def __init__(self, nstates, timelen, pinned_vals, T_dep, pinratio,
               weight=False):
    """Constructor.
    args:
      nstates:  number of states
      timelen:  max time of CDF in sec
      pinned:  tuples of rates that are pinned and values
      T_dep: list of rates that are temperature-dependent.
      pinratio:  tuples of rate constants that scale with another and reference.
      weight:  weight likelihood by number of observations
     """
    self.eq_corr = True
    self.pinned_vals = pinned_vals
    self.eq_T = None
    self.weighted = weight
    self.weights = None
    # ideally should sanity-check pinratio values
    self.pinratio = pinratio
    TempModelOptimizer.__init__(self, nstates, timelen,
                                [v[:2].astype(int) for v in pinned_vals] +
                                [v[:2].astype(int) + 1 for v in pinratio],
                                T_dep)
    self.ratemat = numpy.zeros((nstates, nstates))
    for v in pinned_vals:
      self.ratemat[int(v[0]) - 1, int(v[1]) - 1] = v[2]

  def copy(self):
    """Copy constructor."""
    new_obj = RestrictedTOptimizer(self.nstates, self.model.duration,
                                   [p for p in self.pinned_vals],
                                   [v for v in self.T_dep],
                                   [rv for rv in self.pinratio],
                                   self.weighted)
    new_obj.T = [p for p in self.T]
    new_obj.dat = [d for d in self.dat]
    return new_obj

  def set_measured_state(self, state_idx):
    """Set measured state index for each index in experimental data."""
    for dat in self.dat:
      dat.measured_state = state_idx -1

  def mean_nll(self, rate_tuple):
    """Calculate mean NLL across models.
    Args:
      rate_tuple is a multidimensional array where [0] is dS and [1] is dH.
    """
    dS_mat = numpy.zeros((self.nstates, self.nstates))
    rlen = len(self.unpinned_idx[0])  # because unpinned_idx is a tuple
    dS_mat[self.unpinned_idx] = dS_boost + rate_tuple[:rlen]
    dH_mat = numpy.zeros((self.nstates, self.nstates))
    dH_mat[self.unpinned_idx] = dH_boost + dH_boost/10 * rate_tuple[rlen:]
    for rat in self.pinratio:
      # multiply out everything in pinratio
      rat = rat.astype(int)
      dS_mat[rat[0], rat[1]] = dS_mat[rat[2], rat[3]]
      dH_mat[rat[0], rat[1]] = dH_mat[rat[2], rat[3]]
    nll_vals = [self.model.quantile_err(self.make_Tdep(dS_mat, dH_mat, T),
                                        dat.cdf, dat.measured_state,
                                        self.make_Tdep(dS_mat, dH_mat, dat.eq_T),
                                        dat.eq_time)
                for (T, dat) in zip(self.T, self.dat)]
    if self.weighted:
      return numpy.dot(nll_vals, self.weights)
    return numpy.sum(nll_vals)

  def eq_efficiency(self, rate_tuple, test_all=True, eq_corr=True):
    """Calculate likelihood of equilibration efficiency."""
    dS_mat = numpy.zeros((self.nstates, self.nstates))
    rlen = len(self.unpinned_idx[0])
    dS_mat[self.unpinned_idx] = dS_boost + rate_tuple[:rlen]
    dH_mat = numpy.zeros((self.nstates, self.nstates))
    dH_mat[self.unpinned_idx] = dH_boost + dH_boost/10 * rate_tuple[rlen:]
    for rat in self.pinratio:
      # multiply out everything in pinratio
      rat = rat.astype(int)
      dS_mat[rat[0], rat[1]] = dS_mat[rat[2], rat[3]]
      dH_mat[rat[0], rat[1]] = dH_mat[rat[2], rat[3]]
    eq_matrix = self.make_Tdep(dS_mat, dH_mat,
                               self.dat[0].eq_T if not self.eq_T else self.eq_T)
    nll = 0
    for (T, dat) in zip(self.T, self.dat):
      target_T = self.eq_T if self.eq_T else dat.eq_T
      if test_all or (T == target_T):
        cur_matrix = self.make_Tdep(dS_mat, dH_mat, T)
        vals = self.model.propagate(cur_matrix, eq_matrix, dat.eq_time)
        # subtract out anything that fused during equilibration
        sub_val = vals[dat.measured_state, 0] if eq_corr else 0
        eff = ((vals[dat.measured_state, -1] - sub_val)
               / (numpy.sum(vals[:, -1]) - sub_val))
        not_fus = -1 * numpy.log(1 - eff) * dat.num_not_fused
        fus = -1 * numpy.log(eff) * dat.num_fused
        nll += ((not_fus if numpy.isfinite(not_fus)
                 else 20 * dat.num_not_fused)
                + (fus if numpy.isfinite(fus) else 20 * dat.num_fused))
    return nll

  def optimize(self, start_constants=[], skip_eq_T=True, eq_T=None,
               skip_pre_opt=False, nsteps=10000):
    """Optimize parameters given data.
    args: start_constants, skip_eq_T, eq_T, skip_pre_opt
    rets:
      optimized_rates
      mean_nll
    """
    self.eq_T = eq_T
    if self.weighted:
      # set up weights
      self.weights = numpy.array([d.num_fused + d.num_not_fused
                                  for d in self.dat])
      # normalize
      self.weights /= numpy.sum(self.weights)
    if not len(start_constants):
      start_constants = numpy.ones(2*len(self.unpinned_idx[0]))*0.01
      start_constants[0] = 3000
    else:
      start_constants = numpy.array(start_constants)
    # for dS and dH:
    start_constants = start_constants.reshape((2, -1))
    bounds = [(0, 10*v) for v in numpy.nditer(start_constants)]
    if not skip_pre_opt:
      # optimize to get efficiency right at 310 K to start
      eq_res = scipy.optimize.basinhopping(self.eq_efficiency, start_constants,
                                           disp=True, niter=1000, stepsize=0.5,
                                           minimizer_kwargs={'bounds': bounds})
      posteq = eq_res.x
    else:
      posteq = start_constants
    if skip_eq_T:
      # if this is set, don't optimize further for equilibration T
      # currently haven't migrated this over well
      eq_idx = numpy.where(self.dat[0].eq_T)[0]
      self.T = numpy.delete(self.T, eq_idx)
      del self.dat[eq_idx]
    if False:
      # set up options for this
      res = scipy.optimize.minimize(self.mean_nll, posteq, method='CG',
                                    bounds=bounds,
                                    options={'disp': True, 'gtol': 1e-7})
    else:
      res = scipy.optimize.basinhopping(self.mean_nll, posteq, T=1.0,
                                        interval=nsteps/20,
                                        disp=True, niter=nsteps, stepsize=0.1,
                                        minimizer_kwargs={'bounds': bounds,
                                                          'options': {'maxiter': 1000}})
    print "After equilibration:"
    print posteq
    print "Final:"
    print res
    dS_end = numpy.zeros((self.nstates, self.nstates))
    dS_end = [self.unpinned_idx] = dS_boost + res.x[0]
    dH_end = numpy.zeros((self.nstates, self.nstates))
    dH_end[self.unpinned_idx] = dH_boost + dH_boost/10 * res.x[1]
    for rat in self.pinratio:
      # multiply out everything in pinratio
      rat = rat.astype(int)
      dS_end[rat[0], rat[1]] = dS_end[rat[2], rat[3]]
      dH_end[rat[0], rat[1]] = dH_end[rat[2], rat[3]]
    print self.make_Tdep(dS_end, dH_end, 310)
    import pdb; pdb.set_trace()
    return (res.x, res.fun)

# routines to convert data

def _make_zero_trace(tracelen):
  return numpy.zeros(tracelen)

def _make_trace(jump_time, tracelen):
  return numpy.heaviside(numpy.arange(tracelen)-jump_time, 1)

def _file_to_traces(filename, ftype='mat',
                    keys=['fusiontimes', 'efficiency', 'maxtime', 'tval']):
  if ftype is 'json':
    infile = open(filename, 'r')
    dat = json.load(infile)
    infile.close()
  else:
    dat = scipy.io.loadmat(filename, squeeze_me=True)
  # convert fusion times to traces
  tracelist = [_make_trace(ftime, dat[keys[2]]) for ftime in dat[keys[0]]]
  # add zero traces
  num_zero = int((1/dat[keys[1]]) - 1) * len(tracelist)
  for _x in range(num_zero):
    tracelist.append(_make_zero_trace(dat[keys[2]]))
  tlist = numpy.ones(numpy.array(tracelist).shape) * dat[keys[3]]
  return (numpy.array(tracelist), tlist)

def get_raw_data(data_path=None, fixed_train=False, fraction_test=0.1):
  """Load JSON raw data from data directory "data_path".

  Args:
    data_path: string path to a directory containing JSON files.
    fixed_train: specify training/test data versus random selection.
    fraction_test: fraction data used for test (and validation)

  Returns:
    tuple ((train_data, train_inp), (valid_data, valid_inp),
           (test_data, test_inp), vocabulary)
    where each of the data objects can be passed to the producer.
  """

  if not fixed_train:
    # binary thresholded trace data
    vocabulary = [0.0, 1.0]
    dat_list = []
    inp_list = []
    for datafile in glob.glob('%s/*.mat' % data_path):
      newdat, newinp = _file_to_traces(datafile)
      # dat_list += newdat
      if len(dat_list):
        dat_list = numpy.append(dat_list, newdat, axis=0)
        inp_list = numpy.append(inp_list, newinp, axis=0)
      else:
        dat_list = newdat
        inp_list = newinp
    numpy.random.shuffle(dat_list)
    train_thresh = int(len(dat_list) * (1 - 2*fraction_test))
    valid_thresh = int(len(dat_list) * (1 - fraction_test))

    train_data = dat_list[:train_thresh]
    train_inp = inp_list[:train_thresh]
    valid_data = dat_list[train_thresh:valid_thresh]
    valid_inp = inp_list[train_thresh:valid_thresh]
    test_data = dat_list[valid_thresh:]
    test_inp = inp_list[valid_thresh:]

  else:
    raise ValueError('Fixed training set behavior not yet defined')
  return ((train_data, train_inp), (valid_data, valid_inp),
          (test_data, test_inp), vocabulary)

def make_expdat(dat, t_max, dt=1, FusionState=4):
  """Helper function to compile exp_data.
    args:  dat:  dict of data objects.
    rets:  expdat:  exp_data object.
  """
  res = exp_data()
  # dat should include EfficiencyBefore, optionally EffMax,
  # TimesToFusion (previously SortedpHtoFList)
  if 'timestep' in dat:
    dt = dat['timestep'][0][0]
  res.num_fused = len(dat['fusiontimes'])
  res.num_not_fused = (round(res.num_fused / dat['efficiency'])
                       if numpy.isfinite(dat['efficiency']) else 1)
  res.measured_state = FusionState - 1  # make 0-indexed
  binned_times = numpy.round(dat['fusiontimes'] * dt) / dt
  res.unique_wait_times, res.counts_by_time = numpy.unique(binned_times,
                                                           return_counts=True)
  res.cdf = ecdf(dat['fusiontimes'], t_max, dt) * dat['efficiency'].flatten()
  res.eq_time = 0  # hard-code; was 300
  res.eq_T = 310
  return res

if __name__ == '__main__':
  FLAGS = gflags.FLAGS
  # hard-code for the moment; swap back over to gflags
  gflags.DEFINE_string('pinned',
                       '1-3-0,3-1-0,2-1-0,4-1-0,3-2-0,4-2-0,4-3-0',
                       'Fixed rate constants')
  gflags.DEFINE_string('Tdep', '1-2,2-3,3-4,1-4,2-4',
                       'Temperature-dependent rate constants')
  gflags.DEFINE_string('expdata',
                       '~/Google Drive/Temperature and pH project/parsed',
                       'Path to data')
  gflags.DEFINE_string('pinratio', '', 'Rate constants forced identical')
  gflags.DEFINE_string('startvals', '',
                       'Starting values for nonzero rate constants')
  gflags.DEFINE_string('eff_T', '310', 'Temperature for initial fit')
  gflags.DEFINE_boolean('eq', False, 'Run equilibration')
  gflags.DEFINE_integer('fus_state', 4, 'Fused state')
  gflags.DEFINE_integer('nstates', 4, 'Number of states in model')
  gflags.DEFINE_integer('length', 300, 'Time length of measurements')
  gflags.DEFINE_integer('nsteps', 10000, 'Number of optimization steps')
  gflags.DEFINE_boolean('weighting', False,
                        'Weight likelihood by observations')
  gflags.DEFINE_boolean('dont_fit_eq', False, 'Skip fitting at equil')
  gflags.DEFINE_string('outfile', 'res.json', 'Output parameters')
  argv = FLAGS(sys.argv)

  pin_parse = [numpy.array(x.split('-'), dtype=float)
               for x in FLAGS.pinned.split(',')] if FLAGS.pinned else []
  T_parse = [numpy.array(x.split('-'), dtype=int)
             for x in FLAGS.Tdep.split(',')] if FLAGS.Tdep else []
  # slightly inconsistent to zero-index pinratio here, but that avoids
  # doing so each step in optimization
  pinrat_parse = [numpy.array(x.split('-'), dtype=float) - 1
                  for x in FLAGS.pinratio.split(',')] if FLAGS.pinratio else []
  opt = RestrictedTOptimizer(FLAGS.nstates, FLAGS.length, pin_parse, T_parse,
                             pinrat_parse, FLAGS.weighting)
  if not FLAGS.eq:
    opt.eq_corr = False
  opt.load_data(FLAGS.expdata)
  opt.set_measured_state(FLAGS.fus_state)
  if FLAGS.startvals:
    start_vals = numpy.array(FLAGS.startvals.split(','), dtype=float)
  else:
    start_vals = []
  eff_T = float(FLAGS.eff_T) if FLAGS.eff_T else None
  (optparam, bestval) = opt.optimize(start_vals, False,
                                     eff_T, FLAGS.dont_fit_eq, FLAGS.nsteps)
  dS_opt = opt.ratemat.copy()
  dS_opt = [opt.unpinned_idx] = dS_boost + optparam[0, :len(opt.unpinned_idx[0])]
  dH_opt = opt.ratemat.copy()
  dH_opt[opt.unpinned_idx] = dH_boost + dH_boost/10 * optparam[1, :len(opt.unpinned_idx[0])]
  for r in pinrat_parse:
    # multiply out everything in pinratio
    r = r.astype(int)
    dS_opt[r[0], r[1]] = dS_opt[r[2], r[3]]
    dH_opt[r[0], r[1]] = dH_opt[r[2], r[3]]
  propdata = [opt.model.propagate(opt.make_Tdep(dS_opt, dH_opt, g_T),
                                  opt.make_Tdep(dS_opt, dH_opt, g_T),
                                  g_dat.eq_time).tolist()
              for (g_T, g_dat) in zip(opt.T, opt.dat)]
  outf = open(FLAGS.outfile, 'w')
  json.dump({'params': list(optparam), 'nll': bestval, 'propdata': propdata},
            outf)
  outf.close()
