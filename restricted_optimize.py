#!/usr/bin/python
# Code 2018 by Peter Kasson

import json
import sys
import numpy
import gflags
import scipy.optimize
import optimize

class RestrictedOptimizer(optimize.pHModelOptimizer):
  def __init__(self, nstates, timelen, pinned_vals, pH_dep):
    """Constructor.
    args:
      nstates:  number of states
      timelen:  max time of CDF in sec
      pinned:  tuples of rates that are pinned and values
      pH_dep: list of rates that are pH-dependent.
     """
    self.eq_corr = True
    self.pinned_vals = pinned_vals
    optimize.pHModelOptimizer.__init__(self, nstates, timelen,
                                       [v[:2].astype(int) for v in pinned_vals],
                                       pH_dep)
    self.ratemat = numpy.zeros((nstates, nstates))
    for v in pinned_vals:
      self.ratemat[int(v[0]) - 1, int(v[1]) - 1] = v[2]

  def copy(self):
    """Copy constructor."""
    new_obj = RestrictedOptimizer(self.nstates, self.model.duration,
              [p for p in self.pinned_vals], [v for v in self.pH_dep])
    new_obj.pH = [p for p in self.pH]
    new_obj.dat = [d for d in self.dat]
    return new_obj

  def set_measured_state(self, state_idx):
    """Set measured state index for each index in experimental data."""
    for dat in self.dat:
      dat.measured_state = state_idx -1

  def mean_nll(self, rate_constants):
    """Calculate mean NLL across models."""
    rate_matrix = self.ratemat.copy()
    rate_matrix[self.unpinned_idx] = rate_constants
    return numpy.sum([self.model.calc_nll(self.make_pHdep(rate_matrix, pH),
                                          self.make_pHdep(rate_matrix,
                                                          dat.eq_pH), dat,
                                                          self.eq_corr)
                      / (dat.num_fused + dat.num_not_fused)
                      for (pH, dat) in zip(self.pH, self.dat)])

  def eq_efficiency(self, rate_constants, test_all=True, eq_corr=True):
    """Calculate likelihood of equilibration efficiency."""
    rate_matrix = self.ratemat.copy()
    rate_matrix[self.unpinned_idx] = rate_constants
    eq_matrix = self.make_pHdep(rate_matrix, self.dat[0].eq_pH)
    nll = 0
    for (pH, dat) in zip(self.pH, self.dat):
      if test_all or (pH == dat.eq_pH):
        cur_matrix = self.make_pHdep(rate_matrix, pH)
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

  def optimize(self, start_constants=[]):
    """Optimize parameters given data.
    args: None
    rets:
      optimized_rates
      mean_nll
    """
    if not len(start_constants):
      start_constants = numpy.ones(len(self.unpinned_idx[0]))*0.01
      start_constants[0] = 3000
    bounds = [(0, 100*v) for v in start_constants]
    # optimize to get efficiency right at pH 7.4 to start
    eq_res = scipy.optimize.basinhopping(self.eq_efficiency, start_constants,
                                         disp=True, niter=1000,
                                        minimizer_kwargs={'bounds': bounds})
    if False:
      # set up options for this
      res = scipy.optimize.differential_evolution(self.mean_nll,
                                                  bounds=[(0, 1e4), (0, 1),
                                                          (0, 1)],
                                                  disp=True)
    else:
      res = scipy.optimize.basinhopping(self.mean_nll, eq_res.x,
                                        disp=True, niter=1000,
                                        minimizer_kwargs={'bounds': bounds})
    print "After equilibration:"
    print eq_res
    print "Final:"
    print res
    print self.make_pHdep(self.ratemat.copy(), 7.4)
    return (res.x, res.fun)


if __name__ == '__main__':
  FLAGS = gflags.FLAGS
  gflags.DEFINE_string('expdata', 'expdat.json', 'Experimental data')
  gflags.DEFINE_string('outfile', 'res.json', 'Output parameters')
  gflags.DEFINE_integer('nstates', 3, 'Number of states')
  gflags.DEFINE_integer('length', 300, 'Time in seconds')
  gflags.DEFINE_integer('fus_state', 3, 'State measured experimentally')
  gflags.DEFINE_string('pinned', '', 'Transitions that are invariate. '
                       'Comma-separated list of a-b-val')
  gflags.DEFINE_string('pHdep', '', 'Transitions that are pH-dependent')
  gflags.DEFINE_string('startvals', '', 'Starting parameters')
  gflags.DEFINE_bool('eq', True, 'Correct for equilibration')
  argv = FLAGS(sys.argv)
  pin_parse = [numpy.array(x.split('-'), dtype=float)
               for x in FLAGS.pinned.split(',')] if FLAGS.pinned else []
  pH_parse = [numpy.array(x.split('-'), dtype=int)
              for x in FLAGS.pHdep.split(',')] if FLAGS.pHdep else []
  opt = RestrictedOptimizer(FLAGS.nstates, FLAGS.length, pin_parse, pH_parse)
  if not FLAGS.eq:
    opt.eq_corr = False
  opt.load_data(FLAGS.expdata)
  opt.set_measured_state(FLAGS.fus_state)
  if FLAGS.startvals:
    start_vals = numpy.array(FLAGS.startvals.split(','), dtype=float)
  else:
    start_vals = []
  (optparam, bestval) = opt.optimize(start_vals)
  outf = open(FLAGS.outfile, 'w')
  json.dump({'params': list(optparam), 'nll': bestval}, outf)
  outf.close()
