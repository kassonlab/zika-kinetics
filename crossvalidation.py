#!python
# Perform cross-validation across pH values
# Code 2018 by Peter Kasson

import json
import sys
import numpy
import gflags
from restricted_optimize import RestrictedOptimizer

def cross_validate(optimizer, startvals):
  """Perform cross-validation across pH datasets.
  args:
    optimizer:  optimizer object
    startvals: startinv galues
  rets:
    param_list:  list of best-fit parameters
    gof_list:  list of NLL goodness-of-fit values
    pred_list:  list of NLL on predicted data
  """
  param_list = []
  gof_list = []
  pred_list = []
  for pH in optimizer.pH:
    # copy class
    opt_mask = optimizer.copy()
    # mask out selected pH
    opt_mask.dat = []
    opt_mask.pH = []
    pred = None
    for (dat_it, pH_it) in zip(optimizer.dat, optimizer.pH):
      if pH_it == pH:
        pred = dat_it
      else:
        opt_mask.dat.append(dat_it)
        opt_mask.pH.append(pH_it)
    (cur_param, cur_gof) = opt_mask.optimize(startvals)
    param_list.append(cur_param)
    gof_list.append(cur_gof)
    # now assess prediction
    ratemat = opt_mask.ratemat.copy()
    ratemat[opt_mask.unpinned_idx] = cur_param
    pred_list.append(opt_mask.model.calc_nll(opt_mask.make_pHdep(ratemat, pH),
                                             opt_mask.make_pHdep(ratemat,
                                                                 pred.eq_pH),
                                             pred, opt.eq_corr))
  return (param_list, gof_list, pred_list)


if __name__ == '__main__':
  FLAGS = gflags.FLAGS
  gflags.DEFINE_string('expdata', 'expdat.json', 'Experimental data')
  gflags.DEFINE_string('outfile', 'res.json', 'Output parameters')
  gflags.DEFINE_integer('nstates', 3, 'Number of states')
  gflags.DEFINE_integer('length', 300, 'Time in seconds')
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
  opt.load_data(FLAGS.expdata)
  if not FLAGS.eq:
    opt.eq_corr = False
  if FLAGS.startvals:
    start_vals = numpy.array(FLAGS.startvals.split(','), dtype=float)
  else:
    start_vals = []
  (params, gofs, preds) = cross_validate(opt, start_vals)
  outf = open(FLAGS.outfile, 'w')
  json.dump({'params': [list(p) for p in params], 'gof': gofs, 'pred' : preds},
            outf)
  outf.close()
