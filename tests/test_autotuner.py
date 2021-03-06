from __future__ import absolute_import

from functools import reduce
from operator import mul
try:
    from StringIO import StringIO
except ImportError:
    # Python3 compatibility
    from io import StringIO

import pytest
from conftest import skipif_yask

import numpy as np

from devito import Grid, Function, TimeFunction, Eq, Operator, configuration
from devito.logger import logger, logging, set_log_level
from devito.core.autotuning import options


@skipif_yask
@pytest.mark.parametrize("shape,expected", [
    ((30, 30), 17),
    ((30, 30, 30), 21)
])
def test_at_is_actually_working(shape, expected):
    """
    Check that autotuning is actually running when switched on,
    in both 2D and 3D operators.
    """
    grid = Grid(shape=shape)

    buffer = StringIO()
    temporary_handler = logging.StreamHandler(buffer)
    logger.addHandler(temporary_handler)
    set_log_level('DEBUG')

    infield = Function(name='infield', grid=grid)
    infield.data[:] = np.arange(reduce(mul, shape), dtype=np.int32).reshape(shape)
    outfield = Function(name='outfield', grid=grid)
    stencil = Eq(outfield.indexify(), outfield.indexify() + infield.indexify()*3.0)
    op = Operator(stencil, dle=('blocking', {'blockinner': True, 'blockalways': True}))

    # Expected 3 AT attempts for the given shape
    op(infield=infield, outfield=outfield, autotune=True)
    out = [i for i in buffer.getvalue().split('\n') if 'AutoTuner:' in i]
    assert len(out) == 4

    # Now try the same with aggressive autotuning, which tries 9 more cases
    configuration.core['autotuning'] = 'aggressive'
    op(infield=infield, outfield=outfield, autotune=True)
    out = [i for i in buffer.getvalue().split('\n') if 'AutoTuner:' in i]
    assert len(out) == expected
    configuration.core['autotuning'] = configuration.core._defaults['autotuning']

    logger.removeHandler(temporary_handler)

    temporary_handler.flush()
    temporary_handler.close()
    buffer.flush()
    buffer.close()
    set_log_level('INFO')


@skipif_yask
def test_timesteps_per_at_run():
    """
    Check that each autotuning run (ie with a given block shape) takes
    ``autotuning.options['at_squeezer'] - data.time_order`` timesteps.
    in an operator performing an increment such as
    ``a[t + timeorder, ...] = f(a[t, ...], ...)``.
    """

    buffer = StringIO()
    temporary_handler = logging.StreamHandler(buffer)
    logger.addHandler(temporary_handler)
    set_log_level('DEBUG')

    shape = (30, 30, 30)
    grid = Grid(shape=shape)
    x, y, z = grid.dimensions
    t = grid.stepping_dim

    # Function
    infield = Function(name='infield', grid=grid)
    infield.data[:] = np.arange(reduce(mul, shape), dtype=np.int32).reshape(shape)
    outfield = Function(name='outfield', grid=grid)
    stencil = Eq(outfield.indexify(), outfield.indexify() + infield.indexify()*3.0)
    op = Operator(stencil, dle=('blocking', {'blockalways': True}))
    op(infield=infield, outfield=outfield, autotune=True)
    out = [i for i in buffer.getvalue().split('\n') if 'AutoTuner:' in i]
    assert len(out) == 4
    assert all('in 1 time steps' in i for i in out)
    buffer.truncate(0)

    # TimeFunction with increasing time order
    for to in [1, 2, 4]:
        infield = TimeFunction(name='infield', grid=grid, time_order=to)
        infield.data[:] = np.arange(reduce(mul, infield.shape),
                                    dtype=np.int32).reshape(infield.shape)
        outfield = TimeFunction(name='outfield', grid=grid, time_order=to)
        stencil = Eq(outfield.indexed[t + to, x, y, z],
                     outfield.indexify() + infield.indexify()*3.0)
        op = Operator(stencil, dle=('blocking', {'blockalways': True}))
        op(infield=infield, outfield=outfield, autotune=True)
        out = [i for i in buffer.getvalue().split('\n') if 'AutoTuner:' in i]
        expected = options['at_squeezer'] - to
        assert len(out) == 4
        assert all('in %d time steps' % expected in i for i in out)
        buffer.truncate(0)

    logger.removeHandler(temporary_handler)

    temporary_handler.flush()
    temporary_handler.close()
    buffer.flush()
    buffer.close()
    set_log_level('INFO')
