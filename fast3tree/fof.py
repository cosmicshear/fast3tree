import sys
import numpy as np
from .core import fast3tree

# tqdm automatically switches to the text-based
# progress bar if not running in Jupyter
try: # https://github.com/tqdm/tqdm/issues/506
    ipy_str = str(type(get_ipython()))
    if 'zmqshell' in ipy_str:  # jupyter
        from tqdm.notebook import tqdm
    if 'terminal' in ipy_str:  # ipython
        from tqdm import tqdm 
except:                        # terminal
    if sys.stderr.isatty():
        from tqdm import tqdm
    else:
        def tqdm(iterable, **kwargs):
            return iterable

__all__ = ['find_friends_of_friends']

def find_friends_of_friends(points, linking_length, periodic_box_size=None,
                            reassign_group_indices=True, **tqdm_kwargs):
    if tqdm_kwargs is None:
        tqdm_kwargs={}
    tqdm_kwargs.setdefault('desc', 'FoF matching');
    group_ids = np.repeat(-1, len(points))
    with fast3tree(points) as tree:
        if periodic_box_size:
            tree.set_boundaries(0, periodic_box_size)
        for i, point in enumerate(tqdm(points,**tqdm_kwargs)):
            idx = tree.query_radius(point, linking_length, bool(periodic_box_size))
            group_ids_this = np.unique(group_ids[idx])
            group_ids_this = group_ids_this[group_ids_this != -1]
            if len(group_ids_this) == 1:
                group_ids[idx] = group_ids_this[0]
            else:
                group_ids[idx] = i
                if len(group_ids_this):
                    group_ids[np.in1d(group_ids, group_ids_this)] = i
    if reassign_group_indices:
        group_ids = np.unique(group_ids, return_inverse=True)[1]
    return group_ids
