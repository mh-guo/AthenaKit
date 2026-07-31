'''
Generic per-snapshot reduction driver for AthenaK runs.

The physics lives in an app module (athenakit.app.<name>) that must provide
    reduce_tasks(ctx) -> {letter: Task}
where each Task tells the driver how to list snapshots, name outputs, and
process one snapshot.  The driver handles snapshot selection, skip-if-done,
process parallelism and error isolation.

Examples:
    athenakit-reduce -p <run_dir> --app rad_torus -t cp -n 8
    athenakit-reduce -p <run_dir> --app rad_torus -t c --nlist 500 535
    athenakit-reduce -p <run_dir> --app rad_torus -t cp -b 500 -e 1000 -s 5
'''
import os
import sys
import time
import argparse
import importlib
import traceback
from types import SimpleNamespace


# worker entry shared with forked pool processes (closures aren't picklable,
# but fork-started workers inherit this module-level state)
_WORKER_STATE = {}


def _run_job(job):
    return _WORKER_STATE['run_one'](job)


class Ctx:
    '''Paths and helpers handed to the app module.'''

    def __init__(self, args):
        self.args = args
        self.runpath = os.path.abspath(args.path)
        self.datapath = os.path.join(self.runpath, 'data')
        self.binpath = os.path.join(self.datapath, 'bin')
        self.cbinpath = os.path.join(self.datapath, f'cbin_full_{args.coarsen}')
        self.pklpath = os.path.join(self.datapath, 'pkl')
        self.figpath = os.path.join(self.datapath, 'fig')

    @staticmethod
    def task(name, nums, outfile, run):
        return SimpleNamespace(name=name, nums=nums, outfile=outfile, run=run)

    def prefix(self):
        '''Output file basename: --prefix if given, else auto-detected.'''
        if self.args.prefix:
            return self.args.prefix
        for d in (self.binpath, self.cbinpath):
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if f.endswith(('.bin', '.cbin')) and f.count('.') >= 3:
                        return f.split('.')[0]
        raise RuntimeError(f'cannot detect file prefix under {self.datapath}; '
                           'pass --prefix')

    @staticmethod
    def snapshots(dirpath, prefix, suffix):
        nums = []
        if os.path.isdir(dirpath):
            for f in os.listdir(dirpath):
                if f.startswith(prefix) and f.endswith(suffix):
                    try:
                        nums.append(int(f.split('.')[-2]))
                    except ValueError:
                        pass
        return sorted(set(nums))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='run directory (contains data/)')
    parser.add_argument('--app', type=str, required=True,
                        help='app module name under athenakit.app (e.g. rad_torus)')
    parser.add_argument('-t', '--task', type=str, required=True,
                        help='task letters, see the app module docstring')
    parser.add_argument('-b', '--beg', type=int, default=-1)
    parser.add_argument('-e', '--end', type=int, default=-1)
    parser.add_argument('-s', '--step', type=int, default=1)
    parser.add_argument('--nlist', nargs='+', type=int, default=[])
    parser.add_argument('-n', '--nprocess', type=int, default=1)
    parser.add_argument('-a', '--all', action='store_true',
                        help='redo even if output exists')
    # common knobs apps may use
    parser.add_argument('--prefix', type=str, default=None,
                        help='basename of output files (default: auto-detect)')
    parser.add_argument('--slice', type=str, default='x2', choices=['x1', 'x2'])
    parser.add_argument('--coarsen', type=int, default=4, choices=[2, 4])
    parser.add_argument('--bins', type=int, default=128)
    args = parser.parse_args()

    ctx = Ctx(args)
    os.makedirs(ctx.pklpath, exist_ok=True)

    app = importlib.import_module(f'athenakit.app.{args.app}')
    tasks = app.reduce_tasks(ctx)

    unknown = [k for k in args.task if k not in tasks]
    if unknown:
        sys.exit(f'unknown task letter(s) {unknown}; '
                 f'app {args.app} provides {sorted(tasks)}')

    jobs = []
    for key in args.task:
        t = tasks[key]
        nums = t.nums()
        if args.beg >= 0 and args.end > args.beg:
            nums = [n for n in nums if args.beg <= n < args.end and
                    (n - args.beg) % args.step == 0]
        if args.nlist:
            nums = [n for n in nums if n in set(args.nlist)]
        for n in nums:
            if args.all or not os.path.isfile(t.outfile(n)):
                jobs.append((key, n))
    print(f'{len(jobs)} jobs for {ctx.runpath} (app={args.app}, task={args.task})',
          flush=True)

    def run_one(job):
        key, n = job
        try:
            tasks[key].run(n)
        except Exception:
            print(f'FAILED {tasks[key].name} n={n}', flush=True)
            traceback.print_exc()

    tic = time.time()
    if args.nprocess <= 1:
        for j in jobs:
            run_one(j)
    else:
        import multiprocessing as mp
        _WORKER_STATE['run_one'] = run_one
        with mp.get_context('fork').Pool(args.nprocess) as pool:
            pool.map(_run_job, jobs, chunksize=1)
    print(f'Done, {time.time() - tic:.1f}s', flush=True)


if __name__ == '__main__':
    main()
