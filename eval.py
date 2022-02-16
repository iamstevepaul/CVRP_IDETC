import math
import torch
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from utils import load_model, move_to
from utils.data_utils import save_dataset
from torch.utils.data import DataLoader
import time
from datetime import timedelta
import pickle
import csv
from utils.functions import parse_softmax_temperature
mp = torch.multiprocessing.get_context('spawn')


def get_best(sequences, cost, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]


def eval_dataset_mp(args):
    (dataset_path, width, softmax_temp, opts, i, num_processes) = args

    model, _ = load_model(opts.model)
    val_size = opts.val_size // num_processes
    dataset = model.problem.make_dataset(filename=dataset_path, num_samples=val_size, offset=opts.offset + val_size * i)
    device = torch.device("cuda:{}".format(i))

    return _eval_dataset(model, dataset, width, softmax_temp, opts, device)


def eval_dataset(dataset_path, width, softmax_temp, opts, enc):
    # Even with multiprocessing, we load the model here since it contains the name where to write results
    model, _ = load_model(opts.model, enc)
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    if opts.multiprocessing:
        assert use_cuda, "Can only do multiprocessing with cuda"
        num_processes = torch.cuda.device_count()
        assert opts.val_size % num_processes == 0

        with mp.Pool(num_processes) as pool:
            results = list(itertools.chain.from_iterable(pool.map(
                eval_dataset_mp,
                [(dataset_path, width, softmax_temp, opts, i, num_processes) for i in range(num_processes)]
            )))

    else:
        device = torch.device("cuda:0" if use_cuda else "cpu")
        dataset = model.problem.make_dataset(filename=dataset_path, num_samples=opts.val_size, offset=opts.offset)
        results, times = _eval_dataset(model, dataset, width, softmax_temp, opts, device)

    # This is parallelism, even if we use multiprocessing (we report as if we did not use multiprocessing, e.g. 1 GPU)
    parallelism = opts.eval_batch_size
    costs = []
    for dl in results:
        costs.append(dl[0])

    costs, tours, durations = zip(*results)
    mean_cst = np.array(costs).mean()
    # Not really costs since they should be negative

    # data = {}
    # data["costs"] = costs
    # data["computation_time"] = times

    # pickle.dump(data, open("vrp_20_CAM_test_results.pkl", "wb"))



    # print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
    # print("Average serial duration: {} +- {}".format(
    #     np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
    # print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
    # print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))
    #
    # dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
    # model_name = "_".join(os.path.normpath(os.path.splitext(opts.model)[0]).split(os.sep)[-2:])
    # if opts.o is None:
    #     results_dir = os.path.join(opts.results_dir, model.problem.NAME, dataset_basename)
    #     os.makedirs(results_dir, exist_ok=True)
    #
    #     out_file = os.path.join(results_dir, "{}-{}-{}{}-t{}-{}-{}{}".format(
    #         dataset_basename, model_name,
    #         opts.decode_strategy,
    #         width if opts.decode_strategy != 'greedy' else '',
    #         softmax_temp, opts.offset, opts.offset + len(costs), ext
    #     ))
    # else:
    #     out_file = opts.o
    #
    # assert opts.f or not os.path.isfile(
    #     out_file), "File already exists! Try running with -f option to overwrite."
    #
    # save_dataset((results, parallelism), out_file)

    return costs, tours, durations


def _eval_dataset(model, dataset, width, softmax_temp, opts, device):

    model.to(device)
    model.train()

    model.set_decode_type(
        "greedy" if opts.decode_strategy in ('bs', 'greedy') else "sampling",
        temp=softmax_temp)

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)
    computation_times = []
    results = []
    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        batch = move_to(batch, device)

        start = time.time()
        with torch.no_grad():
            if opts.decode_strategy in ('sample', 'greedy'):
                if opts.decode_strategy == 'greedy':
                    assert width == 0, "Do not set width when using greedy"
                    assert opts.eval_batch_size <= opts.max_calc_batch_size, \
                        "eval_batch_size should be smaller than calc batch size"
                    batch_rep = 1
                    iter_rep = 1
                elif width * opts.eval_batch_size > opts.max_calc_batch_size:
                    assert opts.eval_batch_size == 1
                    assert width % opts.max_calc_batch_size == 0
                    batch_rep = opts.max_calc_batch_size
                    iter_rep = width // opts.max_calc_batch_size
                else:
                    batch_rep = width
                    iter_rep = 1
                assert batch_rep > 0
                # This returns (batch_size, iter_rep shape)
                sequences, costs = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)
                # computation_times.append(computation_time_batch)
                batch_size = len(costs)
                ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)
            else:
                assert opts.decode_strategy == 'bs'

                cum_log_p, sequences, costs, ids, batch_size = model.beam_search(
                    batch, beam_size=width,
                    compress_mask=opts.compress_mask,
                    max_calc_batch_size=opts.max_calc_batch_size
                )

        if sequences is None:
            sequences = [None] * batch_size
            costs = [math.inf] * batch_size
        else:
            sequences, costs = get_best(
                sequences.cpu().numpy(), costs.cpu().numpy(),
                ids.cpu().numpy() if ids is not None else None,
                batch_size
            )
        duration = time.time() - start
        for seq, cost in zip(sequences, costs):
            if model.problem.NAME == "tsp":
                seq = seq.tolist()  # No need to trim as all are same length
            elif model.problem.NAME in ("cvrp", "sdvrp"):
                seq = np.trim_zeros(seq).tolist() + [0]  # Add depot
            elif model.problem.NAME in ("op", "pctsp"):
                seq = np.trim_zeros(seq)  # We have the convention to exclude the depot
            else:
                assert False, "Unkown problem: {}".format(model.problem.NAME)
            # Note VRP only
            results.append((cost, seq, duration))

    return results, computation_times


if __name__ == "__main__":

    # import scipy.io
    # import pickle
    # mat = scipy.io.loadmat('data/t_978.mat')
    # data = list(zip(
    #     (mat["depot"]/1000).tolist(),
    #     (mat["tasks"][None,:, 0:2]/1000).tolist(),
    #     (mat["tasks"][None,:, 2]/1.0).tolist(),
    #     (mat["capacity"][0]/1.0).tolist()
    # ))
    # pickle.dump(data, open('data/X-n979-k43.pkl','wb'))


    # n_tasks = [200]
    n_tasks = [100]
    Encoders = [3]
    dim = [128]
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--val_size', type=int, default=100,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=1,
                        help="Batch size to use during (baseline) evaluation")
    # parser.add_argument('--decode_type', type=str, default='greedy',
    #                     help='Decode type, greedy or sampling')
    parser.add_argument('--width', type=int, nargs='+',
                        help='Sizes of beam to use for beam search (or number of samples for sampling), '
                             '0 to disable (default), -1 for infinite')
    parser.add_argument('--decode_strategy', default="greedy", type=str,
                        help='Beam search (bs), Sampling (sample) or Greedy (greedy)')
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1,
                        help="Softmax temperature (sampling or bs)")
    parser.add_argument('--model', default='outputs/ccn_2L_200', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--max_calc_batch_size', type=int, default=10000, help='Size for subbatches')
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--multiprocessing', action='store_true',
                        help='Use multiprocessing to parallelize over multiple GPUs')
    opts = parser.parse_args()

    for h in dim:
        for i in n_tasks:
            data_set = "data/vrp_1000.pkl"
            # data_set = "data/X-n1001-k43.pkl"
            opts.datasets = [data_set]

            for j in Encoders:
                encoder_path = 'outputs/ccn_2L_100'#'outputs/h_'+str(h)+'/Enc_'+str(j)+'/'

                opts.model = encoder_path
                assert opts.o is None or (len(opts.datasets) == 1 and len(opts.width) <= 1), \
                    "Cannot specify result filename with more than one dataset or more than one width"
                widths = opts.width if opts.width is not None else [0]
                for width in widths:
                    for dataset_path in opts.datasets:
                        costs, tours, durations = eval_dataset(dataset_path, width, opts.softmax_temperature, opts, j)
                        file_name = 'Results/gaussian_h_'+str(h)+'_'+str(i)+'_Enc_'+str(j)+'.csv'
                        # with open(file_name, 'w') as f:
                        #     write = csv.writer(f)
                        #     write.writerows((np.array(costs).T).reshape((100,1)).tolist())


    dt = 0