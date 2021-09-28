# Lint as: python3
"""
Main module to run the algorithms.
"""
import os
import sys
import atexit
import csv
import itertools
import multiprocessing
import socket
import random
import time
#import psutil
import sys 
# absl needs to be upgraded to >= 0.10.0, otherwise joblib might not work
from absl import app
from absl import flags
import numpy as np
import shutil
sys.path.insert(0,os.path.dirname("."))
import configs_getter
import DOS
import payoff
import stock_model_fast
from loguru import logger

import joblib

logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time}</green> | <level>{level} | {message}</level>",
    level="TRACE",
)

# GLOBAL CLASSES

NUM_PROCESSORS = multiprocessing.cpu_count()
NB_JOBS = int(NUM_PROCESSORS) - 1






FLAGS = flags.FLAGS

flags.DEFINE_list("algos", None, "Name of the algos to run.")
flags.DEFINE_bool("print_errors", False, "Set to True to print errors if any.")
flags.DEFINE_integer("nb_jobs", NB_JOBS, "Number of CPUs to use parallelly")
flags.DEFINE_bool("generate_pdf", False, "Whether to generate latex tables")

_CSV_HEADERS = ['algo', 'model', 'payoff',
                'hurst', 'nb_stocks',
                'nb_paths', 'nb_dates', 'nb_epochs', 'hidden_size', 'hidden_size2',
                'step_size', 'gamma', 'eps', 'lr',
                'train_ITM_only', 'use_path',
                'price', 'duration']

_PAYOFFS = {
    "MaxPut": payoff.MaxPut,
    "MaxCall": payoff.MaxCall,
    "GeometricPut": payoff.GeometricPut,
    "BasketCall": payoff.BasketCall,
    "Identity": payoff.Identity,
    "Max": payoff.Max,
    "Mean": payoff.Mean,
}

_STOCK_MODELS = {
    "dice_model": stock_model_fast.Model_dice
}

_ALGOS = {
    "DOS": DOS.DeepOptimalStopping
}





def init_seed():
  random.seed(0)
  np.random.seed(0)


def _run_algos():
    fpath = os.path.join(os.path.dirname(__file__),"../output/metrics_draft",
                        f"{int(time.time()*1000)}.csv")

    tmp_dirpath = f'{fpath}.tmp_results'
    os.makedirs(tmp_dirpath, exist_ok=True)
    atexit.register(shutil.rmtree, tmp_dirpath)
    tmp_files_idx = 0

    delayed_jobs = []

    for config_name, config in configs_getter.get_configs():
        print(f'Config {config_name}', config)
        config.algos = [a for a in config.algos
                        if FLAGS.algos is None or a in FLAGS.algos]
        combinations = list(itertools.product(
            config.algos, config.nb_dates,
            config.nb_paths, config.nb_stocks, config.payoffs,
            config.stock_models, config.strikes, 
            config.nb_epochs, config.hidden_size, config.hidden_size2,
            config.step_size, config.gamma,
            config.eps, config.lr,
            config.train_ITM_only, config.use_path))
        # random.shuffle(combinations)
        for params in combinations:
            logger.debug(f"params are {params}")
            for i in range(config.nb_runs):
                tmp_file_path = os.path.join(tmp_dirpath, str(tmp_files_idx))
                tmp_files_idx += 1
                delayed_jobs.append(joblib.delayed(_run_algo)(
                    tmp_file_path, *params)
                    )

    print(f"Running {len(delayed_jobs)} tasks using "
            f"{FLAGS.nb_jobs}/{NUM_PROCESSORS} CPUs...")
    joblib.Parallel(n_jobs=NB_JOBS)(delayed_jobs)

    print(f'Writing results to {fpath}...')
    with open(fpath, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_CSV_HEADERS)
        writer.writeheader()
        for idx in range(tmp_files_idx):
            tmp_file_path = os.path.join(tmp_dirpath, str(idx))
            try:
                with open(tmp_file_path,  "r") as read_f:
                    csvfile.write(read_f.read())
            except FileNotFoundError:
                pass

    return fpath


def _run_algo(
        metrics_fpath, algo, nb_dates, nb_paths,
        nb_stocks, payoff, stock_model, strike,
        nb_epochs, hidden_size=10, hidden_size2=10,  
        step_size= 1, gamma=0.99, eps = 0.001, lr = 0.001,
        train_ITM_only=True, use_path=False):
        """This functions runs one algo for option pricing. It is called by _run_algos()
        which is called in main(). Below the inputs are listed which have to be
        specified in the config that is passed to main().

        Args:
            metrics_fpath ([type]): [description]
            algo ([type]): [description]
            nb_dates ([type]): [description]
            nb_paths ([type]): [description]
            nb_stocks ([type]): [description]
            payoff ([type]): [description]
            stock_model ([type]): [description]
            strike ([type]): [description]
            nb_epochs ([type]): [description]
            hidden_size (int, optional): [description]. Defaults to 10.
            hidden_size2 (int, optional): [description]. Defaults to 10.
            factors (tuple, optional): [description]. Defaults to (1.,1.,1.).
            step_size (int, optional): [description]. Defaults to 1.
            gamma (float, optional): [description]. Defaults to 0.99.
            eps (float, optional): [description]. Defaults to 0.001.
            lr (float, optional): [description]. Defaults to 0.001.
            train_ITM_only (bool, optional): [description]. Defaults to True.
            use_path (bool, optional): [description]. Defaults to False.
        """
        logger.debug("in algo")
        print(algo, nb_paths, '... ', end="")
        payoff_ = _PAYOFFS[payoff](strike)
        stock_model_ = _STOCK_MODELS[stock_model](values =[1,2,3,4,5,6],prob=[0.1,0.1,0.1,0.4,0.2,0.1],nb_stocks=nb_stocks,
            nb_paths=nb_paths, nb_dates=nb_dates)
        if algo in ['DOS']:
            logger.debug("try pricer")
            pricer = _ALGOS[algo](stock_model_, payoff_, nb_epochs=nb_epochs,
                            hidden_size=hidden_size, use_path=use_path, eps=eps)
            logger.debug(f"pricer introduced")
        else:
            pass

        t_begin = time.time()
        logger.debug(f"start_time is given by {t_begin}")
        try:
            logger.debug("pricer start")
            price = pricer.price()
            duration = time.time() - t_begin
            logger.debug("pricer ends")
        except Exception as e:
            print("Oops!", e.__class__, "occurred.")
            print("pricer not working.")
            print()
        metrics_ = {}
        metrics_['algo'] = algo
        metrics_['model'] = stock_model
        metrics_['payoff'] = payoff
        metrics_['nb_stocks'] = nb_stocks
        metrics_['nb_paths'] = nb_paths
        metrics_['nb_dates'] = nb_dates
        metrics_['strike'] = strike
        metrics_['nb_epochs'] = nb_epochs
        metrics_['hidden_size'] = hidden_size
        metrics_['hidden_size2'] = hidden_size2
        metrics_['step_size'] = step_size
        metrics_['gamma'] = gamma
        metrics_['eps'] = eps
        metrics_['lr'] = lr
        metrics_['train_ITM_only'] = train_ITM_only
        metrics_['use_path'] = use_path
        metrics_['price'] = price
        metrics_['duration'] = duration
        print("price: ", price, "duration: ", duration)
        with open(metrics_fpath, "w") as metrics_f:
            writer = csv.DictWriter(metrics_f, fieldnames=_CSV_HEADERS)
            writer.writerow(metrics_)



def main(argv):
  del argv

  filepath = _run_algos()

  




if __name__ == "__main__":
  app.run(main)
