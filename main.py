import os
import time
import argparse
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skorch import NeuralNetClassifier
from scipy.ndimage.filters import gaussian_filter1d
from scipy.special import logit

from load_data import LoadData
from cnn_model import ConvNN
from active_learning import select_acq_function, active_learning_procedure

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from modAL.models import BayesianOptimizer
from modAL.acquisition import optimizer_EI, max_EI

"""
set random seeds to make results repeatable
"""

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True


def load_CNN_model(args, device):
    """Load new model each time for different acqusition function
    each experiments"""
    model = ConvNN().to(device)
    cnn_classifier = NeuralNetClassifier(
        module=model,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        train_split=None,
        verbose=0,
        device=device
    )
    return cnn_classifier


def save_as_npy(data: np.ndarray, folder: str, name: str):
    """Save result as npy file

    Attributes:
        data: np array to be saved as npy file,
        folder: result folder name,
        name: npy filename
    """
    file_name = os.path.join(folder, name + ".npy")
    np.save(file_name, data)
    print(f"Saved: {file_name}")


def plot_results(data: dict):
    """Plot results histogram using matplotlib"""
    sns.set()
    for key in data.keys():
        # data[key] = gaussian_filter1d(data[key], sigma=0.9) # for smoother graph
        plt.plot(data[key], label=key)
    plt.show()


def print_elapsed_time(start_time: float, exp: int, acq_func: str):
    """Print elapsed time for each experiment of acquiring

    Attributes:
        start_time: Starting time (in time.time()),
        exp: Experiment iteration
        acq_func: Name of acquisition function
    """
    elp = time.time() - start_time
    print(
        f"********** Experiment {exp} ({acq_func}): {int(elp//3600)}:{int(elp%3600//60)}:{int(elp%60)} **********"
    )


def train_active_learning(args, device, datasets: dict) -> dict:
    """Start training process

    Attributes:
        args: Argparse input,
        estimator: Loaded model, e.g. CNN classifier,
        device: Cpu or gpu,
        datasets: Dataset dict that consists of all datasets,
    """
    acq_functions = select_acq_function((args.uncertainty,args.diversity))
    
    results = dict()
    result_para = dict()
    if args.determ:
        state_loop = [True, False]  # dropout VS non-dropout
    else:
        state_loop = [True]  # run dropout only

    """
    Choose the query method
    """
    if args.runmode == 0:
        print("This is weighted product query")
        if args.time_decay == True:
            print("This is time decay version")
        else:
            print("This is constant weight version")
        """
        If using weighted product query and no beta specified, conduct hyperparameter tuning
        """
        if args.beta==100:
            print("No beta specified, conduct hyperparameter tuning")
            for state in state_loop:
                for i, acq_func in enumerate(acq_functions):
                    acq_func_name = str(acq_func).split(" ")[1] + "-MC=" + str(state)

                    if args.time_decay==True:
                        acq_func_name+="time_decay"

                    """
                    To conduct hyperparameter tuning using Gaussian process, get 3 initial observations
                    """
                    beta_test=[0.25, 0.5, 0.75]
                    """
                    Since original beta is in [0, 1], which maybe too narrow. Transform it using logit function
                    """
                    beta_test_trans=logit(beta_test)
                    beta_test_value=[]

                    for b in beta_test:
                        print('tuning beta: get initial '+str(b))

                        val_scores = []
                        for e in range(args.experiments):
                            start_time = time.time()
                            estimator = load_CNN_model(args, device)
                            print(
                                f"********** Experiment Iterations: {e+1}/{args.experiments} **********"
                            )
                            
                            """
                            Pass the args since the disriminator in WAAL uses the same hyperparameters of the CNN. Besides, 
                            the batch size needs to be reset when retraining the CNN
                            """
                            training_hist, val_score = active_learning_procedure(
                                query_strategy=acq_func,
                                X_val=datasets["X_val"],
                                y_val=datasets["y_val"],
                                X_test=datasets["X_test"],
                                y_test=datasets["y_test"],
                                X_pool=datasets["X_pool"],
                                y_pool=datasets["y_pool"],
                                X_init=datasets["X_init"],
                                y_init=datasets["y_init"],
                                estimator=estimator,
                                training=state,
                                args=args,
                                beta=b,
                                totest=False
                            )
                            
                            val_scores.append(val_score)
                            print_elapsed_time(start_time, e + 1, acq_func_name)
                        
                        avg_val = sum(val_scores) / len(val_scores)
                        beta_test_value.append(logit(avg_val))

                    print("start hyperparameter search")

                    beta_test_trans=np.array(beta_test_trans).reshape(-1,1)
                    beta_test_value=np.array(beta_test_value).reshape(-1,1)

                    beta_explore=np.linspace(0.001, 0.999, 999).reshape(-1, 1)
                    beta_explore_trans=logit(beta_explore)

                    """
                    Set the kernel for Gaussian process. The RBF learned in MLPR can meet some problems, so replace it with Matern
                    """
                    kernel = Matern(length_scale=1.0)
                    regressor = GaussianProcessRegressor(kernel=kernel)

                    optimizer = BayesianOptimizer(
                                estimator=regressor,
                                X_training=beta_test_trans, y_training=beta_test_value,
                                query_strategy=max_EI
                                )

                    """
                    Conduct hyperparameter search 5 times
                    """
                    for search_number in range(5):
                        search_idx, search_inst = optimizer.query(beta_explore_trans)
                        """
                        current best beta inferred from the Gaussian process
                        """
                        beta_current=beta_explore[search_idx][0]
                        print('tuning beta: current position '+str(search_number+1)+'\n'+str(beta_current))

                        val_scores = []
                        for e in range(args.experiments):
                            start_time = time.time()
                            estimator = load_CNN_model(args, device)
                            print(
                                f"********** Experiment Iterations: {e+1}/{args.experiments} **********"
                            )
                            
                            training_hist, val_score = active_learning_procedure(
                                query_strategy=acq_func,
                                X_val=datasets["X_val"],
                                y_val=datasets["y_val"],
                                X_test=datasets["X_test"],
                                y_test=datasets["y_test"],
                                X_pool=datasets["X_pool"],
                                y_pool=datasets["y_pool"],
                                X_init=datasets["X_init"],
                                y_init=datasets["y_init"],
                                estimator=estimator,
                                training=state,
                                args=args,
                                beta=beta_current,
                                totest=False
                            )
                            
                            val_scores.append(val_score)
                            print_elapsed_time(start_time, e + 1, acq_func_name)
                        
                        avg_val = sum(val_scores) / len(val_scores)

                        print('current accuracy:' + str(avg_val))

                        beta_current_value = logit(avg_val)
                        optimizer.teach(beta_explore_trans[search_idx].reshape(-1, 1), beta_current_value.reshape(-1,1))

                    """
                    Find final best beta. Test the model on the test set.
                    """
                    beta_max, value_max = optimizer.get_max()
                    idx=np.where(beta_explore_trans==beta_max)[0]
                    beta_final=beta_explore[idx][0]
                    print('final beta:'+str(beta_final))

                    """
                    avg_hist records the accuracy curve on query times
                    """
                    avg_hist=[]
                    test_scores = []
                    for e in range(args.experiments):
                        start_time = time.time()
                        estimator = load_CNN_model(args, device)
                        print(
                            f"********** Experiment Iterations: {e+1}/{args.experiments} **********"
                        )
                        training_hist, test_score = active_learning_procedure(
                            query_strategy=acq_func,
                            X_val=datasets["X_val"],
                            y_val=datasets["y_val"],
                            X_test=datasets["X_test"],
                            y_test=datasets["y_test"],
                            X_pool=datasets["X_pool"],
                            y_pool=datasets["y_pool"],
                            X_init=datasets["X_init"],
                            y_init=datasets["y_init"],
                            estimator=estimator,
                            training=state,
                            args=args,
                            beta=beta_final,
                            totest=True
                        )                      
                        avg_hist.append(training_hist)
                        test_scores.append(test_score)
                        print_elapsed_time(start_time, e + 1, acq_func_name)
                        
                    avg_hist = np.average(np.array(avg_hist), axis=0)
                    avg_test = sum(test_scores) / len(test_scores)

                    """
                    Accuracy on the test set
                    """
                    print(f'final accuracy:{avg_test}')

                    """
                    Save the accuracy curve across query times, best beta and test accuracy
                    """
                    results[acq_func_name] = avg_hist
                    result_para[acq_func_name]=np.array([beta_final, avg_test])
            """
            save results
            """
            save_as_npy(data=results, folder=args.result_dir, name=acq_func_name)
            save_as_npy(data=result_para,folder=args.result_dir,name=acq_func_name+"para")

        else:
            print("beta is specified, directly run the model")
            for state in state_loop:
                for i, acq_func in enumerate(acq_functions):
        
                    if args.time_decay==False:
                        acq_func_name = str(acq_func).split(" ")[1] + "-MC=" + str(state)+"-betagiven"
                    else:
                        acq_func_name = str(acq_func).split(" ")[1] + "-MC=" + str(state) + "-betagiven"+"-decay"
                    
                    print(f"\n---------- Start {acq_func_name} training! ----------")
                    
                    avg_hist=[]
                    test_scores=[]

                    for e in range(args.experiments):
                        start_time = time.time()
                        estimator = load_CNN_model(args, device)
                        print(
                            f"********** Experiment Iterations: {e+1}/{args.experiments} **********"
                        )
                        training_hist, test_score = active_learning_procedure(
                            query_strategy=acq_func,
                            X_val=datasets["X_val"],
                            y_val=datasets["y_val"],
                            X_test=datasets["X_test"],
                            y_test=datasets["y_test"],
                            X_pool=datasets["X_pool"],
                            y_pool=datasets["y_pool"],
                            X_init=datasets["X_init"],
                            y_init=datasets["y_init"],
                            estimator=estimator,
                            training=state,
                            args=args,
                            beta=args.beta,
                            totest=True
                        )
                        
                        avg_hist.append(training_hist)
                        test_scores.append(test_score)
                        print_elapsed_time(start_time, e + 1, acq_func_name)
                    

                    avg_hist = np.average(np.array(avg_hist), axis=0)
                    avg_test = sum(test_scores) / len(test_scores)
                    print(f'final accuracy:{avg_test}')

                    results[acq_func_name] = avg_hist
                    result_para[acq_func_name]=np.array([args.beta, avg_test])

            save_as_npy(data=results, folder=args.result_dir, name=acq_func_name)
            save_as_npy(data=result_para,folder=args.result_dir,name=acq_func_name+"para")

    else:

        print("This is two-stage query")

        for state in state_loop:
            for i, acq_func in enumerate(acq_functions):
                acq_func_name = str(acq_func).split(" ")[1] + "-MC=" + str(state)

                if args.priority == 0:
                    acq_func_name+="uncertainty"
                else:
                    acq_func_name+="diverisity"

                print(f"\n---------- Start {acq_func_name} training! ----------")

                avg_hist=[]
                test_scores=[]
                for e in range(args.experiments):
                    start_time = time.time()
                    estimator = load_CNN_model(args, device)
                    print(
                        f"********** Experiment Iterations: {e+1}/{args.experiments} **********"
                    )
                    """
                    beta is meaningless here, but still pass it to conduct the function
                    """
                    training_hist, test_score = active_learning_procedure(
                        query_strategy=acq_func,
                        X_val=datasets["X_val"],
                        y_val=datasets["y_val"],
                        X_test=datasets["X_test"],
                        y_test=datasets["y_test"],
                        X_pool=datasets["X_pool"],
                        y_pool=datasets["y_pool"],
                        X_init=datasets["X_init"],
                        y_init=datasets["y_init"],
                        estimator=estimator,
                        training=state,
                        args=args,
                        beta=args.beta,
                        totest=True
                    )
                    
                    avg_hist.append(training_hist)
                    test_scores.append(test_score)
                    print_elapsed_time(start_time, e + 1, acq_func_name)
                    

                avg_hist = np.average(np.array(avg_hist), axis=0)
                avg_test = sum(test_scores) / len(test_scores)
                print(f'final accuracy:{avg_test}')

                results[acq_func_name] = avg_hist
                result_para[acq_func_name]=np.array([avg_test])

        save_as_npy(data=results, folder=args.result_dir, name=acq_func_name)
        save_as_npy(data=result_para,folder=args.result_dir,name=acq_func_name+"para")

    print("--------------- Done Training! ---------------")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="EP",
        help="number of epochs to train (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--seed", type=int, default=369, metavar="S", help="random seed (default: 369)"
    )
    parser.add_argument(
        "--experiments",
        type=int,
        default=2,
        metavar="E",
        help="number of experiments (default: 3)",
    )
    parser.add_argument(
        "--dropout_iter",
        type=int,
        default=5,
        metavar="T",
        help="dropout iterations,T (default: 100)",
    )

    """
    The original codes set query times and dropout iteration times same, which should be controlled separately
    """
    parser.add_argument(
        "--query_times",
        type=int,
        default=5,
        metavar="QT",
        help="Times of query (default: 100)",
    )

    parser.add_argument(
        "--query",
        type=int,
        default=5,
        metavar="Q",
        help="number of query (default: 10)",
    )
    """
    Pool size for query
    """
    parser.add_argument(
        "--pool_size",
        type=int,
        default=200,
        metavar="PS",
        help="pool size for query",
    )
    """
    Whether the weight of diversity should decay with time
    """
    parser.add_argument(
        "--time_decay",
        type=bool,
        default=False,
        metavar="TD",
        help="whether to decay weight with time",
    )

    """
    The combination of uncertainty and diverity metrics decides the query strategy
    """
    parser.add_argument(
        "--uncertainty",
        type=int,
        default=0,
        metavar="UN",
        help="uncertainty: 0 = entropy, 1 = bald, 2 = var_ratios, 10 = uniform, 100 = all ",
    )
    parser.add_argument(
        "--diversity",
        type=int,
        default=0,
        metavar="DI",
        help="diverisity: 0 = waal, 1 = density, 2 = minidis, 10 = uniform, 100 = all ",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=100,
        metavar="V",
        help="validation set size (default: 100)",
    )
    parser.add_argument(
        "--determ",
        action="store_true",
        help="Compare with deterministic models (default: False)",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="result_npy",
        metavar="SD",
        help="Save npy file in this folder (default: result_npy)",
    )
    """
    When using weighted product mode, the model will run with specified beta, otherwise run hyperparameter tuning
    """
    parser.add_argument(
        "--beta",
        type=int,
        default=100,
        help="specify beta")
    """
    Choose weighted product or two stage query method
    """
    parser.add_argument(
        "--runmode",
        type=int,
        default=0,
        help="whether to use weighted product or two stage query")

    """
    Choose look which metric first in two stage query 
    """

    parser.add_argument(
        "--priority",
        type=int,
        default=0,
        help="priority of metrics: 0 = uncertainty first, 1 = diversity first")
    """
    Choose the ratio between candidate size and number of query in two stage query 
    """

    parser.add_argument(
        "--candidate_ratio",
        type=int,
        default=2,
        help="The ratio between candidates and number of query in two stage query")

    args = parser.parse_args()
    setup_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datasets = dict()
    DataLoader = LoadData(args.val_size)
    (
        datasets["X_init"],
        datasets["y_init"],
        datasets["X_val"],
        datasets["y_val"],
        datasets["X_pool"],
        datasets["y_pool"],
        datasets["X_test"],
        datasets["y_test"],
    ) = DataLoader.load_all()

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    results = train_active_learning(args, device, datasets)
    plot_results(data=results)


if __name__ == "__main__":
    main()
