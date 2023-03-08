import torch
import numpy as np
from modAL.models import ActiveLearner
from scipy import stats

from acquisition_functions import ew, ed, em, bw, bd, bm, vw, vd, vm, uniform


def active_learning_procedure(
    query_strategy,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_init: np.ndarray,
    y_init: np.ndarray,
    estimator,
    training,
    args,
    beta,
    totest
):
    """Active Learning Procedure

    Attributes:
        query_strategy: Choose between Uniform(baseline), max_entropy, bald,
        X_val, y_val: Validation dataset,
        X_test, y_test: Test dataset,
        X_pool, y_pool: Query pool set,
        X_init, y_init: Initial training set data points,
        estimator: Neural Network architecture, e.g. CNN,
        training: If False, run test without MC Dropout (default: True)
        args: arguments input in the command line,
        totest: Whether test the model on the test set
    """

    """
    T: Number of MC dropout iterations (repeat acqusition process T times),
    n_query: Number of points to query from X_pool,
    query_times: Number of query to conduct
    """
    T=args.dropout_iter
    n_query=args.query
    query_times=args.query_times
    pool_size=args.pool_size



    learner = ActiveLearner(
        estimator=estimator,
        X_training=X_init,
        y_training=y_init,
        query_strategy=query_strategy,
    )
    """
    Arguments saved for the discriminator and for resetting the batch size
    """
    learner.args=args
    """
    Weight to combine uncertainty and diversity. Save in the learner to avoid creating a global variable
    """
    learner.beta=beta


    """
    In hyperparameter tuning, the model is tested on the validation set.
    """
    if totest == False:

        for index in range(query_times):
            """
            Record the time if using time decay weighted product query
            """
            learner.time=index+1
            query_idx, query_instance = learner.query(
                X_pool, n_query=n_query, T=T, training=training, pool_size=pool_size
            )
            learner.estimator.batch_size=args.batch_size
            learner.teach(X_pool[query_idx], y_pool[query_idx])
            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx, axis=0)
            
            if (index + 1) % 5 == 0:
                print(f"query {index+1}/{query_times} completed")
        """
        Calculate model accuracy on BCNN approach
        """
        model_accuracy=cal_acc(learner.estimator,X_val, y_val,T)
        print(f"********** Validation Accuracy per experiment: {model_accuracy} **********")
        """
        The accuracy curve does need to be record during hyperparameter tuning
        """
        return 1, model_accuracy

    else:
        """
        Record accuracy curve across query times
        """
        perf_hist = [cal_acc(learner.estimator,X_test, y_test,T)]

        for index in range(query_times):
            learner.time=index+1
            query_idx, query_instance = learner.query(
                X_pool, n_query=n_query, T=T, training=training
            )
            learner.estimator.batch_size=args.batch_size
            learner.teach(X_pool[query_idx], y_pool[query_idx])
            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx, axis=0)
            
            if (index + 1) % 5 == 0:
                print(f"query {index+1}/{query_times} completed")
            
            model_accuracy = cal_acc(learner.estimator,X_test, y_test,T)
            perf_hist.append(model_accuracy)

    
        print(f"********** Test Accuracy per experiment: {model_accuracy} **********")
        return perf_hist, model_accuracy






def select_acq_function(acq_func: tuple = (0,0)) -> list:
    """Choose types of query strategy

    Attributes:
        acq_func: a tuple indicating the combination of uncertainty and diversity
    """
    acq_func_dict = {
        (0,0): [ew],
        (0,1): [ed],
        (0,2): [em],
        (1,0): [bw],
        (1,1): [bd],
        (1,2): [bm],
        (2,0): [vw],
        (2,1): [vd],
        (2,2): [vm],
        (10,10):[uniform],
        (100,100):[ew, ed, em, bw, bd, bm, vw, vd, vm, uniform]
    }
    return acq_func_dict[acq_func]

def cal_acc(estimator, x, y,T: int = 100):
    """Calculate accuracy on BCNN approach

    Attributes:
        estimator: the neural network
        x: inputs
        y: labels
        T: number of forward propagation
    """

    """
    Get samples for prediction
    """
    with torch.no_grad():
        outputs = np.stack(
            [
                torch.softmax(
                    estimator.forward(x, training=True),
                    dim=-1,
                )
                .cpu()
                .numpy()
                for _ in range(T)
            ]
        )

    """
    Get predicted labels in each sample and integrate them (this method maybe problematic, I will check the codes on the original paper)
    """
    preds = np.argmax(outputs, axis=2)
    preds1, _ = stats.mode(preds, axis=0,keepdims=True)


    return((y==preds1).mean())