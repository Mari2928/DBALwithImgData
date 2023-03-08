import torch
import numpy as np
from scipy import stats
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier


def predictions_from_pool(
    model, X_pool: np.ndarray, T: int = 100, training: bool = True, pool_size: int = 20
):
    """Run random_subset prediction on model and return the output

    Attributes:
        X_pool: Pool set to select uncertainty,
        T: Number of MC dropout iterations aka training iterations,
        training: If False, run test without MC dropout. (default=True)
        pool_size: Size of the pool 
    """
    random_subset = np.random.choice(range(len(X_pool)), size=pool_size, replace=False)
    with torch.no_grad():
        outputs = np.stack(
            [
                torch.softmax(
                    model.estimator.forward(X_pool[random_subset], training=training),
                    dim=-1,
                )
                .cpu()
                .numpy()
                for _ in range(T)
            ]
        )



    """
    Since we need to get embedding of all points in the pool, the batch size need to be reset (this size is also larger than the number of labeled points)
    """
    model.estimator.batch_size=pool_size

    model.estimator.forward(model.X_training,training=False)
    

    label_eb=model.estimator.module.e.cpu().numpy()

    model.estimator.forward(X_pool[random_subset],training=False)

    unlabel_eb=model.estimator.module.e.cpu().numpy()
    """
    Stack the two embedding matrices, standarize each feature (sorry to make a confusing variable name. The manipulation is standardization 
    rather than normalization) and re-split them
    """
    all_eb=np.vstack((unlabel_eb,label_eb))
    all_norm=(all_eb-all_eb.mean(axis=0))/all_eb.std(axis=0)

    unlabel_norm, label_norm=np.vsplit(all_norm, [pool_size])



    return outputs, random_subset, label_norm, unlabel_norm



def combine_metric(model, X_pool: np.ndarray, uncertainty,diversity, n_query: int = 10,T: int = 100, training: bool = True, pool_size: int = 200):
    outputs, random_subset, label_norm, unlabel_norm = predictions_from_pool(model, X_pool, T, training, pool_size)

    """
    Calculate the uncertainty and diversity scores repectively and combine them using a weighted product
    """
    uncertainty_values=uncertainty(outputs).reshape((-1,))

    diversity_values=diversity(model, label_norm, unlabel_norm).reshape((-1,))
    """
    set the weight of diversity to a time-decay mode if time_decay is True
    """
    if model.args.time_decay==False:
        final_score=np.power(uncertainty_values, model.beta)*np.power(diversity_values, 1-model.beta)
    else:
        beta_time = 1-np.exp(-model.beta*model.time)
        final_score=np.power(uncertainty_values, beta_time)*np.power(diversity_values, 1-beta_time)

    idx = (-final_score).argsort()[:n_query]
    query_idx = random_subset[idx]
    return query_idx, X_pool[query_idx]

"""
Each possible hybrid query strategy. The name is the abbervation of methods considered. For example, ew = entropy + waal
"""

def two_stage_query(model, X_pool: np.ndarray, uncertainty,diversity, n_query: int = 10,T: int = 100, training: bool = True, pool_size: int = 200):
    outputs, random_subset, label_norm, unlabel_norm = predictions_from_pool(model, X_pool, T, training, pool_size)
    
    uncertainty_values=uncertainty(outputs).reshape((-1,))

    diversity_values=diversity(model, label_norm, unlabel_norm).reshape((-1,))

    """
    Check the priority of metrics to decide search order
    """
    if model.args.priority == 0:
        idx1 = -(uncertainty_values).argsort()[:(n_query*model.args.candidate_ratio)]
        """
        Select candidates using the first metric
        """
        candidates=diversity_values[idx1]
        """
        Select final points from the candidates using the second metric
        """
        idx2= -(candidates).argsort()[:n_query]
        idx3=idx1[idx2]
        query_idx=random_subset[idx3]
        return query_idx, X_pool[query_idx]
    else:
        idx1 = -(diversity_values).argsort()[:(n_query*model.args.candidate_ratio)]
        candidates=uncertainty_values[idx1]
        idx2= -(candidates).argsort()[:n_query]
        idx3=idx1[idx2]
        query_idx=random_subset[idx3]
        return query_idx, X_pool[query_idx]


"""
Each possible hybrid query strategy. The name is the abbervation of methods considered. For example, ew = entropy + waal. It will check the runmode 
to decide proper query methods
"""
def ew(model, X_pool: np.ndarray, n_query: int = 10,T: int = 100, training: bool = True, pool_size: int = 200):
    if model.args.runmode!=0:
        return two_stage_query(model, X_pool, entropy, waal,n_query, T, training, pool_size)
    else:
        return combine_metric(model, X_pool, entropy, waal,n_query, T, training, pool_size)

def ed(model, X_pool: np.ndarray, n_query: int = 10,T: int = 100, training: bool = True, pool_size: int = 200):
    if model.args.runmode!=0:
        return two_stage_query(model, X_pool, entropy, density,n_query, T, training, pool_size)
    else:
        return combine_metric(model, X_pool, entropy, density,n_query, T, training, pool_size)

def em(model, X_pool: np.ndarray, n_query: int = 10,T: int = 100, training: bool = True, pool_size: int = 200):
    if model.args.runmode!=0:
        return two_stage_query(model, X_pool, entropy, minidis,n_query, T, training, pool_size)
    else:
        return combine_metric(model, X_pool, entropy, minidis, n_query, T, training, pool_size)

def bw(model, X_pool: np.ndarray, n_query: int = 10,T: int = 100, training: bool = True, pool_size: int = 200):
    if model.args.runmode!=0:
        return two_stage_query(model, X_pool, bald, waal,n_query, T, training, pool_size)
    else:
        return combine_metric(model, X_pool, bald, waal, n_query, T, training, pool_size)

def bd(model, X_pool: np.ndarray, n_query: int = 10,T: int = 100, training: bool = True, pool_size: int = 200):
    if model.args.runmode!=0:
        return two_stage_query(model, X_pool, bald, density,n_query, T, training, pool_size)
    else:
        return combine_metric(model, X_pool, bald, density,n_query, T, training, pool_size)

def bm(model, X_pool: np.ndarray, n_query: int = 10,T: int = 100, training: bool = True, pool_size: int = 200):
    if model.args.runmode!=0:
        return two_stage_query(model, X_pool, bald, minidis,n_query, T, training, pool_size)
    else:
        return combine_metric(model, X_pool, bald, minidis,n_query, T, training, pool_size)

def vw(model, X_pool: np.ndarray, n_query: int = 10,T: int = 100, training: bool = True, pool_size: int = 200):
    if model.args.runmode!=0:
        return two_stage_query(model, X_pool, var_ratios, waal,n_query, T, training, pool_size)
    else:
        return combine_metric(model, X_pool, var_ratios, waal,n_query, T, training, pool_size)

def vd(model, X_pool: np.ndarray, n_query: int = 10,T: int = 100, training: bool = True, pool_size: int = 200):
    if model.args.runmode!=0:
        return two_stage_query(model, X_pool, var_ratios, density,n_query, T, training, pool_size)
    else:
        return combine_metric(model, X_pool, var_ratios, density,n_query, T, training, pool_size)

def vm(model, X_pool: np.ndarray, n_query: int = 10,T: int = 100, training: bool = True, pool_size: int = 200):
    if model.args.runmode!=0:
        return two_stage_query(model, X_pool, var_ratios, minidis,n_query, T, training, pool_size)
    else:
        return combine_metric(model, X_pool, var_ratios, minidis, n_query, T, training, pool_size)

def uniform(model, X_pool: np.ndarray, n_query: int = 10,T: int = 100, training: bool = True, pool_size: int = 200):
    """
    conduct an uniform selection
    """
    outputs, random_subset, label_norm, unlabel_norm = predictions_from_pool(model, X_pool, T, training, pool_size)
    query_idx = np.random.choice(random_subset, size=n_query, replace=False)
    return query_idx, X_pool[query_idx]





def shannon_entropy_function(outputs,E_H=False):
    """H[y|x,D_train] := - sum_{c} p(y=c|x,D_train)log p(y=c|x,D_train)

    Attributes:
        outputs: model prediction for all points in the pool
        E_H: If True, compute H and EH for BALD (default: False)

    """
    pc = outputs.mean(axis=0)

    H = (-pc * np.log(pc + 1e-10)).sum(
        axis=-1
    )  # To avoid division with zero, add 1e-10
    if E_H:
        E = -np.mean(np.sum(outputs * np.log(outputs + 1e-10), axis=-1), axis=0)
        return H, E
    return H


def entropy(outputs):
    """
    Uncertainty measured by entropy
    """
    H = shannon_entropy_function(outputs)
    return H


def bald(outputs):
    """
    Uncertainty measured by bald
    """
    H, E_H = shannon_entropy_function(outputs, E_H=True)
    
    return H-E_H


def var_ratios(outputs):
    """
    Uncertainty measured by variation ratios
    """
    preds = np.argmax(outputs, axis=2)
    _, count = stats.mode(preds, axis=0)
    ratio = (1 - count / preds.shape[1]).reshape((-1,))
    return ratio

def waal(model, label_norm, unlabel_norm):
    """
    Diversity measured by waal (I will change the name later since actually no adversirial process)
    Attributes:
    model: the learner object to provide arguments for the discriminator
    label_norm: Standardized embedding matrix for labeled points
    unlabel_norm: Standardized embedding matrix for unlabeled points
    """

    """
    Create target values for the discriminator
    """
    label_target=np.ones(label_norm.shape[0]).reshape(-1,1)
    unlabel_target=np.zeros(unlabel_norm.shape[0]).reshape(-1,1)
    all_features=np.vstack((label_norm,unlabel_norm))
    all_targets=np.vstack((label_target,unlabel_target)).reshape(-1).astype(np.int64)
    index=label_norm.shape[0]
    
    """
    Train the discriminator and get diversity scores
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    discri1 = discri().to(device)
    discriminator = NeuralNetClassifier(
        module=discri1,
        lr=model.args.lr,
        batch_size=model.args.batch_size,
        max_epochs=model.args.epochs,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        train_split=None,
        verbose=0,
        device=device
    )


    discriminator.fit(all_features,all_targets)

    scores=discriminator.predict_proba(all_features)[index:,0]
    


    return(scores)


def density(model, label_norm, unlabel_norm):
    """
    Diversity measured by information density 
    """
    
    return(np.ones(unlabel_norm.shape[0]))

def minidis(model, label_norm, unlabel_norm):
    """
    Diversity measured by minimum distance 
    """
    
    return(np.ones(unlabel_norm.shape[0]))


class discri(nn.Module):
    def __init__(
        self,
        dense1: int =50,
        dense2: int =30,
        dense3: int =10,
        target: int =2
    ):

        """
        The discriminator to judge whether a point is labeled or not, consisting of two hidden layers
        Attributes:
        dense1: Dimension of the input
        dense2: Dimension of the first layer
        dense3: Dimension of the second layer
        target: Dimension of the target (i.e., the number of classes)
        """
        super(discri, self).__init__()
        self.fc1=nn.Linear(dense1,dense2)
        self.fc2=nn.Linear(dense2,dense3)
        self.fc3=nn.Linear(dense3,target)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        out = self.fc3(x)
        
        return out





