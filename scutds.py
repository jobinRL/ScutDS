import logging
import textwrap
import math
from abc import ABCMeta
from operator import attrgetter
import numpy as np
from skmultiflow.utils.utils import get_dimensions, normalize_values_in_dict, calculate_object_size
from skmultiflow.core.base import StreamModel
from imblearn.over_sampling import SMOTE
from skmultiflow.trees import HoeffdingTree
from sklearn.cluster import KMeans
from skmultiflow.utils import check_random_state

GINI_SPLIT = 'gini'
INFO_GAIN_SPLIT = 'info_gain'
MAJORITY_CLASS = 'mc'
NAIVE_BAYES = 'nb'
NAIVE_BAYES_ADAPTIVE = 'nba'

# logger
logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class ScutDS(StreamModel):
    """ ScutDS algorithm. Performs oversampling and undersampling of the minority and majority classes respectively and then gives the input Data
        to another selected model.

    Parameters
    ----------
    model: classifier (optional)
        Model to use once ScutDS is done resampling. Can use any one that is in Sci-Kit Multiflow. if null then HoeffdingTree is used.
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Raises
    ------

    ValueError: A ValueError is raised if the 'classes' parameter is
    not passed in the first partial_fit call.

    Notes
    -----
    This algorithm is specially created to work on imbalanced dataset and will have close to no significant difference
    if used with a balanced dataset.

    Examples
    --------
    from skmultiflow.data import WaveformGenerator
    from skmultiflow.trees import HoeffdingTree
    from skmultiflow.evaluation import EvaluatePrequential

    import scutds
    # 1. Create a stream
    stream = WaveformGenerator()
    stream.prepare_for_use()

    # 2. Instantiate the HoeffdingTree classifier
    sd = scutds.ScutDS()

    # 3. Setup the evaluator
    evaluator = EvaluatePrequential(show_plot=False,
                                    pretrain_size=1000,
                                    batch_size = 1000,
                                    max_samples=10000)

    # 4. Run evaluation
    evaluator.evaluate(stream=stream, model=sd)

    >>>Prequential Evaluation
    >>>Evaluating 1 target(s).
    >>>Pre-training on 1000 sample(s).
    >>>Evaluating...
    >>>##------------------ [10%] [4.95s]
    >>>####---------------- [20%] [9.18s]
    >>>######-------------- [30%] [13.87s]
    >>>########------------ [40%] [16.40s]
    >>>##########---------- [50%] [19.84s]
    >>>############-------- [60%] [23.25s]
    >>>##############------ [70%] [26.62s]
    >>>################---- [80%] [29.79s]
    >>>##################-- [90%] [32.91s]
    >>>##################-- [90%] [32.91s]
    >>>Processed samples: 10000
    >>>Mean performance:
    >>>M0 - Accuracy     : 0.7999
    >>>M0 - Kappa        : 0.6989

    """
    def __init__(self, model = HoeffdingTree(),
                 random_state = None):

        super().__init__()

        self.classes = None
        self._alpha = 0
        self._model = model
        self._majority_cutoff = 1
        self._training_set_X = []
        self._training_set_y = []
        self._batch_num = 1

        self._past_instances = {}

        self._original_random_state = random_state
        self.random_state = None

    def majority_sampling(self,X):
        self.random_state = check_random_state(self._original_random_state)
        permutation = self.random_state.permutation(len(X))
        n_clusters = 5
        neighbours = KMeans(n_clusters).fit_predict(X)

        count = 0
        max_per_cluster = math.floor(self._majority_cutoff / n_clusters)
        resampledX = []
        clusters_count = {}

        for i in range(n_clusters):
            clusters_count[i] = 0

        i = 0

        while (count < self._majority_cutoff and i < len(X)):
            curr_label = neighbours[permutation[i]]
            curr_X = X[permutation[i]]
            if(clusters_count[curr_label] < max_per_cluster):
                resampledX.append(curr_X)
                count += 1
            i += 1
        return resampledX


    def resample(self,X,y):

        X_by_class = {}
        class_in_training_set = {}
        class_majority = {}
        j = {}
        for var in self.classes:
            X_by_class[var] = []
            class_in_training_set[var] = False
            class_majority[var] = False
            j[var] = 0

        for i in range(0,len(y)):
            X_by_class[y[i]].append(X[i])
            if(j[y[i]] <= 4):
                self._past_instances[y[i]][j[y[i]]] = X[i]
                j[y[i]] += 1

        for var in self.classes:
            if(len(X_by_class[var]) > self._majority_cutoff):
                X_by_class[var] = self.majority_sampling(X_by_class[var])
                class_majority[var] = True


        self._training_set_X = np.array([],dtype = list)
        self._training_set_y = np.array([],dtype = int)
        smote_neighbors = 5
        first_majority = True

        for var in self.classes:
            if((not class_majority[var]) or first_majority):
                j = 4
                while(len(X_by_class[var]) < 5 and j > 0):

                    if(self._past_instances[var][j] != 0):
                        X_by_class[var].append(self._past_instances[var][j])
                    j -= 1

                if(self._training_set_X != [] and X_by_class[var] != []):
                    self._training_set_X = np.concatenate((self._training_set_X,X_by_class[var]), axis = 0)
                    self._training_set_y = np.concatenate((self._training_set_y, np.full(len(X_by_class[var]),var)), axis = 0)
                else:
                    self._training_set_X = np.array(X_by_class[var], dtype = list)
                    self._training_set_y = np.array(np.full(len(X_by_class[var]),var), dtype = int)

                smote_neighbors = min(len(X_by_class[var]), smote_neighbors)
                class_in_training_set[var] = True
                if(class_majority[var]):
                    first_majority = False


        if(smote_neighbors > 1):
            sm = SMOTE(random_state = self.random_state, k_neighbors = smote_neighbors-1)
            self._training_set_X, self._training_set_y = sm.fit_resample(self._training_set_X, self._training_set_y)




        for var in self.classes:
            if(not class_in_training_set[var]):
                self._training_set_X = np.concatenate((self._training_set_X,X_by_class[var]), axis = 0)
                self._training_set_y = np.concatenate((self._training_set_y, np.full(len(X_by_class[var]),var)), axis = 0)


    def fit(self, X, y, classes, weight=None):
        raise NotImplementedError

    def partial_fit(self, X, y, classes = None):
        """ Partially fits the model.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            The feature's matrix.

        y: numpy.ndarray, shape (n_samples)
            The class labels for all samples in X.

        classes: list
            List of all existing classes. This is an optional parameter, except
            for the first partial_fit call, when it becomes obligatory.

        """
        if self.classes is None:
            if classes is None:
                raise ValueError("The first partial_fit call should pass all the classes.")
            else:
                self.classes = classes
                self._alpha = 1/len(self.classes)
                if(self._batch_num <= 1):
                    for var in self.classes:
                        self._past_instances[var] = np.zeros(5, dtype = list)
        self._majority_cutoff = math.floor(len(y)*self._alpha)

        self.resample(X,y)
        self._model.partial_fit(self._training_set_X,self._training_set_y,self.classes)

        self._batch_num = self._batch_num + 1

    def predict(self, X):
        """ Uses the current model to predict samples in X.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            The feature's matrix.

        Returns
        -------
        numpy.ndarray
            An array containing the predicted labels for all instances in X.

        Note
        ----
        Only works if the model matched with ScutDS has this method.

        """
        return self._model.predict(X)

    def predict_proba(self, X):
        """ Predicts the probability of each sample belonging to each one of the
        known classes.

        Parameters
        ----------
        X: Numpy.ndarray, shape (n_samples, n_features)
            A matrix of the samples we want to predict.

        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, n_features), in which each outer entry is
            associated with the X entry of the same index. And where the list in
            index [i] contains len(self.target_values) elements, each of which represents
            the probability that the i-th sample of X belongs to a certain label.

        Note
        ----
        Only works if the model matched with ScutDS has this method.
        """
        return self._model.predict_proba(X)

    def score(self, X, y):
        return self._model.score(X,y)

    def get_info(self):
        """ Collects information about the classifier's configuration.

        Returns
        -------
        string
            Configuration for this classifier instance and the model it uses.
        """
        description = type(self).__name__ + ': '
        description += 'random_state: {} - '.format(self._original_random_state)
        description += 'model: {} - '.format(self._model.get_info())
        return description

    def reset(self):
        self.__init__(self._original_random_state, self._model.reset())
