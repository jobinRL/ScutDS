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
    classes: array-like
        List of the classes in the stream used for this model.
    model: classifier (optional)
        Model to use once ScutDS is done resampling. Can use any one that is in Sci-Kit Multiflow. if null then HoeffdingTree is used.
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Notes
    -----
    This algorithm is specially created to work on imbalanced dataset and will have close to no significant difference
    if used with a balanced dataset.




    """
    def __init__(self, classes, model = HoeffdingTree(),
                 random_state = None):

        super().__init__()

        self._classes = classes
        self._alpha = 1/len(self._classes)
        self._model = model
        self._majority_cutoff = 1
        self._training_set_X = []
        self._training_set_y = []
        self._num_instance_per_class = {}
        self._batch_num = 1
        self._original_random_state = random_state
        self.random_state = None

        for var in self._classes:
            self._num_instance_per_class[var] = 0;


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
        for var in self._classes:
            X_by_class[var] = []
            class_in_training_set[var] = False
            class_majority[var] = False

        for i in range(0,len(y)):
            X_by_class[y[i]].append(X[i])

        if(self._batch_num == 1):
            if(not isinstance(X,list)):
                self._actual_instances_X = X.tolist()
            else:
                self._actual_instances_X = X

            if(not isinstance(y,list)):
                self._actual_instances_y = y.tolist()
            else:
                self._actual_instances_y = y
            for var in self._classes:
                if(self._num_instance_per_class[var] > self._majority_cutoff):
                    X_by_class[var] = self.majority_sampling(X_by_class[var])
                    class_majority[var] = True
        else:


            #merge the training set without the syntethic values into the main set.
            for i in range(len(self._actual_instances_X)):
                X_by_class[self._actual_instances_y[i]].append(self._actual_instances_X[i])



            if(not isinstance(X,list)):
                self._actual_instances_X.extend(X.tolist())
            else:
                self._actual_instances_X.extend(X)
            if(not isinstance(y,list)):
                self._actual_instances_y.extend(y.tolist())
            else:
                self._actual_instances_y.extend(y)

            for var in self._classes:
                if(self._num_instance_per_class[var] > self._majority_cutoff):
                    X_by_class[var] = self.majority_sampling(X_by_class[var])
                    class_majority[var] = True


        self._training_set_X = []
        self._training_set_y = []
        smote_neighbors = 5
        first_majority = True

        for var in self._classes:
            if((not class_majority[var]) or first_majority):
                smote_neighbors = min(len(X_by_class[var]), smote_neighbors)
                self._training_set_X.extend(X_by_class[var])
                for i in range(len(X_by_class[var])):
                    self._training_set_y.append(var)
                class_in_training_set[var] = True
                if(class_majority[var]):
                    first_majority = False


        actual_instances = self._training_set_X

        self.random_state = check_random_state(self._original_random_state)

        sm = SMOTE(random_state = self.random_state, k_neighbors = smote_neighbors)
        self._training_set_X, self._training_set_y = sm.fit_resample(self._training_set_X, self._training_set_y)


        self._training_set_X = self._training_set_X.tolist()
        self._training_set_y = self._training_set_y.tolist()
        for var in self._classes:
            if(not class_in_training_set[var]):
                self._training_set_X.extend(X_by_class[var])
                for i in range(len(X_by_class[var])):
                    self._training_set_y.append(var)


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

        """
        total_instances = 0

        for i in range(0,len(y)):
            self._num_instance_per_class[y[i]] = self._num_instance_per_class[y[i]] + 1

        for var in self._classes:
            total_instances = total_instances + self._num_instance_per_class[var]

        self._majority_cutoff = math.floor(total_instances*self._alpha)

        self.resample(X,y)
        self._model.partial_fit(self._training_set_X,self._training_set_y,self._classes)

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
        self.__init__(self._classes, self._original_random_state, self._model.reset())
