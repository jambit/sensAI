from . import columngen
from . import data_transformation
from . import featuregen
from . import hyperopt
from . import local_search
from . import naive_bayes
from . import nearest_neighbors
from . import sklearn
from . import util
from .data_ingest import InputOutputDataFrames
from .data_transformation import DataFrameTransformer, RuleBasedDataFrameTransformer
from .ensemble import AveragingVectorRegressionModel
from .evaluation.eval_stats import eval_stats_classification, eval_stats_regression
from .normalisation import NormalisationMode
from .tensor_model import TensorToTensorRegressionModel, TensorToScalarRegressionModel, \
    TensorToTensorClassificationModel, TensorToScalarClassificationModel
from .vector_model import VectorModel, VectorRegressionModel, VectorClassificationModel

__version__ = "0.0.5.dev0"

# The following submodules are not imported by default to avoid necessarily requiring their dependencies:
# tensorflow
# torch
# lightgbm
# catboost
