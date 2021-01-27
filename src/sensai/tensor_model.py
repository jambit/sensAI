"""
This module contains base classes for models that input and output tensors, for examples CNNs.
The fitting and predictions will still be performed on data frames, like in VectorModel,
but now it will be expected that all entries of the input data frame passed to the model are tensors of the same shape.
Lists of scalars of the same lengths are also accepted. The same is expected of the ground truth data frames.
Everything will work as well if the entries are just scalars but in this case it is recommended to use
VectorModel instead.

If we denote the shapes of entries in the dfs as inputTensorShape and outputTensorShape,
the model will be fit on input tensors of shape (N_rows, N_inputColumns, inputTensorShape) and output tensors of
shape (N_rows, N_outputColumns, outputTensorShape), where empty dimensions (e.g. for one-column data frames)
will be stripped.
"""


import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from sensai import VectorRegressionModel, VectorClassificationModel, VectorModel

log = logging.getLogger(__name__)
# we set the default level to debug because it is often interesting for the user to receive
# debug information about shapes as data frames get converted to arrays
log.setLevel(logging.DEBUG)


class InvalidShapeError(Exception): pass


def _getDatapointShape(df: pd.DataFrame):
    firstRowDf = df.iloc[:1]
    return _extractArray(firstRowDf)[0].shape


def _checkDfShape(df: pd.DataFrame, desiredShape: tuple):
    datapointShape = _getDatapointShape(df)
    if datapointShape != desiredShape:
        raise InvalidShapeError(f"Wrong input shape for data point. Expected {desiredShape} but got {datapointShape}")


def _extractArray(df: pd.DataFrame):
    """
    Extracts array from data frame. It is expected that each row corresponds to a data point and
    each column corresponds to a "channel". Moreover, all entries are expected to be arrays of the same shape
    (or scalars or sequences of the same length). We will refer to that shape as tensorShape.

    The output will be of shape (N_rows, N_columns, tensorShape). Thus, N_rows can be interpreted as dataset length
    (or batch size, if a single batch is passed) and N_columns can be interpreted as number of channels.
    Empty dimensions will be stripped, thus if the data frame has only one column, the array will have shape
    (N_rows, tensorShape).
    E.g. an image with three channels could equally be passed as data frame of the type


    | ----|-----R-----|-----G-----|-----B------
    | =========================================
    | 0---|--channel--|--channel--|--channel
    | 1---| ...

     or as df of the type

    | ----|----image----|
    | ====================
    | 0---|--RGBArray--|
    | 1---| ...

    In both cases the returned array will have shape (N_images, 3, width, height)

    :param df: data frame where each entry is an array of shape tensorShape
    :return: array of shape N_rows, N_columns, tensorShape with stripped empty dimensions
    """
    log.debug(f"Stacking tensors of shape {np.array(df.iloc[0, 0]).shape}")
    try:
        return np.stack(df.apply(np.stack, axis=1))
    except ValueError:
        raise ValueError(f"No array can be extracted from frame of length {len(df)} with columns {list(df.columns)}. "
                         f"Make sure that all entries have the same shape")


# This has to be implemented as a mixin because there can be no functional common class for tensor models.
# The reason is that actual implementations need to inherit from Vector-Regression/Classification-Model
# (or duplicate a lot of code) and thus it is not possible to inherit from something like TensorModel(VectorModel)
# without getting into a mess. Despite that, we want things to "be a TensorModel", hence this class.
class TensorModel(ABC):
    # will be set during _fitToTensorModel. None is immutable so it is safe to set the default on class level
    # We avoid using an init here to ensure the mixin-like nature of this class. Unfortunately, it is a bit hacky...
    _modelInputShape = None
    _modelOutputShape = None

    @abstractmethod
    def _fitToArray(self, X: np.ndarray, Y: np.ndarray):
        pass

    def _fitTensorModel(self, X: pd.DataFrame, Y: pd.DataFrame):
        """
        To be used within _fit in implementations of this class
        """
        log.debug(f"Stacking input tensors from columns {X.columns} and from all rows to a single array. "
                  f"Note that all tensors need to have the same shape")
        X = _extractArray(X)
        Y = _extractArray(Y)
        self._modelInputShape = X[0].shape
        self._modelOutputShape = Y[0].shape
        log.debug(f"Fitting on {len(X)} datapoints of shape {self._modelInputShape}. "
                  f"The ground truth are tensors of shape {self._modelOutputShape}")
        self._fitToArray(X, Y)

    def getModelInputShape(self) -> Optional[Tuple]:
        return self._modelInputShape

    def getModelOutputShape(self):
        return self._modelInputShape


class TensorToScalarRegressionModel(VectorRegressionModel, TensorModel, ABC):
    def __init__(self, checkInputShape=True, checkInputColumns=True):
        """
        Base class for regression models that take tensors as input and output scalars. They can be evaluated
        in the same way as non-tensor regression models

        :param checkInputShape: Whether to check if during predict input tensors have the same shape as during fit.
            For certain applications, e.g. using CNNs on larger inputs than the training set, this has
            to be disabled
        :param checkInputColumns: Whether to check if input columns at predict time coincide with those at fit time
        """
        super().__init__(checkInputColumns=checkInputColumns)
        self.checkInputShape = checkInputShape

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        self._fitTensorModel(X, Y)

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.checkInputShape:
            _checkDfShape(x, self.getModelInputShape())
        return super().predict(x)


class TensorToScalarClassificationModel(VectorClassificationModel, TensorModel, ABC):
    def __init__(self, checkInputShape=True, checkInputColumns=True):
        """
        Base class for classification models that take tensors as input and output scalars. They can be evaluated
        in the same way as non-tensor classification models

        :param checkInputShape: Whether to check if during predict input tensors have the same shape as during fit.
            For certain applications, e.g. using CNNs on larger inputs than the training set, this has
            to be disabled
        :param checkInputColumns: Whether to check if input columns at predict time coincide with those at fit time
        """
        super().__init__(checkInputColumns=checkInputColumns)
        self.checkInputShape = checkInputShape

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        self._fitTensorModel(X, Y)

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.checkInputShape:
            _checkDfShape(x, self.getModelInputShape())
        return super().predict(x)


# Note: for tensor to tensor models the output shape is not trivial. There will be dedicated evaluators
# and metrics for them. Examples for such models are auto-encoders, models performing semantic segregation,
# models for super-resolution and so on
class TensorToTensorRegressionModel(VectorRegressionModel, TensorModel, ABC):
    def __init__(self, checkInputShape=True, checkOutputShape=True, checkInputColumns=True):
        """
        Base class for regression models that output tensors. Multiple targets can be used by putting
        them into separate columns. In that case it is required that all target tensors have the same shape.

        :param checkInputShape: Whether to check if during predict tensors have the same shape as during fit.
            For certain applications, e.g. using CNNs on larger inputs than the training set, this has
            to be disabled
        :param checkOutputShape: Whether to check if predictions have the same shape as ground truth data during fit.
            For certain applications, e.g. using CNNs on larger inputs than the training set, this has
            to be disabled
        :param checkInputColumns: Whether to check if input columns at predict time coincide with those at fit time
        """
        super().__init__(checkInputColumns=checkInputColumns)
        self.checkInputShape = checkInputShape
        self.checkOutputShape = checkOutputShape

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        self._fitTensorModel(X, Y)

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        if not self.isFitted():
            # TODO: raise an Exception instead?
            log.warning(f"Calling predict with unfitted model. "
                        f"This might lead to errors down the line, especially if input/output checks are enabled")
        if self.checkInputShape:
            _checkDfShape(x, self.getModelInputShape())
        y = super().predict(x)
        if self.checkOutputShape:
            _checkDfShape(y, self.getModelOutputShape())
        return y


class TensorToTensorClassificationModel(VectorModel, TensorModel, ABC):
    def __init__(self, checkInputShape=True, checkOutputShape=True, checkInputColumns=True):
        """
        Base class for classification models that output tensors, e.g. for semantic segregation. The models
        can be fit on a ground truth data frame with a single column. The entries in this column should be
        binary tensors with one-hot-encoded labels, i.e. of shape `(*predictionShape, numLabels)`

        :param checkInputShape: Whether to check if during predict tensors have the same shape as during fit.
            For certain applications, e.g. using CNNs on larger inputs than the training set, this has
            to be disabled
        :param checkOutputShape: Whether to check if predictions have the same shape as ground truth data during fit.
            For certain applications, e.g. using CNNs on larger inputs than the training set, this has
            to be disabled
        :param checkInputColumns: Whether to check if input columns at predict time coincide with those at fit time
        """
        super().__init__(checkInputColumns=checkInputColumns)
        self.checkInputShape = checkInputShape
        self.checkOutputShape = checkOutputShape
        self._numPredictedClasses: Optional[int] = None

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        self._fitTensorModel(X, Y)

    def isRegressionModel(self) -> bool:
        return False

    def getNumPredictedClasses(self):
        return self._numPredictedClasses

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, fitPreprocessors=True):
        """

        :param X: data frame containing input tensors on which to train
        :param Y: ground truth has to be an array containing only zeroes and ones (one-hot-encoded labels) of the shape
            `(*predictionShape, numLabels)`

        :param fitPreprocessors:
        :return:
        """
        if len(Y.columns) != 1:
            raise ValueError(f"{self.__class__.__name__} requires exactly one output "
                             f"column with tensors containing one-hot-encoded labels")

        # checking if Y is a binary array of one hot encoded labels
        dfYToCheck = _extractArray(Y.iloc[:5])
        if not np.array_equal(dfYToCheck, dfYToCheck.astype(bool)):
            raise Exception(f"Ground truth data points have to be binary arrays of one-hot-encoded labels "
                            f"of shape (*predictionShape, numLabels). Did you forget to one-hot-encode your labels "
                            f"before training?")
        # dfYToCheck has shape (N_datapoints=5, *predictionShape, N_labels)
        predictionShape = dfYToCheck.shape[1:-1]
        if len(predictionShape) == 0:
            raise InvalidShapeError(f"Ground truth data points have to be binary arrays of one-hot-encoded labels "
                                    f"of shape (*predictionShape, numLabels). However, received array of trivial "
                                    f"predictionShape. If the predictions are scalars, a TensorToScalarClassificationModel "
                                    f"should be used instead of {self.__class__.__name__}")
        self._numPredictedClasses = dfYToCheck.shape[-1]
        super().fit(X, Y, fitPreprocessors=fitPreprocessors)

    def getModelOutputShape(self):
        # The ground truth contains one-hot-encoded labels in the last dimension
        # The model output predicts the labels as ints, without one-hot-encoding
        oneHotEncodedOutputShape = super().getModelOutputShape()
        if oneHotEncodedOutputShape is None:
            return None
        return oneHotEncodedOutputShape[:-1]

    def convertClassProbabilitiesToPredictions(self, df: pd.DataFrame):
        """
        Converts from a result returned by predictClassProbabilities to a result as return by predict.

        :param df: the output data frame from predictClassProbabilities
        :return: an output data frame as it would be returned by predict
        """
        df = df.copy()
        colName = self.getPredictedVariableNames()[0]
        df[colName] = df[colName].apply(lambda probasArray: probasArray.argmax(axis=-1))
        return df

    def predictClassProbabilities(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        :param x: the input data
        :return: a data frame where the list of columns is the list of class labels and the values are probabilities.
            Returns None if the classifier cannot predict probabilities.
        """
        x = self._computeModelInputs(x)
        if self.checkInputShape:
            _checkDfShape(x, self.getModelInputShape())
        result = self._predictClassProbabilities(x)
        self._checkPrediction(result)
        return result

    def _checkPrediction(self, predictionDf: pd.DataFrame, maxRowsToCheck=5):
        """
        Checks whether the column name is correctly, whether the shapes match set and whether the entries
        correspond to probabilities
        """
        if self.checkOutputShape:
            _checkDfShape(predictionDf, self.getModelOutputShape())

        arrayToCheck = _extractArray(predictionDf.iloc[:maxRowsToCheck])

        if not np.all(0 <= arrayToCheck) or not np.all(arrayToCheck <= 1):
            log.warning(f"Probability arrays may not be correctly normalised, "
                        f"got probabilities outside the range [0, 1]")

        s = arrayToCheck.sum(axis=-1)
        if not np.all(np.isclose(s, 1)):
            log.warning(
                f"Probability array data frame may not be correctly normalised, "
                f"received probabilities do not sum to 1")

    @abstractmethod
    def _predictClassProbabilities(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        If you are implementing a probabilistic classifier, this method has to return a data frame
        containing an array with probabilities, of shape `(*tensorShape, numLabels)`.
        The default implementation of _predict will then use the output of this method and convert it to an array
        of predicted labels of shape `tensorShape` (via argmax).

        In case you want to predict labels only or have a more efficient implementation of predicting labels than
        using argmax, your will have to override _predict in your implementation. In the former case of a
        non-probabilistic classifier, the implementation of this method should raise an exception, like the one below.
        """
        raise NotImplementedError(f"Model {self.__class__.__name__} does not support prediction of probabilities")

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        try:
            predictedProbabilitiesDf = self._predictClassProbabilities(x)
        except Exception:
            raise Exception(f"Wrong implementation of {self.__class__.__name__}. For non-probabilistic classifiers "
                            "_predict has to be overrode!")
        return self.convertClassProbabilitiesToPredictions(predictedProbabilitiesDf)

    # TODO or not TODO: I don't see how to reduce the code duplication here...
    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Returns an array of integers. If the model was fitted on binary ground truth arrays of
        shape `(*tensorShape, numLabels)`, predictions will have the shape `tensorShape` and contain integers
        0, 1, ..., numLabels - 1. They correspond to the predicted labels
        """
        if not self.isFitted():
            # TODO: raise an Exception instead?
            log.warning(f"Calling predict with unfitted model. "
                        f"This might lead to errors down the line, especially if input/output checks are enabled")
        if self.checkInputShape:
            _checkDfShape(x, self.getModelInputShape())
        y = super().predict(x)
        if self.checkOutputShape:
            _checkDfShape(y, self.getModelOutputShape())
        return y