"""
This module defines base classes for models that use pandas.DataFrames for inputs and outputs, where each data frame row represents
a single model input or output. Since every row contains a vector of data (one-dimensional array), we refer to them as vector-based
models. Hence the name of the module and of the central base class VectorModel.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Any, Optional, Union, Type

import numpy as np
import pandas as pd

from .data_transformation import DataFrameTransformer, DataFrameTransformerChain, InvertibleDataFrameTransformer
from .featuregen import FeatureGenerator, FeatureCollector
from .util.cache import PickleLoadSaveMixin
from .util.sequences import getFirstDuplicate

log = logging.getLogger(__name__)


class PredictorModel(ABC):
    """
    Base class for models that map data frames to predictions
    """

    def __init__(self):
        self._name = None

    @abstractmethod
    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def isRegressionModel(self) -> bool:
        pass

    @abstractmethod
    def getPredictedVariableNames(self) -> list:
        pass

    def withName(self, name: str):
        """
        Sets the model's name.

        :param name: the name
        :return: self
        """
        self.setName(name)
        return self

    def setName(self, name):
        self._name = name

    def getName(self):
        if self._name is None:
            return "unnamed-%s-%x" % (self.__class__.__name__, id(self))
        return self._name


class FittableModel(PredictorModel, ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        pass

    @abstractmethod
    def isFitted(self) -> bool:
        pass


class VectorModel(FittableModel, PickleLoadSaveMixin, ABC):
    """
    Base class for models that map data frames to predictions and can be fitted on data frames
    """

    def __init__(self, checkInputColumns=True):
        """
        :param checkInputColumns: Whether to check if the input column list (after feature generation)
            during inference coincides with the input column list during fit.
            This should be disabled if feature generation is not performed by the model itself,
            e.g. in ensemble models.
        """
        super().__init__()
        self._featureGenerator: Optional[FeatureGenerator] = None
        self._inputTransformerChain = DataFrameTransformerChain()
        self._isFitted = False  # Note: this keeps track only of the actual model being fitted, not the pre/postprocessors
        self._predictedVariableNames: Optional[list] = None
        self._modelInputVariableNames: Optional[list] = None
        self.checkInputColumns = checkInputColumns

    def withInputTransformers(self, *inputTransformers: Union[DataFrameTransformer, List[DataFrameTransformer]]) -> __qualname__:
        """
        Makes the model use the given input transformers. Call with empty input to remove existing input transformers.

        :param inputTransformers: DataFrameTransformers for the transformation of inputs
        :return: self
        """
        self._inputTransformerChain = DataFrameTransformerChain(*inputTransformers)
        return self

    def withFeatureGenerator(self, featureGenerator: Optional[FeatureGenerator]) -> __qualname__:
        """
        Makes the model use the given feature generator, which shall be used to compute
        the actual inputs of the model from the data frame that is given.
        Cannot be used in conjunction with withFeatureCollector

        Note: Feature computation takes place before input transformation.

        :param featureGenerator: the feature generator to use for input computation
        :return: self
        """
        self._featureGenerator = featureGenerator
        return self

    def withFeatureCollector(self, featureCollector: FeatureCollector) -> __qualname__:
        """
        Makes the model use the given feature collector's multi-feature generator
        in order compute the actual inputs of the model from the data frame that is given.
        Cannot be used in conjunction with withFeatureGenerator.

        Note: Feature computation takes place before input transformation.

        :param featureCollector: the feature collector whose feature generator shall be used for input computation
        :return: self
        """
        self._featureGenerator = featureCollector.getMultiFeatureGenerator()
        return self

    def _preProcessorsAreFitted(self):
        result = self._inputTransformerChain.isFitted()
        if self.getFeatureGenerator() is not None:
            result = result and self.getFeatureGenerator().isFitted()
        return result

    def isFitted(self):
        underlyingModelIsFitted = not self._underlyingModelRequiresFitting() or self._isFitted
        if not underlyingModelIsFitted:
            return False
        if not self._preProcessorsAreFitted():
            return False
        return True

    def _checkModelInputColumns(self, modelInput: pd.DataFrame):
        if self.checkInputColumns and list(modelInput.columns) != self._modelInputVariableNames:
            raise Exception(f"Inadmissible input data frame: "
                            f"expected columns {self._modelInputVariableNames}, got {list(modelInput.columns)}")

    def computeModelInputs(self, X):
        """
        Returns the dataframe that is passed to the model, i.e. the result of applying preprocessors to X.
        """
        return self._computeModelInputs(X)

    def _computeModelInputs(self, X: pd.DataFrame, Y: pd.DataFrame = None, fit=False) -> pd.DataFrame:
        """
        :param X:
        :param Y: Only has to be provided if fit is True and preprocessors require Y for fitting
        :param fit: if True, preprocessors will be fitted before being applied to X
        :return:
        """
        if fit:
            if self._featureGenerator is not None:
                X = self._featureGenerator.fitGenerate(X, Y, self)
            X = self._inputTransformerChain.fitApply(X)
        else:
            if self._featureGenerator is not None:
                X = self._featureGenerator.generate(X, self)
            X = self._inputTransformerChain.apply(X)
        return X

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Performs a prediction for the given input data frame

        :param x: the input data
        :return: a DataFrame with the same index as the input
        """
        if not self.isFitted():
            raise Exception(f"Calling predict with unfitted model. "
                            f"This might lead to errors down the line, especially if input/output checks are enabled")
        x = self._computeModelInputs(x)
        self._checkModelInputColumns(x)
        y = self._predict(x)
        y.index = x.index
        return y

    @abstractmethod
    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        pass

    def _underlyingModelRequiresFitting(self) -> bool:
        """
        Designed to be overridden for rule-based models.

        :return: True iff the underlying model requires fitting
        """
        return True

    def _fitPreprocessors(self, X: pd.DataFrame, Y: pd.DataFrame = None):
        # no need for fitGenerate if chain is empty
        if self._featureGenerator is not None:
            if len(self._inputTransformerChain) == 0:
                self._featureGenerator.fit(X, Y)
            else:
                X = self._featureGenerator.fitGenerate(X, Y, self)
        self._inputTransformerChain.fit(X)

    def fit(self, X: pd.DataFrame, Y: Optional[pd.DataFrame], fitPreprocessors=True):
        """
        Fits the model using the given data

        :param X: a data frame containing input data
        :param Y: a data frame containing output data. None may be passed if the underlying model does not require
            fitting, e.g. with rule-based models
        :param fitPreprocessors: if False, the model's feature generator and input transformers will not be fitted.
            If a preprocessor requires fitting, was not separately fit before and this option is set to False,
            an exception will be raised.
        """
        log.info(f"Training {self.__class__.__name__}")
        self._predictedVariableNames = list(Y.columns)
        if not self._underlyingModelRequiresFitting():
            self._fitPreprocessors(X, Y=Y)
        else:
            if Y is None:
                raise Exception(f"The underlying model requires a data frame for fitting but Y=None was passed")
            X = self._computeModelInputs(X, Y=Y, fit=fitPreprocessors)
            self._modelInputVariableNames = list(X.columns)
            log.info(
                f"Training with outputs[{len(Y.columns)}]={list(Y.columns)}, inputs[{len(self._modelInputVariableNames)}]=[{', '.join([n + '/' + X[n].dtype.name for n in self._modelInputVariableNames])}]")
            self._fit(X, Y)
            self._isFitted = True

    @abstractmethod
    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        pass

    def getPredictedVariableNames(self):
        return self._predictedVariableNames

    def getInputTransformer(self, cls: Type[DataFrameTransformer]):
        for it in self._inputTransformerChain.dataFrameTransformers:
            if isinstance(it, cls):
                return it
        return None

    def getInputTransformerChain(self):
        return self._inputTransformerChain

    def setFeatureGenerator(self, featureGenerator: Optional[FeatureGenerator]):
        self.withFeatureGenerator(featureGenerator)

    def getFeatureGenerator(self) -> Optional[FeatureGenerator]:
        return self._featureGenerator


class VectorRegressionModel(VectorModel, ABC):
    def __init__(self, checkInputColumns=True):
        """
        :param checkInputColumns: Whether to check if the input column list (after feature generation)
            during inference coincides with the input column list during fit.
            This should be disabled if feature generation is not performed by the model itself,
            e.g. in ensemble models.
        """
        super().__init__(checkInputColumns=checkInputColumns)
        self._outputTransformerChain = DataFrameTransformerChain()
        self._modelOutputVariableNames: Optional[list] = None
        self._targetTransformer: Optional[InvertibleDataFrameTransformer] = None

    def isRegressionModel(self) -> bool:
        return True

    def withOutputTransformers(self, *outputTransformers: Union[DataFrameTransformer, List[DataFrameTransformer]]) -> __qualname__:
        """
        Makes the model use the given output transformers. Call with empty input to remove existing output transformers.
        The transformers are ignored during the fit phase. Not supported for rule-based models.

        **Important**: The output columns names of the last output transformer should be the same
        as the first one's input column names. If this fails to hold, an exception will be raised when .predict() is called
        (fit will run through without problems, though).

        **Note**: Output transformers perform post-processing after the actual predictions have been made. Contrary
        to invertible target transformers, they are not invoked during the fit phase. Therefore, any losses computed there,
        including the losses on validation sets (e.g. for early stopping), will be computed on the non-post-processed data.
        A possible use case for such post-processing is if you know how improve the predictions of your fittable model
        by some heuristics or by hand-crafted rules.

        **How not to use**: Output transformers are not meant to transform the predictions into something with a
        different semantic meaning (e.g. normalized into non-normalized or something like that) - you should consider
        using a targetTransformer for this purpose. Instead, they give the possibility to improve predictions through
        post processing, when this is desired.

        :param outputTransformers: DataFrameTransformers for the transformation of outputs
            (after the model has been applied)
        :return: self
        """
        # There is no reason for post processing in rule-based models
        if not self._underlyingModelRequiresFitting():
            raise Exception(f"Output transformers are not supported for model of type {self.__class__.__name__}")
        self._outputTransformerChain = DataFrameTransformerChain(*outputTransformers)
        return self

    def withTargetTransformer(self, targetTransformer: Optional[InvertibleDataFrameTransformer]) -> __qualname__:
        """
        Makes the model use the given target transformers. Not supported for rule-based models.

        NOTE: all feature generators and data frame transformers will be fit on the untransformed target.
        The targetTransformer only affects the fit of the internal model.

        :param targetTransformer: a transformer which transforms the targets (training data outputs) prior to learning the model, such
            that the model learns to predict the transformed outputs. When predicting, the inverse transformer is applied after applying
            the model, i.e. the transformation is completely transparent when applying the model.
        :return: self
        """
        # Disabled for rule-based models which do not apply fitting and therefore cannot make use of transformed targets
        if not self._underlyingModelRequiresFitting():
            raise Exception(f"Target transformers are not supported for model of type {self.__class__.__name__}")
        self._targetTransformer = targetTransformer
        return self

    def getTargetTransformer(self):
        return self._targetTransformer

    def getOutputTransformerChain(self):
        return self._outputTransformerChain

    def _applyPostProcessing(self, y: pd.DataFrame):
        if self._targetTransformer is not None:
            y = self._targetTransformer.applyInverse(y)
        y = self._outputTransformerChain.apply(y)

        if list(y.columns) != self.getPredictedVariableNames():
            raise Exception(
                f"The model's predicted variable names are not correct. Got "
                f"{list(y.columns)} but expected {self.getPredictedVariableNames()}. "
                f"This kind of error can happen if the model's outputTransformerChain changes a data frame's "
                f"columns (e.g. renames them or changes order). Only output transformer chains that do not change "
                f"columns are permitted in VectorModel. You can fix this by modifying this instance's outputTransformerChain, "
                f"e.g. by calling .withOutputTransformers() with the correct input "
                f"(which can be empty to remove existing output transformers)"
            )
        return y

    def fit(self, X: pd.DataFrame, Y: Optional[pd.DataFrame], fitPreprocessors=True):
        """
        Fits the model using the given data

        :param X: a data frame containing input data
        :param Y: a data frame containing output data. None may be passed if the underlying model does not require
            fitting, e.g. with rule-based models
        :param fitPreprocessors: if False, the model's feature generator and input transformers will not be fitted.
            If a preprocessor requires fitting, was not separately fit before and this option is set to False,
            an exception will be raised.
        """
        if self._targetTransformer is not None:
            Y = self._targetTransformer.fitApply(Y)
        self._modelOutputVariableNames = list(Y.columns)
        super().fit(X, Y, fitPreprocessors=fitPreprocessors)

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        y = super().predict(x)
        return self._applyPostProcessing(y)

    def isFitted(self):
        if not super().isFitted():
            return False
        if self._targetTransformer is not None and not self._targetTransformer.isFitted():
            return False
        if not self._outputTransformerChain.isFitted():
            return False
        return True

    def getModelOutputVariableNames(self):
        """
        Gets the list of variable names predicted by the underlying model.
        For the case where at training time the ground truth is transformed by a target transformer
        which changes column names, the names of the variables prior to the transformation will be returned.
        Thus this method always returns the variable names that are actually predicted by the underlying model alone.
        For the variable names that are ultimately output by the entire VectorModel instance when calling predict,
        use getPredictedVariableNames.
        """
        return self._modelOutputVariableNames


class VectorClassificationModel(VectorModel, ABC):
    def __init__(self, checkInputColumns=True):
        """
        :param checkInputColumns: Whether to check if the input column list (after feature generation)
            during inference coincides with the input column list during fit.
            This should be disabled if feature generation is not performed by the model itself,
            e.g. in ensemble models.
        """
        super().__init__(checkInputColumns=checkInputColumns)
        self._labels = None

    def isRegressionModel(self) -> bool:
        return False

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        if len(Y.columns) != 1:
            raise ValueError("Classification requires exactly one output column with class labels")
        self._labels = sorted([label for label in Y.iloc[:, 0].unique()])
        self._fitClassifier(X, Y)

    def getClassLabels(self) -> List[Any]:
        return self._labels

    @abstractmethod
    def _fitClassifier(self, X: pd.DataFrame, y: pd.DataFrame):
        pass

    def convertClassProbabilitiesToPredictions(self, df: pd.DataFrame):
        """
        Converts from a data frame as returned by predictClassProbabilities to a result as return by predict.

        :param df: the output data frame from predictClassProbabilities
        :return: an output data frame as it would be returned by predict
        """
        labels = self.getClassLabels()
        dfCols = list(df.columns)
        if sorted(dfCols) != labels:
            raise ValueError(f"Expected data frame with columns {labels}, got {dfCols}")
        yArray = df.values
        maxIndices = np.argmax(yArray, axis=1)
        result = [dfCols[i] for i in maxIndices]
        return pd.DataFrame(result, columns=self.getPredictedVariableNames())

    def predictClassProbabilities(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        :param x: the input data
        :return: a data frame where the list of columns is the list of class labels and the values are probabilities.
            Raises an exception if the classifier cannot predict probabilities.
        """
        if not self.isFitted():
            raise Exception(f"Calling predict with unfitted model. "
                            f"This might lead to errors down the line, especially if input/output checks are enabled")
        x = self._computeModelInputs(x)
        result = self._predictClassProbabilities(x)
        self._checkPrediction(result)
        return result

    def _checkPrediction(self, predictionDf: pd.DataFrame, maxRowsToCheck=5):
        """
        Checks whether the column names are correctly set, sorted and whether the entries correspond to probabilities
        """
        labels = self.getClassLabels()
        if list(predictionDf.columns) != labels:
            raise Exception(f"{self} _predictClassProbabilities returned DataFrame with incorrect columns: "
                            f"expected {labels}, got {predictionDf.columns}")

        dfToCheck = predictionDf.iloc[:maxRowsToCheck]
        for i, (_, valueSeries) in enumerate(dfToCheck.iterrows(), start=1):

            if not all(0 <= valueSeries) or not all(valueSeries <= 1):
                log.warning(f"Probabilities data frame may not be correctly normalised, "
                            f"got probabilities outside the range [0, 1]: checked row {i}/{maxRowsToCheck} contains {list(valueSeries)}")

            s = valueSeries.sum()
            if not np.isclose(s, 1, atol=1e-2):
                log.warning(
                    f"Probabilities data frame may not be correctly normalised: checked row {i}/{maxRowsToCheck} contains {list(valueSeries)}")

    @abstractmethod
    def _predictClassProbabilities(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        If you are implementing a probabilistic classifier, this method has to return a data frame with probabilities
        (one column per label). The default implementation of _predict will then use the output of
        this method and convert it to predicted labels (via argmax).

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


class RuleBasedVectorRegressionModel(VectorRegressionModel, ABC):
    def __init__(self, predictedVariableNames: list):
        """
        :param predictedVariableNames: These are typically known at init time for rule-based models
        """
        super().__init__(checkInputColumns=False)
        self._predictedVariableNames = predictedVariableNames
        # guaranteed to be the same as predictedVariableNames since target transformers and output transformers are disallowed
        self._modelOutputVariableNames = predictedVariableNames

    def _underlyingModelRequiresFitting(self):
        return False

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        pass

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame = None, **kwargs):
        """
        Fits the model using the given data

        :param X: a data frame containing input data
        :param Y: a data frame containing output data or None. Preprocessors may require Y for fitting.
        :param kwargs: for consistency with VectorModel interface, will be ignored
        """
        super().fit(X, Y, fitPreprocessors=True)


class RuleBasedVectorClassificationModel(VectorClassificationModel, ABC):
    def __init__(self, labels: list, predictedVariableName="predictedLabel"):
        """
        :param labels:
        :param predictedVariableName:
        """
        super().__init__(checkInputColumns=False)

        duplicate = getFirstDuplicate(labels)
        if duplicate is not None:
            raise Exception(f"Found duplicate label: {duplicate}")
        self._labels = sorted(labels)
        self._predictedVariableNames = [predictedVariableName]

    def _underlyingModelRequiresFitting(self):
        return False

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        pass

    def _fitClassifier(self, X: pd.DataFrame, y: pd.DataFrame):
        pass

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame = None, **kwargs):
        """
        Fits the model using the given data

        :param X: a data frame containing input data
        :param Y: a data frame containing output data or None. Preprocessors may require Y for fitting.
        :param kwargs: for consistency with VectorModel interface, will be ignored
        """
        super().fit(X, Y, fitPreprocessors=True)
