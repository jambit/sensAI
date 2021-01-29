import copy
import logging
import warnings
from abc import ABC, abstractmethod
from typing import Tuple, Any, Generator, Generic, TypeVar, List

import numpy as np

from .eval_stats.eval_stats_base import PredictionEvalStats, EvalStatsCollection
from .eval_stats.eval_stats_classification import ClassificationEvalStats, ClassificationEvalStatsCollection
from .eval_stats.eval_stats_regression import RegressionEvalStats, RegressionEvalStatsCollection
from .evaluator import RegressionModelEvaluationData, ClassificationModelEvaluationData, \
    PredictorModelEvaluationData, VectorClassificationModelEvaluator, VectorRegressionModelEvaluator, \
    MetricsDictProvider
from ..data_ingest import InputOutputDataFrames
from ..util.typing import PandasNamedTuple
from ..vector_model import VectorClassificationModel, VectorRegressionModel, VectorModel, PredictorModel

log = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=PredictorModel)
TEvalStats = TypeVar("TEvalStats", bound=PredictionEvalStats)
TEvalStatsCollection = TypeVar("TEvalStatsCollection", bound=EvalStatsCollection)
TEvalData = TypeVar("TEvalData", bound=PredictorModelEvaluationData)


class PredictorModelCrossValidationData(ABC, Generic[TModel, TEvalData, TEvalStats, TEvalStatsCollection]):
    def __init__(self, predictorModels: List[TModel], evalDataList: List[TEvalData], predictedVarNames: List[str], testIndicesList=None):
        self.predictedVarNames = predictedVarNames
        self.predictorModels = predictorModels
        self.evalDataList = evalDataList
        self.testIndicesList = testIndicesList

    @property
    def trainedModels(self):
        warnings.warn("Use predictorModels instead, the field trainedModels will be removed in a future release",
                      DeprecationWarning)
        return self.predictorModels

    @property
    def modelName(self):
        return self.evalDataList[0].modelName

    @abstractmethod
    def _createEvalStatsCollection(self, l: List[TEvalStats]) -> TEvalStatsCollection:
        pass

    def getEvalStatsCollection(self, predictedVarName=None) -> TEvalStatsCollection:
        if predictedVarName is None:
            if len(self.predictedVarNames) != 1:
                raise Exception("Must provide name of predicted variable")
            else:
                predictedVarName = self.predictedVarNames[0]
        evalStatsList = [evalData.getEvalStats(predictedVarName) for evalData in self.evalDataList]
        return self._createEvalStatsCollection(evalStatsList)

    def iterInputOutputGroundTruthTuples(self, predictedVarName=None) -> Generator[Tuple[PandasNamedTuple, Any, Any], None, None]:
        for evalData in self.evalDataList:
            evalStats = evalData.getEvalStats(predictedVarName)
            for i, namedTuple in enumerate(evalData.inputData.itertuples()):
                yield namedTuple, evalStats.y_predicted[i], evalStats.y_true[i]


TCrossValData = TypeVar("TCrossValData", bound=PredictorModelCrossValidationData)


class VectorModelCrossValidator(MetricsDictProvider, Generic[TCrossValData], ABC):
    def __init__(self, data: InputOutputDataFrames, folds: int = 5, randomSeed=42, returnTrainedModels=False, evaluatorParams: dict = None):
        """
        :param data: the data set
        :param folds: the number of folds
        :param randomSeed: the random seed to use
        :param returnTrainedModels: whether to create a copy of the model for each fold and return each of the models
            (requires that models can be deep-copied); if False, the model that is passed to evalModel is fitted several times
        :param evaluatorParams: keyword parameters with which to instantiate model evaluators
        """
        self.returnTrainedModels = returnTrainedModels
        self.evaluatorParams = evaluatorParams if evaluatorParams is not None else {}
        numDataPoints = len(data)
        permutedIndices = np.random.RandomState(randomSeed).permutation(numDataPoints)
        numTestPoints = numDataPoints // folds
        self.modelEvaluators = []
        for i in range(folds):
            testStartIdx = i * numTestPoints
            testEndIdx = testStartIdx + numTestPoints
            testIndices = permutedIndices[testStartIdx:testEndIdx]
            trainIndices = np.concatenate((permutedIndices[:testStartIdx], permutedIndices[testEndIdx:]))
            self.modelEvaluators.append(self._createModelEvaluator(data.filterIndices(trainIndices), data.filterIndices(testIndices)))

    @abstractmethod
    def _createModelEvaluator(self, trainingData: InputOutputDataFrames, testData: InputOutputDataFrames):
        pass

    @abstractmethod
    def _createResultData(self, trainedModels, evalDataList, testIndicesList, predictedVarNames) -> TCrossValData:
        pass

    def evalModel(self, model: VectorModel):
        trainedModels = [] if self.returnTrainedModels else None
        evalDataList = []
        testIndicesList = []
        predictedVarNames = None
        for evaluator in self.modelEvaluators:
            modelToFit: VectorModel = copy.deepcopy(model) if self.returnTrainedModels else model
            evaluator.fitModel(modelToFit)
            if predictedVarNames is None:
                predictedVarNames = modelToFit.getPredictedVariableNames()
            if self.returnTrainedModels:
                trainedModels.append(modelToFit)
            evalDataList.append(evaluator.evalModel(modelToFit))
            testIndicesList.append(evaluator.testData.outputs.index)
        return self._createResultData(trainedModels, evalDataList, testIndicesList, predictedVarNames)

    def _computeMetrics(self, model: VectorModel):
        data = self.evalModel(model)
        return data.getEvalStatsCollection().aggStats()


class VectorRegressionModelCrossValidationData(PredictorModelCrossValidationData[VectorRegressionModel, RegressionModelEvaluationData, RegressionEvalStats, RegressionEvalStatsCollection]):
    def _createEvalStatsCollection(self, l: List[RegressionEvalStats]) -> RegressionEvalStatsCollection:
        return RegressionEvalStatsCollection(l)


class VectorRegressionModelCrossValidator(VectorModelCrossValidator[VectorRegressionModelCrossValidationData]):
    # TODO: after switching to python3.8 we can move both methods to the base class by accessing the generic type at runtime
    def _createModelEvaluator(self, trainingData: InputOutputDataFrames, testData: InputOutputDataFrames) -> VectorRegressionModelEvaluator:
        return VectorRegressionModelEvaluator(trainingData, testData=testData, **self.evaluatorParams)

    def _createResultData(self, trainedModels, evalDataList, testIndicesList, predictedVarNames) -> VectorRegressionModelCrossValidationData:
        return VectorRegressionModelCrossValidationData(trainedModels, evalDataList, predictedVarNames, testIndicesList)


class VectorClassificationModelCrossValidationData(PredictorModelCrossValidationData[VectorClassificationModel, ClassificationModelEvaluationData, ClassificationEvalStats, ClassificationEvalStatsCollection]):
    def _createEvalStatsCollection(self, l: List[ClassificationEvalStats]) -> ClassificationEvalStatsCollection:
        return ClassificationEvalStatsCollection(l)


class VectorClassificationModelCrossValidator(VectorModelCrossValidator[VectorClassificationModelCrossValidationData]):
    def _createModelEvaluator(self, trainingData: InputOutputDataFrames, testData: InputOutputDataFrames):
        return VectorClassificationModelEvaluator(trainingData, testData=testData, **self.evaluatorParams)

    def _createResultData(self, trainedModels, evalDataList, testIndicesList, predictedVarNames) -> VectorClassificationModelCrossValidationData:
        return VectorClassificationModelCrossValidationData(trainedModels, evalDataList, predictedVarNames, testIndicesList)
