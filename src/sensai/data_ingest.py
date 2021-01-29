from abc import ABC, abstractmethod
from typing import Tuple, Sequence, TypeVar

import numpy as np
import pandas as pd
import scipy.stats


class InputOutputData(ABC):
    def __init__(self, inputs, outputs):
        """
        :param inputs: expected to have shape and __len__
        :param outputs: expected to have shape and __len__
        """
        if len(inputs) != len(outputs):
            raise ValueError("Lengths do not match")
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    @abstractmethod
    def filterIndices(self, indices: Sequence[int]) -> __qualname__:
        pass


class InputOutputArrays(InputOutputData):
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        super().__init__(inputs, outputs)

    def filterIndices(self, indices: Sequence[int]) -> __qualname__:
        inputs = self.inputs[indices]
        outputs = self.outputs[indices]
        return InputOutputArrays(inputs, outputs)

    def toTorchDataLoader(self, batchSize=64, shuffle=True):
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError(f"Could not import torch, did you install it?")
        dataSet = TensorDataset(torch.tensor(self.inputs), torch.tensor(self.outputs))
        return DataLoader(dataSet, batch_size=batchSize, shuffle=shuffle)


class InputOutputDataFrames(InputOutputData):
    def __init__(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        super().__init__(inputs, outputs)

    def filterIndices(self, indices: Sequence[int]) -> __qualname__:
        inputs = self.inputs.iloc[indices]
        outputs = self.outputs.iloc[indices]
        return InputOutputDataFrames(inputs, outputs)

    @property
    def inputDim(self):
        return self.inputs.shape[1]

    @property
    def outputDim(self):
        return self.outputs.shape[1]

    def computeInputOutputCorrelation(self):
        correlations = {}
        for outputCol in self.outputs.columns:
            correlations[outputCol] = {}
            outputSeries = self.outputs[outputCol]
            for inputCol in self.inputs.columns:
                inputSeries = self.inputs[inputCol]
                pcc, pvalue = scipy.stats.pearsonr(inputSeries, outputSeries)
                correlations[outputCol][inputCol] = pcc
        return correlations


class DataSplitter(ABC):
    @abstractmethod
    def split(self, data: InputOutputDataFrames) -> Tuple[InputOutputDataFrames, InputOutputDataFrames]:
        pass


IOData = TypeVar("IOData", bound=InputOutputData)


class DataSplitterFractional(DataSplitter):
    def __init__(self, fractionalSizeOfFirstSet: float, shuffle=True, randomSeed=42):
        if not 0 <= fractionalSizeOfFirstSet <= 1:
            raise Exception(f"invalid fraction: {fractionalSizeOfFirstSet}")
        self.fractionalSizeOfFirstSet = fractionalSizeOfFirstSet
        self.shuffle = shuffle
        self.randomSeed = randomSeed

    def split(self, data: IOData) -> Tuple[IOData, IOData]:
        numDataPoints = len(data)
        splitIndex = int(numDataPoints * self.fractionalSizeOfFirstSet)
        rand = np.random.RandomState(self.randomSeed)
        if self.shuffle:
            indices = rand.permutation(numDataPoints)
        else:
            indices = range(numDataPoints)
        indicesA = indices[:splitIndex]
        indicesB = indices[splitIndex:]
        A = data.filterIndices(list(indicesA))
        B = data.filterIndices(list(indicesB))
        return A, B
