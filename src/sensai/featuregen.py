import logging
from abc import ABC, abstractmethod
from typing import Sequence, List, Union, Callable, Any, Dict, TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from . import util, data_transformation
from .util.string import orRegexGroup
from .columngen import ColumnGenerator

if TYPE_CHECKING:
    from .vector_model import VectorModel

_log = logging.getLogger(__name__)


class DuplicateColumnNamesException(Exception):
    pass


class FeatureGenerator(ABC):
    """
    Base class for feature generators that create a new DataFrame containing feature values
    from an input DataFrame
    """
    def __init__(self, categoricalFeatureNames: Sequence[str] = (),
                 normalisationRules: Sequence[data_transformation.DFTNormalisation.Rule] = (),
                 normalisationRuleTemplate: data_transformation.DFTNormalisation.RuleTemplate = None, addCategoricalDefaultRules=True):
        """
        :param categoricalFeatureNames: if provided, will ensure that the respective columns in the generated data frames will
            have dtype 'category'.
            Furthermore, presence of meta-information can later be leveraged for further transformations, e.g. one-hot encoding.
        :param normalisationRules: Rules to be used by DFTNormalisation (e.g. for constructing an input transformer for a model).
            These rules are only relevant if a DFTNormalisation object consuming them is instantiated and used
            within a data processing pipeline. They do not affect feature generation.
        :param normalisationRuleTemplate: This parameter can be supplied instead of normalisationRules for the case where
            there shall be a single rule that applies to all columns generated by this feature generator that were not labeled as categorical.
        :param addCategoricalDefaultRules:
            If True, normalisation rules for categorical features (which are unsupported by normalisation) and their corresponding one-hot
            encoded features (with "_<index>" appended) will be added.
        """
        if normalisationRules and normalisationRuleTemplate is not None:
            raise ValueError(f"normalisationRules should be empty when a normalisationRuleTemplate is provided")

        self._generatedColumnNames = None
        self._generatedColumnsRule = None
        self._normalisationRuleTemplate = normalisationRuleTemplate
        self._categoricalFeatureNames = categoricalFeatureNames
        self._categoricalFeatureRules = []
        self._normalisationRules = list(normalisationRules)
        if addCategoricalDefaultRules and len(categoricalFeatureNames) > 0:
            self._categoricalFeatureRules.append(data_transformation.DFTNormalisation.Rule(orRegexGroup(categoricalFeatureNames), unsupported=True))
            self._categoricalFeatureRules.append(data_transformation.DFTNormalisation.Rule(orRegexGroup(categoricalFeatureNames) + r"_\d+", skip=True))

    def getNormalisationRules(self) -> List[data_transformation.DFTNormalisation.Rule]:
        return self._normalisationRules + self._categoricalFeatureRules

    def getNormalisationRuleTemplate(self) -> Optional[data_transformation.DFTNormalisation.RuleTemplate]:
        return self._normalisationRuleTemplate

    def getCategoricalFeatureNames(self) -> Sequence[str]:
        return self._categoricalFeatureNames

    def getGeneratedColumnNames(self) -> Optional[List[str]]:
        """
        :return: Column names of the data frame generated by the most recent call of the feature generators 'generate' method.
            Returns None if generate was never called.
        """
        return self._generatedColumnNames

    @abstractmethod
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, ctx=None):
        """
        Fits the feature generator based on the given data

        :param X: the input/features data frame for the learning problem
        :param Y: the corresponding output data frame for the learning problem
            (which will typically contain regression or classification target columns)
        :param ctx: a context object whose functionality may be required for feature generation;
            this is typically the model instance that this feature generator is to generate inputs for
        """
        pass

    def generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        """
        Generates features for the data points in the given data frame

        :param df: the input data frame for which to generate features
        :param ctx: a context object whose functionality may be required for feature generation;
            this is typically the model instance that this feature generator is to generate inputs for
        :return: a data frame containing the generated features, which uses the same index as X (and Y)
        """
        resultDF = self._generate(df, ctx=ctx)

        isColumnDuplicatedArray = resultDF.columns.duplicated()
        if any(isColumnDuplicatedArray):
            duplicatedColumns = set(resultDF.columns[isColumnDuplicatedArray])
            raise DuplicateColumnNamesException(f"Feature data frame contains duplicate column names: {duplicatedColumns}")

        # ensure that categorical columns have dtype 'category'
        if len(self._categoricalFeatureNames) > 0:
            resultDF = resultDF.copy()  # resultDF we got might be a view of some other DF, so before we modify it, we must copy it
            for colName in self._categoricalFeatureNames:
                series = resultDF[colName].copy()
                if series.dtype.name != 'category':
                    resultDF[colName] = series.astype('category', copy=False)

        self._generatedColumnNames = resultDF.columns
        if self._normalisationRuleTemplate is not None:
            nonCategoricalFeatures = list(set(self._generatedColumnNames).difference(self._categoricalFeatureNames))
            self._normalisationRules = [self._normalisationRuleTemplate.toRule(orRegexGroup(nonCategoricalFeatures))]

        return resultDF

    @abstractmethod
    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        """
        Generates features for the data points in the given data frame.

        :param df: the input data frame for which to generate features
        :param ctx: a context object whose functionality may be required for feature generation;
            this is typically the model instance that this feature generator is to generate inputs for
        :return: a data frame containing the generated features, which uses the same index as X (and Y).
            The data frame's columns holding categorical columns are not required to have dtype 'category';
            this will be ensured by the encapsulating call.
        """
        pass

    def fitGenerate(self, X: pd.DataFrame, Y: pd.DataFrame, ctx=None) -> pd.DataFrame:
        """
        Fits the feature generator and subsequently generates features for the data points in the given data frame

        :param X: the input data frame for the learning problem and for which to generate features
        :param Y: the corresponding output data frame for the learning problem
            (which will typically contain regression or classification target columns)
        :param ctx: a context object whose functionality may be required for feature generation;
            this is typically the model instance that this feature generator is to generate inputs for
        :return: a data frame containing the generated features, which uses the same index as X (and Y)
        """
        self.fit(X, Y, ctx)
        return self.generate(X, ctx)


class RuleBasedFeatureGenerator(FeatureGenerator, ABC):
    """
    A feature generator which does not require fitting
    """
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, ctx=None):
        pass


class MultiFeatureGenerator(FeatureGenerator):
    """
    Wrapper for multiple feature generators. Calling generate here applies all given feature generators independently and
    returns the concatenation of their outputs
    """
    def __init__(self, featureGenerators: Sequence[FeatureGenerator]):
        """
        :param featureGenerators:
        """
        self.featureGenerators = featureGenerators
        categoricalFeatureNames = util.concatSequences([fg.getCategoricalFeatureNames() for fg in featureGenerators])
        normalisationRules = util.concatSequences([fg.getNormalisationRules() for fg in featureGenerators])
        super().__init__(categoricalFeatureNames=categoricalFeatureNames, normalisationRules=normalisationRules,
            addCategoricalDefaultRules=False)

    def _generateFromMultiple(self, generateFeatures: Callable[[FeatureGenerator], pd.DataFrame], index) -> pd.DataFrame:
        dfs = []
        for fg in self.featureGenerators:
            df = generateFeatures(fg)
            dfs.append(df)
        if len(dfs) == 0:
            return pd.DataFrame(index=index)
        else:
            return pd.concat(dfs, axis=1)

    def _generate(self, inputDF: pd.DataFrame, ctx=None):
        def generateFeatures(fg: FeatureGenerator):
            return fg.generate(inputDF, ctx=ctx)
        return self._generateFromMultiple(generateFeatures, inputDF.index)

    def fitGenerate(self, X: pd.DataFrame, Y: pd.DataFrame, ctx=None) -> pd.DataFrame:
        def generateFeatures(fg: FeatureGenerator):
            return fg.fitGenerate(X, Y, ctx)
        return self._generateFromMultiple(generateFeatures, X.index)

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, ctx=None):
        for fg in self.featureGenerators:
            fg.fit(X, Y)


class FeatureGeneratorFromNamedTuples(FeatureGenerator, ABC):
    """
    Generates feature values for one data point at a time, creating a dictionary with
    feature values from each named tuple
    """
    def __init__(self, cache: util.cache.PersistentKeyValueCache = None, categoricalFeatureNames: Sequence[str] = (),
                 normalisationRules: Sequence[data_transformation.DFTNormalisation.Rule] = (),
                 normalisationRuleTemplate: data_transformation.DFTNormalisation.RuleTemplate = None):
        super().__init__(categoricalFeatureNames=categoricalFeatureNames, normalisationRules=normalisationRules, normalisationRuleTemplate=normalisationRuleTemplate)
        self.cache = cache

    def _generate(self, df: pd.DataFrame, ctx=None):
        dicts = []
        for idx, nt in enumerate(df.itertuples()):
            if idx % 100 == 0:
                _log.debug(f"Generating feature via {self.__class__.__name__} for index {idx}")
            value = None
            if self.cache is not None:
                value = self.cache.get(nt.Index)
            if value is None:
                value = self._generateFeatureDict(nt)
                if self.cache is not None:
                    self.cache.set(nt.Index, value)
            dicts.append(value)
        return pd.DataFrame(dicts, index=df.index)

    @abstractmethod
    def _generateFeatureDict(self, namedTuple) -> Dict[str, Any]:
        """
        Creates a dictionary with feature values from a named tuple

        :param namedTuple: the data point for which to generate features
        :return: a dictionary mapping feature names to values
        """
        pass


class FeatureGeneratorTakeColumns(RuleBasedFeatureGenerator):
    def __init__(self, columns: Union[str, List[str]] = None, exceptColumns: Sequence[str] = (), categoricalFeatureNames: Sequence[str] = (),
                 normalisationRules: Sequence[data_transformation.DFTNormalisation.Rule] = (),
                 normalisationRuleTemplate: data_transformation.DFTNormalisation.RuleTemplate = None):
        """

        :param columns: name of the column or list of names of columns to be taken. If None, all columns will be taken.
        :param exceptColumns: list of names of columns to not take if present in the input df
        :param categoricalFeatureNames:
        :param normalisationRules:
        :param normalisationRuleTemplate:
        """
        super().__init__(categoricalFeatureNames=categoricalFeatureNames, normalisationRules=normalisationRules, normalisationRuleTemplate=normalisationRuleTemplate)
        if isinstance(columns, str):
            columns = [columns]
        self.columns = columns
        self.exceptColumns = exceptColumns

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        columnsToTake = self.columns if self.columns is not None else df.columns
        columnsToTake = [col for col in columnsToTake if col not in self.exceptColumns]

        missingCols = set(columnsToTake).difference(df.columns)
        if len(missingCols) > 0:
            raise Exception(f"Columns {missingCols} not present in data frame; available columns: {list(df.columns)}")

        return df[columnsToTake]


class FeatureGeneratorFlattenColumns(RuleBasedFeatureGenerator):
    """
    Instances of this class take columns with vectors and creates a data frame with columns containing entries of
    these vectors.

    For example, if columns "vec1", "vec2" contain vectors of dimensions dim1, dim2, a data frame with dim1+dim2 new columns
    will be created. It will contain the columns "vec1_<i1>", "vec2_<i2>" with i1, i2 ranging in (0, dim1), (0, dim2).

    """
    def __init__(self, columns: Optional[Union[str, Sequence[str]]] = None, categoricalFeatureNames: Sequence[str] = (),
                 normalisationRules: Sequence[data_transformation.DFTNormalisation.Rule] = (),
                 normalisationRuleTemplate: data_transformation.DFTNormalisation.RuleTemplate = None):
        """

        :param columns: name of the column or list of names of columns to be flattened. If None, all columns will be flattened.
        :param categoricalFeatureNames:
        :param normalisationRules:
        :param normalisationRuleTemplate:
        """
        super().__init__(categoricalFeatureNames=categoricalFeatureNames, normalisationRules=normalisationRules, normalisationRuleTemplate=normalisationRuleTemplate)
        if isinstance(columns, str):
            columns = [columns]
        self.columns = columns

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        resultDf = pd.DataFrame(index=df.index)
        columnsToFlatten = self.columns if self.columns is not None else df.columns
        for col in columnsToFlatten:
            _log.info(f"Flattening column {col}")
            values = np.stack(df[col].values)
            if len(values.shape) != 2:
                raise ValueError(f"Column {col} was expected to contain one dimensional vectors, something went wrong")
            dimension = values.shape[1]
            new_columns = [f"{col}_{i}" for i in range(dimension)]
            _log.info(f"Adding {len(new_columns)} new columns to feature dataframe")
            resultDf[new_columns] = pd.DataFrame(values, index=df.index)
        return resultDf


class FeatureGeneratorFromColumnGenerator(RuleBasedFeatureGenerator):
    """
    Implements a feature generator via a column generator
    """
    _log = _log.getChild(__qualname__)

    def __init__(self, columnGen: ColumnGenerator, takeInputColumnIfPresent=False, isCategorical=False,
            normalisationRuleTemplate: data_transformation.DFTNormalisation.RuleTemplate = None):
        """
        :param columnGen: the underlying column generator
        :param takeInputColumnIfPresent: if True, then if a column whose name corresponds to the column to generate exists
            in the input data, simply copy it to generate the output (without using the column generator); if False, always
            apply the columnGen to generate the output
        :param isCategorical: whether the resulting column is categorical
        :param normalisationRuleTemplate: template for a DFTNormalisation for the resulting column.
            This should only be provided if isCategorical is False
        """
        if isCategorical and normalisationRuleTemplate is not None:
            raise ValueError(f"normalisationRuleTemplate should be None when the generated column is categorical")

        categoricalFeatureNames = (columnGen.generatedColumnName, ) if isCategorical else ()
        super().__init__(categoricalFeatureNames=categoricalFeatureNames, normalisationRuleTemplate=normalisationRuleTemplate)

        self.takeInputColumnIfPresent = takeInputColumnIfPresent
        self.columnGen = columnGen

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        colName = self.columnGen.generatedColumnName
        if self.takeInputColumnIfPresent and colName in df.columns:
            self._log.debug(f"Taking column '{colName}' from input data frame")
            series = df[colName]
        else:
            self._log.debug(f"Generating column '{colName}' via {self.columnGen}")
            series = self.columnGen.generateColumn(df)
        return pd.DataFrame({colName: series})


class ChainedFeatureGenerator(FeatureGenerator):
    """
    Chains feature generators such that they are executed one after another. The output of generator i>=1 is the input of
    generator i+1 in the generator sequence.
    """
    def __init__(self, *featureGenerators: FeatureGenerator, categoricalFeatureNames: Sequence[str] = None,
                 normalisationRules: Sequence[data_transformation.DFTNormalisation.Rule] = None,
                 normalisationRuleTemplate: data_transformation.DFTNormalisation.RuleTemplate = None):
        """
        :param featureGenerators: the list of feature generators to apply in order
        :param categoricalFeatureNames: the list of categorical feature names being generated; if None, use the ones
            indicated by the last feature generator in the list
        :param normalisationRules: normalisation rules to use; if None, use rules of the last feature generator in the list
        :param normalisationRuleTemplate: rule template to apply to all output columns of the chain.
            This should only be used if no other rules have been provided
        """
        if len(featureGenerators) == 0:
            raise ValueError("Empty list of feature generators")
        if categoricalFeatureNames is None:
            categoricalFeatureNames = featureGenerators[-1].getCategoricalFeatureNames()
        if normalisationRules is None:
            normalisationRules = featureGenerators[-1].getNormalisationRules()
        super().__init__(categoricalFeatureNames=categoricalFeatureNames, normalisationRules=normalisationRules, normalisationRuleTemplate=normalisationRuleTemplate)
        self.featureGenerators = featureGenerators

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        for featureGen in self.featureGenerators:
            df = featureGen.generate(df, ctx)
        return df

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, ctx=None):
        self.fitGenerate(X, Y, ctx)

    def fitGenerate(self, X: pd.DataFrame, Y: pd.DataFrame, ctx=None) -> pd.DataFrame:
        for fg in self.featureGenerators:
            X = fg.fitGenerate(X, Y, ctx)
        return X


class FeatureGeneratorTargetDistribution(FeatureGenerator):
    """
    A feature generator, which, for a column T (typically the categorical target column of a classification problem
    or the continuous target column of a regression problem),

    * can ensure that T takes on limited set of values t_1, ..., t_n by allowing the user to apply
      binning using given bin boundaries
    * computes for each value c of a categorical column C the conditional empirical distribution
      P(T | C=c) in the training data during the training phase,
    * generates, for each requested column C and value c in the column, n features
      '<C>_<T>_distribution_<t_i>' = P(T=t_i | C=c) if flatten=True
      or one feature '<C>_<T>_distribution' = [P(T=t_i | C=c), ..., P(T=t_n | C=c)] if flatten=False

    Being probability values, the features generated by this feature generator are already normalised.
    """
    def __init__(self, columns: Union[str, Sequence[str]], targetColumn: str,
            targetColumnBins: Optional[Union[Sequence[float], int, pd.IntervalIndex]], targetColumnInFeaturesDf=False,
            flatten=True):
        """
        :param columns: the categorical columns for which to generate distribution features
        :param targetColumn: the column the distributions over which will make up the features.
            If targetColumnBins is not None, this column will be discretised before computing the conditional distributions
        :param targetColumnBins: if not None, specifies the binning to apply via pandas.cut
            (see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html).
            Note that if a value should match no bin, NaN will generated. To avoid this when specifying bin boundaries in a list,
            -inf and +inf should be used as the first and last entries.
        :param targetColumnInFeaturesDf: if True, when fitting will look for targetColumn in the features data frame (X) instead of in target data frame (Y)
        :param flatten: whether to generate a separate scalar feature per distribution value rather than one feature
            with all of the distribution's values
        """
        self.flatten = flatten
        if isinstance(columns, str):
            columns = [columns]
        self.columns = columns
        self.targetColumn = targetColumn
        self.targetColumnInInputDf = targetColumnInFeaturesDf
        self.targetColumnBins = targetColumnBins
        if self.flatten:
            normalisationRuleTemplate = data_transformation.DFTNormalisation.RuleTemplate(skip=True)
        else:
            normalisationRuleTemplate = data_transformation.DFTNormalisation.RuleTemplate(unsupported=True)
        super().__init__(normalisationRuleTemplate=normalisationRuleTemplate)
        self._targetColumnValues = None
        # This will hold the mapping: column -> featureValue -> targetValue -> targetValueEmpiricalProbability
        self._discreteTargetDistributionsByColumn: Dict[str, Dict[Any, Dict[Any, float]]] = None

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, ctx=None):
        """
        This will persist the empirical target probability distributions for all unique values in the specified columns
        """
        if self.targetColumnInInputDf:
            target = X[self.targetColumn]
        else:
            target = Y[self.targetColumn]
        if self.targetColumnBins is not None:
            discretisedTarget = pd.cut(target, self.targetColumnBins)
        else:
            discretisedTarget = target
        self._targetColumnValues = discretisedTarget.unique()

        self._discreteTargetDistributionsByColumn = {}
        for column in self.columns:
            self._discreteTargetDistributionsByColumn[column] = {}
            columnTargetDf = pd.DataFrame()
            columnTargetDf[column] = X[column]
            columnTargetDf["target"] = discretisedTarget.values
            for value, valueTargetsDf in columnTargetDf.groupby(column):
                # The normalized value_counts contain targetValue -> targetValueEmpiricalProbability for the current value
                self._discreteTargetDistributionsByColumn[column][value] = valueTargetsDf["target"].value_counts(normalize=True).to_dict()

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        if self._discreteTargetDistributionsByColumn is None:
            raise Exception("Feature generator has not been fitted")
        resultDf = pd.DataFrame(index=df.index)
        for column in self.columns:
            targetDistributionByValue = self._discreteTargetDistributionsByColumn[column]
            if self.flatten:
                for targetValue in self._targetColumnValues:
                    # Important: pd.Series.apply should not be used here, as it would label the resulting column as categorical
                    resultDf[f"{column}_{self.targetColumn}_distribution_{targetValue}"] = [targetDistributionByValue[value].get(targetValue, 0.0) for value in df[column]]
            else:
                distributions = [[targetDistributionByValue[value].get(targetValue, 0.0) for targetValue in self._targetColumnValues] for value in df[column]]
                resultDf[f"{column}_{self.targetColumn}_distribution"] = pd.Series(distributions, index=df[column].index)
        return resultDf


################################
#
# generator registry
#
################################


class FeatureGeneratorRegistry:
    """
    Represents a registry for named feature generators which can be instantiated via factories.
    Each named feature generator is a singleton, i.e. each factory will be called at most once.

    Feature generators can be registered and retrieved by \n
    registry.<name> = <featureGeneratorFactory> \n
    registry.<name> \n
    or through the corresponding methods

    Example:
        >>> from sensai.featuregen import FeatureGeneratorRegistry, FeatureGeneratorTakeColumns
        >>> import pandas as pd

        >>> df = pd.DataFrame({"foo": [1, 2, 3], "bar": [7, 8, 9]})
        >>> registry = FeatureGeneratorRegistry()
        >>> registry.testFgen = lambda: FeatureGeneratorTakeColumns("foo")
        >>> registry.testFgen().generate(df)
           foo
        0    1
        1    2
        2    3
    """
    def __init__(self):
        # Important: Don't set public members in init. Since we override setattr this would lead to undesired consequences
        self._featureGeneratorFactories = {}
        self._featureGeneratorSingletons = {}

    def __setattr__(self, name: str, value):
        if not name.startswith("_"):
            self.registerFactory(name, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, item: str):
        if item.startswith("_"):
            raise ValueError(f"Access to private variables in {self.__class__.__name__} is forbidden")
        return self.getFeatureGenerator(item)

    def registerFactory(self, name, factory: Callable[[], FeatureGenerator]):
        """
        Registers a feature generator factory which can subsequently be referenced by models via their name
        :param name: the name
        :param factory: the factory
        """
        if name in self._featureGeneratorFactories:
            raise ValueError(f"Generator for name '{name}' already registered")
        super().__setattr__(name, factory)
        self._featureGeneratorFactories[name] = factory

    def getFeatureGenerator(self, name):
        """
        Creates a feature generator from a name, which must

        :param name: the name of the generator
        :return: a new feature generator instance
        """
        generator = self._featureGeneratorSingletons.get(name)
        if generator is None:
            factory = self._featureGeneratorFactories.get(name)
            if factory is None:
                raise ValueError(f"No factory registered for name '{name}': known names: {list(self._featureGeneratorFactories.keys())}. Use registerFeatureGeneratorFactory to register a new feature generator factory.")
            generator = factory()
            self._featureGeneratorSingletons[name] = generator
        return generator


class FeatureCollector(object):
    """
    A feature collector which can provide a multi-feature generator from a list of names/generators and registry
    """

    def __init__(self, *featureGeneratorsOrNames: Union[str, FeatureGenerator], registry=None):
        """
        :param featureGeneratorsOrNames: generator names (known to articleFeatureGeneratorRegistry) or generator instances.
        :param registry: the feature generator registry for the case where names are passed
        """
        self._featureGeneratorsOrNames = featureGeneratorsOrNames
        self._registry = registry
        self._multiFeatureGenerator = self._createMultiFeatureGenerator()

    def getMultiFeatureGenerator(self) -> MultiFeatureGenerator:
        return self._multiFeatureGenerator

    def _createMultiFeatureGenerator(self):
        featureGenerators = []
        for f in self._featureGeneratorsOrNames:
            if isinstance(f, FeatureGenerator):
                featureGenerators.append(f)
            elif type(f) == str:
                if self._registry is None:
                    raise Exception(f"Received feature name '{f}' instead of instance but no registry to perform the lookup")
                featureGenerators.append(self._registry.getFeatureGenerator(f))
            else:
                raise ValueError(f"Unexpected type {type(f)} in list of features")
        return MultiFeatureGenerator(featureGenerators)


class FeatureGeneratorFromVectorModel(FeatureGenerator):
    def __init__(self, vectorModel: "VectorModel", targetFeatureGenerator: FeatureGenerator, categoricalFeatureNames: Sequence[str] = (),
                 normalisationRules: Sequence[data_transformation.DFTNormalisation.Rule] = (),
                 normalisationRuleTemplate: data_transformation.DFTNormalisation.RuleTemplate = None,
                 inputFeatureGenerator: FeatureGenerator = None, useTargetFeatureGeneratorForTraining=False):
        """
        Provides a feature via predictions of a given model

        :param vectorModel: model used for generate features from predictions
        :param targetFeatureGenerator: generator for target to be predicted
        :param categoricalFeatureNames:
        :param normalisationRules:
        :param normalisationRuleTemplate:
        :param inputFeatureGenerator: optional feature generator to be applied to input of vectorModel's fit and predict
        :param useTargetFeatureGeneratorForTraining: if False, this generator will always apply the model
            to generate features.
            If True, this generator will use targetFeatureGenerator to generate features, bypassing the
            model. This is useful for the case where the model which is
            to receive the generated features shall be trained on the original targets rather than the predictions
            thereof.
        """
        super().__init__(categoricalFeatureNames=categoricalFeatureNames, normalisationRules=normalisationRules, normalisationRuleTemplate=normalisationRuleTemplate)

        self.useTargetFeatureGeneratorForTraining = useTargetFeatureGeneratorForTraining
        self.targetFeatureGenerator = targetFeatureGenerator
        self.inputFeatureGenerator = inputFeatureGenerator
        self.useTargetFeatureGeneratorForTraining = useTargetFeatureGeneratorForTraining
        self.vectorModel = vectorModel

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, ctx=None):
        targetDF = self.targetFeatureGenerator.fitGenerate(X, Y)
        if self.inputFeatureGenerator:
            X = self.inputFeatureGenerator.fitGenerate(X, Y)
        self.vectorModel.fit(X, targetDF)

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        if self.inputFeatureGenerator:
            df = self.inputFeatureGenerator.generate(df)
        if self.useTargetFeatureGeneratorForTraining and not ctx.isFitted():
            _log.info(f"Using targetFeatureGenerator {self.targetFeatureGenerator.__class__.__name__} to generate target features")
            return self.targetFeatureGenerator.generate(df)
        else:
            _log.info(f"Generating target features via {self.vectorModel.__class__.__name__}")
            return self.vectorModel.predict(df)


def flattenedFeatureGenerator(fgen: FeatureGenerator, columnsToFlatten: List[str] = None,
                            normalisationRules=None, normalisationRuleTemplate: data_transformation.DFTNormalisation.RuleTemplate = None):
    """
    Return a flattening version of the input feature generator.

    :param fgen: feature generator to be flattened
    :param columnsToFlatten: list of names of output columns to be flattened.
        If None, all output columns will be flattened.
    :param normalisationRules: additional normalisation rules for the flattened output columns
    :param normalisationRuleTemplate: This parameter can be supplied instead of normalisationRules for the case where
        there shall be a single rule that applies to all flattened output columns
    :return: FeatureGenerator instance that will generate flattened versions of the specified output columns and leave
        all non specified output columns as is.

    Example:
        >>> from sensai.featuregen import FeatureGeneratorTakeColumns, flattenedFeatureGenerator
        >>> import pandas as pd
        >>>
        >>> df = pd.DataFrame({"foo": [[1, 2], [3, 4]], "bar": ["a", "b"]})
        >>> fgen = flattenedFeatureGenerator(FeatureGeneratorTakeColumns(), columnsToFlatten=["foo"])
        >>> fgen.generate(df)
           foo_0  foo_1 bar
        0      1      2   a
        1      3      4   b
    """

    flatteningGenerator = ChainedFeatureGenerator(fgen, FeatureGeneratorTakeColumns(columnsToFlatten), FeatureGeneratorFlattenColumns(),
                                                  normalisationRules=normalisationRules, normalisationRuleTemplate=normalisationRuleTemplate)
    exceptColumns = columnsToFlatten if columnsToFlatten is not None else []
    passthroughColumnsGenerator = ChainedFeatureGenerator(fgen, FeatureGeneratorTakeColumns(exceptColumns=exceptColumns))

    return MultiFeatureGenerator([flatteningGenerator, passthroughColumnsGenerator])
