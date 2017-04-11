from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import time
import itertools
from collections import OrderedDict, namedtuple, defaultdict

import util
import dragonn.synthetic.fileProcessing as fp

coreDeepLIFTtrackName = 'coreDeepLIFTtrack'


def identifyPeaks(arr):
    # use a state machine to identify peaks
    # "peaks" as defined by larger than neighbours
    # for tied region, take the middle of the tie.
    # return tuples of idx + peak val
    previousVal = None
    potentialPeakStartIdx = None
    foundPeaks = []
    for idx, val in enumerate(arr):
        if (previousVal is not None):
            if (val > previousVal):
                potentialPeakStartIdx = idx
            elif (val < previousVal):
                if (potentialPeakStartIdx is not None):
                    # peak found!
                    foundPeaks.append(
                        (int(0.5 * (potentialPeakStartIdx + (idx - 1))), previousVal))
                potentialPeakStartIdx = None
                potentialPeakStartVal = None
            else:
                # tie...don't change anything.
                pass
        previousVal = val
    return foundPeaks


def dnaRevCompFunc(arr):
    assert len(arr.shape) == 2, arr.shape
    assert arr.shape[0] == 4
    return arr[::-1, ::-1]


def reverseFunc(arr):
    return arr[:, ::-1]


def identity(arr):
    return arr


class RevCompWithDNArowsSubset(object):
    __name__ = "RevCompWithDNArowsSubset"  # for json saving...

    def __init__(self, dnaRowsStart, dnaRowsEnd):
        self.dnaRowsStart = dnaRowsStart
        self.dnaRowsEnd = dnaRowsEnd

    def __call__(self, arr):
        toReturn = np.zeros(arr.shape)
        assert self.dnaRowsEnd - self.dnaRowsStart == 4
        assert arr.shape[0] >= self.dnaRowsEnd, arr.shape
        if (self.dnaRowsStart > 0):
            toReturn[:self.dnaRowsStart, :] = arr[:self.dnaRowsStart, ::-1]
        toReturn[self.dnaRowsStart:self.dnaRowsEnd, :] =\
            dnaRevCompFunc(arr[self.dnaRowsStart:self.dnaRowsEnd])
        if (arr.shape[0] > self.dnaRowsEnd):
            toReturn[self.dnaRowsEnd:, :] = arr[self.dnaRowsEnd, ::-1]
        return toReturn


class DataTrack(object):
    __slots__ = ["data", "revCompData", "pseudocount", "effectiveStride",
                 "effectiveWidth", "layerFromAbove", "fillValue", "minVisibility"]
    JsonKeys = util.enum(data="data", revCompData="revCompData", pseudocount="pseudocount", effectiveStride="effectiveStride",
                         effectiveWidth="effectiveWidth", layerFromAbove="layerFromAbove", fillValue="fillValue", minVisibility="minVisibility")

    def __init__(self, data, revCompData, pseudocount, effectiveStride, effectiveWidth, layerFromAbove, fillValue, minVisibility):
        """
            effectiveStride and effectiveWidth will be 1 for most purposes...they are only
                relevant when your Grammar object has augmented data tracks that are of
                different lengths - eg: the DeepLIFT scores of a convolutional layer
                and the sequence underlying the convolutional layer. In that case,
                assuming that the grammar object was created on the convolutional layer
                deepLIFT scores, then the effective stride and effective width of
                the sequence track must be set to the stride and width of the conv layer.
        """
        self.data = data
        self.revCompData = revCompData
        self.pseudocount = pseudocount
        self.effectiveStride = effectiveStride
        self.effectiveWidth = effectiveWidth
        self.layerFromAbove = layerFromAbove
        self.fillValue = fillValue
        self.minVisibility = minVisibility

    def reverseComplement(self):
        return DataTrack(
            data=self.revCompData, revCompData=self.data, pseudocount=self.pseudocount, effectiveStride=self.effectiveStride, effectiveWidth=self.effectiveWidth, layerFromAbove=self.layerFromAbove, fillValue=self.fillValue, minVisibility=self.minVisibility)

    def getJsonableObject(self):
        return {DataTrack.JsonKeys.data: self.data.tolist(), DataTrack.JsonKeys.revCompData: self.revCompData.tolist(), DataTrack.JsonKeys.pseudocount: self.pseudocount, DataTrack.JsonKeys.effectiveStride: self.effectiveStride, DataTrack.JsonKeys.effectiveWidth: self.effectiveWidth, DataTrack.JsonKeys.layerFromAbove: self.layerFromAbove, DataTrack.JsonKeys.fillValue: self.fillValue, DataTrack.JsonKeys.minVisibility: self.minVisibility}

    @classmethod
    def loadFromJsonableObject(cls, jsonableObject):
        return DataTrack(data=np.array(jsonableObject[DataTrack.JsonKeys.data]), revCompData=np.array(jsonableObject[DataTrack.JsonKeys.revCompData]), pseudocount=jsonableObject[DataTrack.JsonKeys.pseudocount], effectiveStride=jsonableObject[DataTrack.JsonKeys.effectiveStride], effectiveWidth=jsonableObject[DataTrack.JsonKeys.effectiveWidth], layerFromAbove=jsonableObject[DataTrack.JsonKeys.layerFromAbove], fillValue=jsonableObject[DataTrack.JsonKeys.fillValue], minVisibility=jsonableObject[DataTrack.JsonKeys.minVisibility])


class Grammar(object):
    coreDeepLIFTtrackName = coreDeepLIFTtrackName
    JsonKeys = util.enum(numUnderlyingObservations="numUnderlyingObservations", totalObservationsEver="totalObservationsEver",
                         summedDataTracks="summedDataTracks", minPseudocount="minPseudocount", pseudocountFrac="pseudocountFrac", derivedClass="derivedClass")

    def __init__(self, numUnderlyingObservations, totalObservationsEver, summedDataTracks, minPseudocount=0, pseudocountFrac=0.1):
        if (isinstance(numUnderlyingObservations, int) or isinstance(numUnderlyingObservations, float)):
            self.numUnderlyingObservations =\
                np.zeros((summedDataTracks[Grammar.coreDeepLIFTtrackName].data.shape[1],), dtype="float")\
                + numUnderlyingObservations
        else:
            self.numUnderlyingObservations = util.npArrayIfList(
                numUnderlyingObservations)
        self.totalObservationsEver = totalObservationsEver
        self.minPseudocount = minPseudocount
        self.pseudocountFrac = pseudocountFrac
        self.pseudocountMultiplier = np.maximum(self.minPseudocount,
                                                np.floor(self.pseudocountFrac * np.max(self.numUnderlyingObservations)))
        for dataTrack in summedDataTracks.values():
            assert util.assertIsType(dataTrack, DataTrack, "dataTrack")
        self.summedDataTracks = {}
        self.normalisedDataTracks = {}
        self.revCompedNormalisedDataTracks = {}
        for key, dataTrack in summedDataTracks.items():
            self.addSummedDataTrack(key, dataTrack)

    def getJsonableObject(self):
        theClass = self.__class__.__name__
        return {Grammar.JsonKeys.numUnderlyingObservations: self.numUnderlyingObservations.tolist(), Grammar.JsonKeys.totalObservationsEver: self.totalObservationsEver, Grammar.JsonKeys.summedDataTracks: OrderedDict(
            [(key, self.summedDataTracks[key].getJsonableObject())
             for key in self.summedDataTracks]), Grammar.JsonKeys.minPseudocount: self.minPseudocount, Grammar.JsonKeys.pseudocountFrac: self.pseudocountFrac, Grammar.JsonKeys.derivedClass: theClass}

    @staticmethod
    def saveListOfGrammarsToJson(jsonFileName, listOfGrammars):
        import json
        jsonifiedGrammars = [x.getJsonableObject() for x in listOfGrammars]
        fp.writeToFile(jsonFileName, json.dumps(jsonifiedGrammars))

    @staticmethod
    def loadListOfGrammarsFromJson(jsonFileName):
        jsonifiedGrammars = util.parseJsonFile(jsonFileName)
        return [Grammar.loadSingleGrammarOfArbitraryClass(x) for x in jsonifiedGrammars]

    @staticmethod
    def loadSingleGrammarOfArbitraryClass(jsonableObject):
        """
            Will figure out the appropriate subclass to call for loading
        """
        theClass = eval(jsonableObject[Grammar.JsonKeys.derivedClass])
        return theClass.loadFromJsonableObject(jsonableObject)

    @classmethod
    def loadFromJsonableObject(cls, jsonableObject):
        return cls(**cls.getLoadingKwargsFromJsonableObject(jsonableObject))

    @classmethod
    def getLoadingKwargsFromJsonableObject(cls, jsonableObject):
        summedDataTracksJsonable = jsonableObject[
            Grammar.JsonKeys.summedDataTracks]
        summedDataTracks = OrderedDict([
            (key, DataTrack.loadFromJsonableObject(
                summedDataTracksJsonable[key]))
            for key in summedDataTracksJsonable.keys()])
        return {Grammar.JsonKeys.numUnderlyingObservations:
                jsonableObject[Grammar.JsonKeys.numUnderlyingObservations], Grammar.JsonKeys.totalObservationsEver:
                jsonableObject[Grammar.JsonKeys.totalObservationsEver], Grammar.JsonKeys.summedDataTracks: summedDataTracks, Grammar.JsonKeys.minPseudocount:
                jsonableObject[Grammar.JsonKeys.minPseudocount], Grammar.JsonKeys.pseudocountFrac:
                jsonableObject[Grammar.JsonKeys.pseudocountFrac]}

    def getRevCompGrammar(self):
        grammar = Grammar(
            numUnderlyingObservations=self.numUnderlyingObservations[::-1], totalObservationsEver=self.totalObservationsEver, summedDataTracks={}, minPseudocount=0, pseudocountFrac=0.1)
        grammar.summedDataTracks =\
            OrderedDict([(key, val.reverseComplement()) for (key, val) in
                         self.summedDataTracks.items()])
        grammar.normalisedDataTracks = self.revCompedNormalisedDataTracks
        grammar.revCompedNormalisedDataTracks = self.normalisedDataTracks
        return grammar

    def getRange(self, start, end):
        grammar = Grammar(
            numUnderlyingObservations=self.numUnderlyingObservations[start:end], totalObservationsEver=self.totalObservationsEver, summedDataTracks={key:
                                                                                                                                                     DataTrack(
                                                                                                                                                         data=dataTrack.data[:, start:end], revCompData=dataTrack.data[:,
                                                                                                                                                                                                                       (dataTrack.data.shape[1] - end):(dataTrack.data.shape[1] - start)], pseudocount=dataTrack.pseudocount, effectiveStride=dataTrack.effectiveStride, effectiveWidth=dataTrack.effectiveWidth)
                                                                                                                                                     for (key, dataTrack) in self.summedDataTracks.items()}, minPseudocount=0, pseudocountFrac=0.1)
        return grammar

    @property
    def grammarArray(self):
        print(".grammarArray is deprecated; "
              "use normedCoreDeepLIFTtrack instead")
        return self.normedCoreDeepLIFTtrack

    @property
    def normedCoreDeepLIFTtrack(self):
        return self.getNormalisedDataTrack(self.coreDeepLIFTtrackName)

    @property
    def summedGrammar(self):
        print(".summedGrammar is deprecated; "
              "use summedCoreDeepLIFTtrack instead")
        return self.summedCoreDeepLIFTtrack

    @property
    def summedCoreDeepLIFTtrack(self):
        return self.getSummedDataTrack(self.coreDeepLIFTtrackName)

    def getNormalisedDataTrack(self, key):
        return self.normalisedDataTracks[key]

    def getSummedDataTrack(self, key):
        return self.summedDataTracks[key].data

    def getRevCompedNormalisedDataTrack(self, key):
        return self.revCompedNormalisedDataTracks[key]

    def getRevCompedSummedDataTrack(self, key):
        return self.summedDataTracks[key].revCompData

    @staticmethod
    def transformNumObsAccordingToWidthAndStride(numObsArr, effectiveStride, effectiveWidth, layerFromAbove, minVisibility):
        # I believe layerFromAbove is a boolean indicating whether
        # this layer lay above or below the layer that was used to
        # initialize the seqlets. WffectiveStride/width will be
        # used differently, accordingly...
        # ugh this whole thing needs to be refactored to make it less kludgy
        if (layerFromAbove == False):
            # repeats entries of array, spreading
            # them out according to effectiveStride/effectiveWidth
            newObsArrLen = (len(numObsArr) - 1) * \
                effectiveStride + effectiveWidth
            newObsArr = np.ones(newObsArrLen) * -1
            for (idx, numObs) in enumerate(numObsArr):
                startIdx = idx * effectiveStride
                endIdx = startIdx + effectiveWidth
                maximums = np.maximum(numObs, newObsArr[startIdx:endIdx])
                newObsArr[startIdx:endIdx] = maximums
        else:
            assert effectiveStride == 1  # only handling stride of 1 for now
            positionsWithFullVisibility = len(numObsArr) - (effectiveWidth - 1)
            positionsWithPartialVisibility = 2 * \
                (effectiveWidth - minVisibility)
            newObsArrLen = positionsWithFullVisibility +\
                positionsWithPartialVisibility
            newObsArr = np.ones(newObsArrLen) * -1
            for (idx, numObs) in enumerate(numObsArr):
                # find the indexes in conv layer such that the neuron in
                # the conv layer sees idx and also satisfies the
                # minVisibility constraint assuming idx is the left
                # boundary of a region or the right boundary of a region
                convIdx_assumeIdxIsLeftBoundary = idx
                convIdx_assumeIdxIsRightBoundary =\
                    idx + (1 + effectiveWidth - 2 * (minVisibility))
                # the +1 for endIdx is in order to satisfy array slicing
                startIdx, endIdx = min(convIdx_assumeIdxIsLeftBoundary,
                                       convIdx_assumeIdxIsRightBoundary),\
                    max(convIdx_assumeIdxIsRightBoundary,
                        convIdx_assumeIdxIsLeftBoundary) + 1
                maximums = np.maximum(numObs, newObsArr[startIdx:endIdx])
                newObsArr[startIdx:endIdx] = maximums
        assert all([x != -1 for x in newObsArr]), newObsArr
        assert len(newObsArr) == newObsArrLen
        return newObsArr

    def addSummedDataTrack(self, key, dataTrack):
        #assert key not in self.summedDataTracks\
        #    , key+" already in summedDataTracks"
        self.summedDataTracks[key] = dataTrack
        transformedNumObs = (
            self.transformNumObsAccordingToWidthAndStride(
                self.numUnderlyingObservations, effectiveStride=dataTrack.effectiveStride, effectiveWidth=dataTrack.effectiveWidth, layerFromAbove=dataTrack.layerFromAbove, minVisibility=dataTrack.minVisibility)[None, :])
        assert transformedNumObs.shape[1] == dataTrack.data.shape[1]\
            , key + " " + str(dataTrack.data.shape) +\
            " " + str(transformedNumObs.shape) + " "\
            + str(self.numUnderlyingObservations.shape) + " "\
            + str(dataTrack.effectiveStride)\
            + " " + str(dataTrack.effectiveWidth)
        if (dataTrack.pseudocount is not None):
            (self.normalisedDataTracks[key],
             self.revCompedNormalisedDataTracks[key]) =\
                [(data + self.pseudocountMultiplier * dataTrack.pseudocount)
                 / (numObs + self.pseudocountMultiplier)
                 for data, numObs in
                 [(dataTrack.data, transformedNumObs),
                     (dataTrack.revCompData, transformedNumObs[:, ::-1])]]
        else:
            (self.normalisedDataTracks[key],
             self.revCompedNormalisedDataTracks[key]) =\
                [(data) / transformedNumObs for data in
                 [dataTrack.data, dataTrack.revCompData]]

    def merge(self, otherGrammar, subtracksToInclude, subtrackNormaliseFunc, normaliseFunc, smallerPerPosNormFuncs, largerPerPosNormFuncs, revComp):
        """
            subtracksToInclude: subtracks to use for finding optimal
                                alignment
            subtracksToInclude, subtrackNormaliseFunc, normaliseFunc
                smallerPerPosNormFuncs, largerPerPosNormFuncs, revComp:
                    you should make these the same as the arguments
                    passed to getCorrelationMatrix
            revComp: boolean indicating whether to consider the
                reverse complement as well
        """
        # selfTransformed, otherTransfored are basically the numpy array
        # that will be used for cross correlation (obtained from the
        # appropriate subtracks), with any normalization applied
        selfTransformed, otherTransformed =\
            [getArrayForCrossCorrFromGrammar(
                grammar=grammar, subtracksToInclude=subtracksToInclude, subtrackNormaliseFunc=subtrackNormaliseFunc, useSummed=False, revComp=False)
             for grammar in [self, otherGrammar]]
        if (revComp):
            otherTransformedRevComp = getArrayForCrossCorrFromGrammar(
                grammar=otherGrammar, subtracksToInclude=subtracksToInclude, subtrackNormaliseFunc=subtrackNormaliseFunc, useSummed=False, revComp=True)

        # find the alignment with the optimal overlap
        bestCorrelation, shift, firstIsSmaller =\
            util.getBestLengthwiseCrossCorrelationOfArrays(
                selfTransformed, otherTransformed, normaliseFunc=normaliseFunc, smallerPerPosNormFuncs=smallerPerPosNormFuncs, largerPerPosNormFuncs=largerPerPosNormFuncs)
        useRevComp = False
        if (revComp):
            bestCorrelationRevComp, shiftRevComp, firstIsSmallerRevComp =\
                util.getBestLengthwiseCrossCorrelationOfArrays(
                    selfTransformed, otherTransformedRevComp, normaliseFunc=normaliseFunc, smallerPerPosNormFuncs=smallerPerPosNormFuncs, largerPerPosNormFuncs=largerPerPosNormFuncs)
            assert firstIsSmallerRevComp == firstIsSmaller
            if (bestCorrelationRevComp > bestCorrelation):
                useRevComp = True
                shift = shiftRevComp
                otherGrammar = otherGrammar.getRevCompGrammar()
        if (firstIsSmaller):
            smaller = self
            larger = otherGrammar
        else:
            smaller = otherGrammar
            larger = self
        # having found the optimal alignment, do merging
        return self.mergeArraysTogether(smaller=smaller, larger=larger, shift=shift)

    @staticmethod
    def obtainLeftPadRightPadLeftIdxRightIdx(smallerLen, largerLen, shift, effectiveStride, effectiveWidth, layerFromAbove):
        # effectiveWidth does not actually factor in.
        if (layerFromAbove == True):
            assert effectiveStride == 1, "have not dealt with stride>1 for"\
                + " layerFromAbove=True"
        assert effectiveStride <= effectiveWidth  # usually catches if you have
        # flipped width & stride
        shift = shift * effectiveStride
        leftPad = max(0, -shift)
        rightPad = max(0, (smallerLen + shift) - largerLen)
        leftIdx = (shift + leftPad)
        rightIdx = (leftIdx + smallerLen)

        return (leftPad, rightPad, leftIdx, rightIdx)

    def mergeArraysTogether(self, smaller, larger, shift):
        newTotalObservationsEver = larger.totalObservationsEver\
            + smaller.totalObservationsEver
        newNumUnderlyingObservations = self.padAndAdd_1d(
            smaller.numUnderlyingObservations, larger.numUnderlyingObservations, shift, effectiveStride=1, effectiveWidth=1, layerFromAbove=False)
        newSummedDataTracks = {}
        for aKey in larger.summedDataTracks:
            effectiveStride = smaller.summedDataTracks[aKey].effectiveStride
            effectiveWidth = smaller.summedDataTracks[aKey].effectiveWidth
            layerFromAbove = smaller.summedDataTracks[aKey].layerFromAbove
            fillValue = smaller.summedDataTracks[aKey].fillValue
            minVisibility = smaller.summedDataTracks[aKey].minVisibility
            data, revCompData = self.padAndAdd_2d(
                smallerArray=smaller.summedDataTracks[aKey].data, largerArray=larger.summedDataTracks[aKey].data, smallerArrayRevComp=smaller.summedDataTracks[aKey].revCompData, largerArrayRevComp=larger.summedDataTracks[aKey].revCompData, shift=shift, effectiveStride=effectiveStride, effectiveWidth=effectiveWidth, layerFromAbove=layerFromAbove)
            newSummedDataTracks[aKey] =\
                DataTrack(
                data=data, revCompData=revCompData, pseudocount=self.summedDataTracks[aKey].pseudocount, effectiveStride=effectiveStride, effectiveWidth=effectiveWidth, layerFromAbove=layerFromAbove, fillValue=fillValue, minVisibility=minVisibility)
        return Grammar(summedDataTracks=newSummedDataTracks, numUnderlyingObservations=newNumUnderlyingObservations, totalObservationsEver=newTotalObservationsEver, minPseudocount=self.minPseudocount, pseudocountFrac=self.pseudocountFrac)

    def padAndAdd_1d(self, smallerArray, largerArray, shift, effectiveStride, effectiveWidth, layerFromAbove):
        assert len(smallerArray.shape) == 1
        assert len(largerArray.shape) == 1
        (leftPad, rightPad, leftIdx, rightIdx) = self.obtainLeftPadRightPadLeftIdxRightIdx(len(
            smallerArray), len(largerArray), shift, effectiveStride, effectiveWidth, layerFromAbove)
        newArray = np.pad(largerArray, pad_width=[
                          (leftPad, rightPad)], mode='constant')
        newArray[leftIdx:rightIdx] += smallerArray
        return newArray

    def padAndAdd_2d(self, smallerArray, largerArray,
                     smallerArrayRevComp, largerArrayRevComp,
                     shift, effectiveStride,
                     effectiveWidth, layerFromAbove):
        assert len(smallerArray.shape) == 2
        assert len(largerArray.shape) == 2
        (leftPad, rightPad, leftIdx, rightIdx) =\
            self.obtainLeftPadRightPadLeftIdxRightIdx(
            smallerArray.shape[1], largerArray.shape[1], shift, effectiveStride, effectiveWidth, layerFromAbove)

        newArray = np.pad(largerArray, pad_width=[
            (0, 0), (leftPad, rightPad)], mode='constant')
        newArray[:, leftIdx:rightIdx] += smallerArray
        assert(newArray.shape[1] >= largerArray.shape[1])
        assert(newArray.shape[1] >= smallerArray.shape[1])

        newArrayRevComp = np.pad(largerArrayRevComp, pad_width=[
            (0, 0), (rightPad, leftPad)], mode='constant')
        newArrayRevComp[:,
                        (newArray.shape[1] - rightIdx):
                        (newArray.shape[1] - leftIdx)]\
            += smallerArrayRevComp

        return newArray, newArrayRevComp


class Seqlet(Grammar):
    SeqletJsonKeys = util.enum(location="location", sequenceId="sequenceId")

    def __init__(self, location, sequenceId, *args, **kwargs):
        super(type(self), self).__init__(*args, **kwargs)
        self.location = location
        self.sequenceId = sequenceId

    def extractDataForSummedDataTrack(self, keyName, fullDataArr, pseudocount, fullRevCompDataArr, revCompFunc, effectiveStride, effectiveWidth, layerFromAbove, fillValue, minVisibility):
        """
            layerFromAbove: changes interpretation of stride and width;
                the layer being augmented exists *above* the layer
                that self.location references
            minVisibility: if a layerFromAbove, make sure that the neurons
                in this conv layer that are included can see at least
                minVisibility bases of the layer in question
        """
        assert revCompFunc is None or fullRevCompDataArr is None,\
            "exactly one of revCompFunc or fullRevCompDataArr should be None"
        assert revCompFunc is not None or fullRevCompDataArr is not None,\
            "exactly one of revCompFunc or fullRevCompDataArr should be None"
        if (fullRevCompDataArr is not None):
            assert fullDataArr.shape == fullRevCompDataArr.shape,\
                "fullDataArr and fullRevCompDataArr need to be the same shape;"\
                + " are "\
                + str(fullDataArr.shape) + " and " + \
                str(fullRevCompDataArr.shape)
        if (layerFromAbove == False):
            leftIdx = self.location[0] * effectiveStride
            rightIdx = (self.location[1] - 1) * \
                effectiveStride + effectiveWidth
            data = fullDataArr[self.sequenceId, :, leftIdx:rightIdx]
            if (fullRevCompDataArr is not None):
                revCompData = fullRevCompDataArr[self.sequenceId, :,
                                                 (fullDataArr.shape[-1] - rightIdx):
                                                 (fullDataArr.shape[-1] - leftIdx)]
            else:
                revCompData = revCompFunc(data)
        elif (layerFromAbove == True):
            if (minVisibility is None):
                minVisibility = np.ceil(effectiveWidth / 2.0)
            assert (self.location[1] - self.location[0]) > minVisibility,\
                "minVisibility is set to " + str(minVisibility) + " but the len"\
                + " of the location is: " + \
                str(self.location[1] - self.location[0])
            assert fillValue is not None,\
                "fillValue must not be None if layerFromAbove=True because"\
                " layers above have fewer dimensions and may require filling"
            assert effectiveStride == 1  # only handling stride of 1 for now
            # find idx in conv layer whose right-most edge-of-view can just
            #(see location[0]+minVisibility-1) in prev layer
            leftIdx = (self.location[0] +
                       minVisibility - 1) - (effectiveWidth - 1)
            # in conv layer, idx i is the one whose leftmost edge-of-view
            # can see idx i in prev layer. Since we are doing slices,
            # this is the index in the conv layer whose leftmost edge-of-view
            # just misses the segment.
            rightIdx = (self.location[1] - (minVisibility - 1))
            lengthToExtract = rightIdx - leftIdx
            numChannels = fullDataArr.shape[1]
            maxLen = fullDataArr.shape[2]
            # note that the computed leftidx and rightidx may be before the
            # start or beyond the end of the conv layer, if there exists
            # no neuron which "just touches" the provided segment. In that
            # situation we just use the padding provided by fillValue as a
            # placeholder for those nonexistent conv neurons
            leftPadding = max(0, 0 - leftIdx)
            rightPadding = max(0, (rightIdx - maxLen))
            # unpadded length is the length we will actually extract
            # from the data
            unpaddedLength = lengthToExtract - (rightPadding + leftPadding)
            data_unpaddedLeftIdx = leftPadding
            data_unpaddedRightIdx = leftPadding + unpaddedLength
            fullDataArr_leftIdx = max(leftIdx, 0)
            fullDataArr_rightIdx = min(maxLen, rightIdx)
            data = np.ones((numChannels, lengthToExtract)) * fillValue
            data[:, data_unpaddedLeftIdx:data_unpaddedRightIdx]\
                = fullDataArr[self.sequenceId, :,
                              (fullDataArr_leftIdx):
                              (fullDataArr_rightIdx)]
            if (fullRevCompDataArr is not None):
                revCompData = np.ones((numChannels, lengthToExtract))\
                    * fillValue
                revCompData[:, (data.shape[1] - data_unpaddedRightIdx):
                               (data.shape[1] - data_unpaddedLeftIdx)]\
                    = fullRevCompDataArr[
                    self.sequenceId, :,
                    (fullDataArr.shape[-1] - fullDataArr_rightIdx):
                    (fullDataArr.shape[-1] - fullDataArr_leftIdx)]
            else:
                revCompData = revCompFunc(data)
        else:
            raise RuntimeError(
                "Unsupported val for layerFromAbove: " + str(layerFromAbove))

        self.addSummedDataTrack(keyName,
                                DataTrack(data, revCompData=revCompData, pseudocount=pseudocount, effectiveStride=effectiveStride, effectiveWidth=effectiveWidth, layerFromAbove=layerFromAbove, fillValue=fillValue, minVisibility=minVisibility))

    def getJsonableObject(self):
        theClass = self.__class__.__name__
        jsonableObject = super(type(self), self).getJsonableObject()
        jsonableObject[Seqlet.SeqletJsonKeys.location] = self.location
        jsonableObject[Seqlet.SeqletJsonKeys.sequenceId] = self.sequenceId
        return jsonableObject

    @classmethod
    def getLoadingKwargsFromJsonableObject(cls, jsonableObject):
        loadingKwargs = Grammar.getLoadingKwargsFromJsonableObject(
            jsonableObject)
        loadingKwargs[Seqlet.SeqletJsonKeys.location] = jsonableObject[
            Seqlet.SeqletJsonKeys.location]
        loadingKwargs[Seqlet.SeqletJsonKeys.sequenceId] = jsonableObject[
            Seqlet.SeqletJsonKeys.sequenceId]
        return loadingKwargs

# for each example, find the critical subset of positives that outweights
# the negatives


def findCriticalSubset(singleExampleContribs, outputBeforeActivation=None, activation=None, thresholdProb=1.0, includeNeg=True):
    if (outputBeforeActivation is None or activation is None):
        assert outputBeforeActivation is None and activation is None  # all or nothing
        assert thresholdProb == 1.0  # don't need these args if including everything
    else:
        assert activation == "sigmoid", "Non sigmoid activation not supported yet!"
    assert thresholdProb >= 0.5 and thresholdProb <= 1.0
    ravelledContribs = enumerate(np.ravel(singleExampleContribs))
    summed_negative = 0
    totalContribsSum = 0
    ravelled_positive = []
    ravelled_all = []
    #assert len(ravelledContribs) > 0
    # partition by negative and positive
    for contrib in ravelledContribs:
        totalContribsSum += contrib[1]
        if contrib[1] < 0:
            summed_negative += contrib[1]
        elif (contrib[1] > 0):
            ravelled_positive.append(contrib)
        # mystery bug...using ravelledContribs in place of
        # ravelled_all does not work...somehow in some cases
        # ravelledContribs gets emptied...
        ravelled_all.append(contrib)
    if (outputBeforeActivation is not None):
        netBias = outputBeforeActivation - totalContribsSum
        sumSoFar = summed_negative + netBias
    criticalSubsetIndices = []
    criticalSubsetContributions = []

    if (outputBeforeActivation is not None):
        if (activation == "sigmoid"):
            activationFunc = lambda x: 1.0 / \
                (1.0 + np.exp(-x))  # sigmoid activation
        else:
            raise RuntimeError("Unsupported activation:", activation)

    sortedPositives = sorted(ravelled_positive, key=lambda x: -x[1])
    sortedAll = sorted(ravelled_all, key=lambda x: -x[1])
    assert len(sortedPositives) > 0
    if (len(sortedAll) == 0):
        print("wtf", sortedPositives)
    assert len(sortedAll) > 0
    i = 0
    if (thresholdProb == 1.0):
        if (includeNeg == True):
            criticalSubsetIndices.extend(x[0] for x in sortedAll)
            criticalSubsetContributions.extend(x[1] for x in sortedAll)
        else:
            criticalSubsetIndices.extend(x[0] for x in sortedPositives)
            criticalSubsetContributions.extend(x[1] for x in sortedPositives)
    else:
        while (activationFunc(sumSoFar) < thresholdProb) and (i < len(sortedPositives)):
            # while (sortedPositives[i][1] >= (0.1)*sortedPositives[0][1] and i
            # < len(sortedPositives)):
            sumSoFar += sortedPositives[i][1]
            criticalSubsetIndices.append(sortedPositives[i][0])
            criticalSubsetContributions.append(sortedPositives[i][1])
            i += 1
    # convert the ravelled incides to unravelled indices
    if (len(criticalSubsetIndices) == 0):
        print("WARN: found an example which has no positive deepLIFT"
              "contribs and output before activation:", outputBeforeActivation)
        print(summed_negative)
        print(outputBeforeActivation)
        print(totalContribsSum)
        unravelledIndices = []
        assert False
    else:
        unravelledIndices = list(
            zip(*np.unravel_index(criticalSubsetIndices, singleExampleContribs.shape)))
    return zip(unravelledIndices, criticalSubsetContributions)


def groupContribsByPos(criticalSubset):
    """
        criticalSubset: array with elements of shape:
            ((channel, row, col), contribution)
        at least one of channel or row must have max
            size 1.
        Returns: dictionary of the form:
            pos -> [array of (channel+row, contribution)]
            channel+row because at least one of them will be 0

    """
    # return pos -> (channel/row, contrib)
    contribsGroupedByPos = defaultdict(list)
    for ((channel, row, col), contribution) in criticalSubset:
        assert row == 0 or channel == 0
        contribsGroupedByPos[col].append((channel + row, contribution))
    return contribsGroupedByPos


def getTotalContribAtPoses(positions, contribsGroupedByPos):
    return [sum(x[1] for x in contribsGroupedByPos[pos])
            for pos in positions]


def getPosToTotalContrib(contribsGroupedByPos):
    return dict((pos, sum(x[1] for x in contribsGroupedByPos[pos]))
                for pos in contribsGroupedByPos)


def getRepresentedChannels(criticalSubset):
    return sorted(Set(x[0] for x in criticalSubset))


def findContiguousWithoutGaps(positions, allowedGap):
    """
        positions: sorted array of positions
        allowedGap = None means take whole seq
    """
    lastPos = positions[0]
    segments = []
    if allowedGap is None:
        segments.append((positions[0], positions[-1]))
        return segments
    start = lastPos
    for pos in positions[1:]:
        if pos - lastPos > allowedGap:
            segments.append((start, lastPos))
            start = pos
        lastPos = pos
    if start != pos:  # handle last position
        segments.append((start, pos))
    return segments


class AbstractSegmentIdentifier(object):

    def __call__(self, criticalSubset, numCols):
        """
            return continuous segments
        """
        raise NotImplementedError()


class FullSegment(AbstractSegmentIdentifier):

    def __call__(self, contribsGroupedByPos, numCols):
        # find the min position
        sortedPositions = sorted(contribsGroupedByPos.keys())
        return [(sortedPositions[0], sortedPositions[-1])]


class FixedWindowAroundPeaks(AbstractSegmentIdentifier):
    """
    Algorithm is as follows:
       compute sums of the deepLIFT contributions in sliding window of size slidingWindowForMaxSize
       find peaks (points whose sliding window sums are larger than their neighbours; for plateaus, take the middle)
       filter out peaks which are not at least ratioToTopPeakToInclude of the tallest peak
       for each peak in order of highest peak first:
          add (peakStart-flankToExpandAroundPeakSize
              , peakStart+slidingWindowForMaxSize+flankToExpandAroundPeakSize)
          to your list of identified segments
          filter out any peaks that are within excludePeaksWithinWindow of this peak to your list
       loop until there are no more candidate peaks left or the total number of segments identified is maxSegments
    """

    def __init__(self, slidingWindowForMaxSize, flankToExpandAroundPeakSize, excludePeaksWithinWindow, ratioToTopPeakToInclude, maxSegments):
        self.slidingWindowForMaxSize = slidingWindowForMaxSize
        self.flankToExpandAroundPeakSize = flankToExpandAroundPeakSize
        self.excludePeaksWithinWindow = excludePeaksWithinWindow
        self.ratioToTopPeakToInclude = ratioToTopPeakToInclude
        self.maxSegments = maxSegments

    def __call__(self, contribsGroupedByPos, numCols):
        posToTotalContrib = getPosToTotalContrib(contribsGroupedByPos)
        return self.getSegments(util.SparseArrFromDict(
            theDict=posToTotalContrib, defaultVal=0, totalLen=numCols))

    def getSegments(self, arr):
        # compute sum using sliding window
        totalContribsRunningWindowSum = util.computeRunningWindowSum(
            arr=arr, windowSize=self.slidingWindowForMaxSize)
        return self.getSegmentsFromRunningWindowSum(totalContribsRunningWindowSum)

    def getSegmentsFromRunningWindowSum(self, totalContribsRunningWindowSum):
        numCols = len(totalContribsRunningWindowSum) + \
            self.slidingWindowForMaxSize - 1
        # find peaks
        potentialPeaks = identifyPeaks(totalContribsRunningWindowSum)
        if (len(potentialPeaks) == 0):
            topLocation = np.argmax(totalContribsRunningWindowSum)
            assert topLocation == 0 or topLocation == len(
                totalContribsRunningWindowSum) - 1
            potentialPeaks = [
                (topLocation, totalContribsRunningWindowSum[topLocation])]
            maxPeak = potentialPeaks[0][1]
        else:
            maxPeak = max([x[1] for x in potentialPeaks])
        # filter out all peaks < "ratio" of max peak
        potentialPeaks = [x for x in potentialPeaks
                          if x[1]
                          >= self.ratioToTopPeakToInclude * maxPeak]
        segments = []
        # find the max peak
        while len(potentialPeaks) > 0 and len(segments) < self.maxSegments:
            ((maxPeakIdx, maxPeak), maxPeak) = util.getBest(
                potentialPeaks, lambda x: x[1], takeMax=True)
            # the running window sum returns the leftmost index. So
            segments.append((max(0, maxPeakIdx - self.flankToExpandAroundPeakSize), min(maxPeakIdx + self.slidingWindowForMaxSize
                                                                                        + self.flankToExpandAroundPeakSize, numCols)))
            # filter out any peaks within self.excludePeaksWithinWindow of
            # the peak
            potentialPeaks = [x for x in potentialPeaks
                              if abs(x[0] - maxPeakIdx) > self.excludePeaksWithinWindow]
        return segments


class AbstractKeepLookingFunc(object):

    def __call__(self, potentialNextPos, currentPos, potentialNextContrib, thisPeakContrib, maxContrib):
        raise NotImplementedError()

# converts a list of dictionaries to a numpy mat


def dictListToNumpyMatrix(dictList, numRows):
    toReturn = np.zeros((numRows, len(dictList)))
    for (posIdx, posDict) in enumerate(dictList):
        for channel in posDict:
            toReturn[int(channel), posIdx] += posDict[channel]
    return toReturn

# single region!


def getSeqletsForArrayOfContribs(singleExampleContribs, wideRevCompArray, revCompFunc, outputBeforeActivation, activation, segmentIdentifier, thresholdProb, sequenceId, includeNeg):
    """
        Returns an array of Grammar objects for a SINGLE REGION
        (singleExampleContribs is for a single region)
    """
    assert wideRevCompArray is None or revCompFunc is None,\
        "Exactly one of wideRevCompArray and revCompFunc should not be None"
    assert (wideRevCompArray is not None) or (revCompFunc is not None),\
        "Exactly one of wideRevCompArray and revCompFunc should not be None"

    contribsInKeySegments, keySegments =\
        extractKeySegments(singleExampleContribs, outputBeforeActivation,
                           activation, segmentIdentifier, thresholdProb, includeNeg)
    numRows = singleExampleContribs.shape[
        0] + singleExampleContribs.shape[1] - 1
    seqlets = []
    for contribs, keySegment in zip(contribsInKeySegments, keySegments):
        data = dictListToNumpyMatrix(contribs, numRows)
        if wideRevCompArray is not None:
            revCompData = wideRevCompArray[:,
                                           (wideRevCompArray.shape[-1] - keySegment[1]):
                                           (wideRevCompArray.shape[-1] - keySegment[0])]
        else:
            revCompData = revCompFunc(data)
        seqlets.append(Seqlet(
            summedDataTracks={
                Grammar.coreDeepLIFTtrackName:
                DataTrack(
                    data=data, revCompData=revCompData, pseudocount=0, effectiveStride=1, effectiveWidth=1, layerFromAbove=False, fillValue=None, minVisibility=None)}, numUnderlyingObservations=1, totalObservationsEver=1, location=keySegment, sequenceId=sequenceId))
    return seqlets


def extractKeySegments(singleExampleContribs, outputBeforeActivation, activation, segmentIdentifier, thresholdProb, includeNeg):
    """
        segmentIdentifier: instance of AbstractSegmentIdentifier; the
            rule for identifying key segments
        numCols: length of the underlying region
    """
    assert activation is None or util.assertIsType(
        activation, str, "activation")
    criticalSubset = findCriticalSubset(singleExampleContribs=singleExampleContribs, outputBeforeActivation=outputBeforeActivation,
                                        activation=activation, thresholdProb=thresholdProb, includeNeg=includeNeg)
    numCols = singleExampleContribs.shape[2]
    assert singleExampleContribs.shape[0] == 1 or singleExampleContribs.shape[
        1] == 1  # either channels or rows must have dim 1
    #util.assertIsType(segmentIdentifier, AbstractSegmentIdentifier, "segmentIdentifier");
    contribsGroupedByPos = groupContribsByPos(criticalSubset)
    keySegments = segmentIdentifier(contribsGroupedByPos, numCols)
    # each of the things in key segment seqlets is an array of dicts
    # and the indices of the dicts are supposed to be the fitlers
    contribsInKeySegments = []
    for keySegment in keySegments:
        contribsInKeySegment = [{}
                                for i in range(keySegment[1] - keySegment[0])]
        for (idx, pos) in enumerate(xrange(keySegment[0], keySegment[1])):
            if pos in contribsGroupedByPos:
                for (aFilter, contribution) in contribsGroupedByPos[pos]:
                    assert aFilter not in contribsInKeySegment[idx]
                    contribsInKeySegment[idx][aFilter] = contribution
        contribsInKeySegments.append(contribsInKeySegment)
    # sort the keySegments by the most important first
    tuplesToSplitUp = sorted(zip(contribsInKeySegments, keySegments),
                             key=lambda x: -sum([contrib for pos in x[0] for contrib in pos.values()]))
    # split them up
    contribsInKeySegments = [x[0] for x in tuplesToSplitUp]
    keySegments = [x[1] for x in tuplesToSplitUp]
    return contribsInKeySegments, keySegments

###################################################
# I have to endure this nonsense because the function
# pickled by Pool.map has to be accessible at the top level
_seqletsForIdx_singleExampleContribs = util.VariableWrapper(None)
_seqletsForIdx_fullRevCompDataArr = util.VariableWrapper(None)
_seqletsForIdx_revCompFunc = util.VariableWrapper(None)
_seqletsForIdx_outputsBeforeActivation = util.VariableWrapper(None)
_seqletsForIdx_activation = util.VariableWrapper(None)
_seqletsForIdx_segmentIdentifier = util.VariableWrapper(None)
_seqletsForIdx_thresholdProb = util.VariableWrapper(None)
_seqletsForIdx_includeNeg = util.VariableWrapper(None)
# Nonsense endured
####################################################


def computeSeqletsForIdx(idx):
    assert _seqletsForIdx_singleExampleContribs.var is not None
    seqlets = getSeqletsForArrayOfContribs(
        singleExampleContribs=_seqletsForIdx_singleExampleContribs.var[idx], wideRevCompArray=(
            None if _seqletsForIdx_fullRevCompDataArr.var is None else
            _seqletsForIdx_fullRevCompDataArr.var[idx]), revCompFunc=_seqletsForIdx_revCompFunc.var, outputBeforeActivation=None if _seqletsForIdx_outputsBeforeActivation.var is None
        else _seqletsForIdx_outputsBeforeActivation.var[idx], activation=_seqletsForIdx_activation.var, segmentIdentifier=_seqletsForIdx_segmentIdentifier.var, thresholdProb=_seqletsForIdx_thresholdProb.var, includeNeg=_seqletsForIdx_includeNeg.var, sequenceId=idx)
    return (seqlets, [idx] * len(seqlets))


def getGrammars(rawDeepLIFTContribs, indicesToGetGrammarsOn, outputsBeforeActivation, activation, thresholdProb, segmentIdentifier, **kwargs):
    print("Get grammars is deprecated as the name was confusing; use getSeqlets")
    return getSeqlets(rawDeepLIFTContribs=rawDeepLIFTContribs, indicesToGetSeqletsOn=indicesToGetGrammarsOn, outputsBeforeActivation=outputsBeforeActivation, activation=activation, thresholdProb=thresholdProb, segmentIdentifier=segmentIdentifier, **kwargs)


def getSeqlets(rawDeepLIFTContribs, indicesToGetSeqletsOn, outputsBeforeActivation, activation, thresholdProb, segmentIdentifier, fullRevCompDataArr=None, revCompFunc=None, includeNeg=True, numThreads=1, secondsBetweenUpdates=1):
    if (revCompFunc is None and fullRevCompDataArr is None):
        revCompFunc = RevCompWithDNArowsSubset(dnaRowsStart=0, dnaRowsEnd=4)
        print("No reverse comp function or rev comp array provided"
              "so assuming you have dna as first 4 rows")
    if (indicesToGetSeqletsOn is None):
        indicesToGetSeqletsOn = xrange(len(rawDeepLIFTContribs))
    assert outputsBeforeActivation is None or\
        (len(rawDeepLIFTContribs) == len(outputsBeforeActivation))\
        , "rawDeepLIFTContribs and outputsBeforeActivation should be the same length"\
        "but are " + str(rawDeepLIFTContribs.shape) + \
        " and " + str(outputsBeforeActivation.shape)
    reload(util)
    assert activation is None or util.assertIsType(
        activation, str, "activation")
    assert len(rawDeepLIFTContribs.shape) == 4  # example, channel, rows, cols
    util.assertIsType(thresholdProb, float, "thresholdProb")
    #util.assertIsType(segmentIdentifier, AbstractSegmentIdentifier, "segmentIdentifier");
    _seqletsForIdx_singleExampleContribs.var = rawDeepLIFTContribs
    _seqletsForIdx_fullRevCompDataArr.var = fullRevCompDataArr
    _seqletsForIdx_revCompFunc.var = revCompFunc
    _seqletsForIdx_outputsBeforeActivation.var = outputsBeforeActivation
    _seqletsForIdx_activation.var = activation
    _seqletsForIdx_segmentIdentifier.var = segmentIdentifier
    _seqletsForIdx_thresholdProb.var = thresholdProb
    _seqletsForIdx_includeNeg.var = includeNeg
    if (numThreads > 1):
        seqletsAndIndicesTuples = util.multiprocessing_map_printProgress(
            secondsBetweenUpdates=secondsBetweenUpdates, numThreads=numThreads, func=computeSeqletsForIdx, iterable=indicesToGetSeqletsOn)
    else:
        seqletsAndIndicesTuples = []
        for x in indicesToGetSeqletsOn:
            if (x % 100 == 0):
                print("Done", x, "of", len(indicesToGetSeqletsOn))
            seqletsAndIndicesTuples.append(computeSeqletsForIdx(x))
    # disentangle/unlist seqletsAndIndicesTuples
    seqletsOnAllExamples = []
    indicesOfSeqlets = []
    for seqletsAndIndicesTuple in seqletsAndIndicesTuples:
        seqletsOnAllExamples.extend(seqletsAndIndicesTuple[0])
        indicesOfSeqlets.extend(seqletsAndIndicesTuple[1])
    assert len(seqletsOnAllExamples) == len(indicesOfSeqlets)
    for (seqlet, index) in zip(seqletsOnAllExamples, indicesOfSeqlets):
        assert seqlet.sequenceId == index
    # sort them by highest contributing seqlets
    contribsForSeqlets = [np.sum(seqlet.summedCoreDeepLIFTtrack)
                          for seqlet in seqletsOnAllExamples]
    sortOrder = [x[0] for x in sorted(
        enumerate(contribsForSeqlets), key=lambda x: -x[1])]
    seqletsOnAllExamples = [seqletsOnAllExamples[i] for i in sortOrder]
    indicesOfSeqlets = [indicesOfSeqlets[i] for i in sortOrder]
    return seqletsOnAllExamples, indicesOfSeqlets


###################################################
# I have to endure this nonsense because the function
# pickled by Pool.map has to be accessible at the top level
_computeBestCorrelation_arrays = util.VariableWrapper(None)
_computeBestCorrelation_revCompArrays = util.VariableWrapper(None)
_computeBestCorrelation_accountForRevComp = util.VariableWrapper(None)
_computeBestCorrelation_normaliseFunc = util.VariableWrapper(None)
_computeBestCorrelation_smallerPerPosNormFuncs = util.VariableWrapper(None)
_computeBestCorrelation_largerPerPosNormFuncs = util.VariableWrapper(None)
# Nonsense endured
####################################################


def computeBestCorrelation(tupleToCorrelate):
    bestCorrelation, shift, firstIsSmaller =\
        util.getBestLengthwiseCrossCorrelationOfArrays(
            _computeBestCorrelation_arrays.var[tupleToCorrelate[0]], _computeBestCorrelation_arrays.var[tupleToCorrelate[1]], normaliseFunc=_computeBestCorrelation_normaliseFunc.var, smallerPerPosNormFuncs=_computeBestCorrelation_smallerPerPosNormFuncs.var, largerPerPosNormFuncs=_computeBestCorrelation_largerPerPosNormFuncs.var)
    if (_computeBestCorrelation_accountForRevComp.var == True):
        bestCorrelationRev, shiftRev, firstIsSmallerRev =\
            util.getBestLengthwiseCrossCorrelationOfArrays(
                _computeBestCorrelation_arrays.var[tupleToCorrelate[0]], _computeBestCorrelation_revCompArrays.var[tupleToCorrelate[1]], normaliseFunc=_computeBestCorrelation_normaliseFunc.var, smallerPerPosNormFuncs=_computeBestCorrelation_smallerPerPosNormFuncs.var, largerPerPosNormFuncs=_computeBestCorrelation_largerPerPosNormFuncs.var)
        return max(bestCorrelation, bestCorrelationRev)
    else:
        return bestCorrelation


def getArrayForCrossCorrFromGrammar(grammar, subtracksToInclude, subtrackNormaliseFunc, useSummed, revComp):
    arr = np.concatenate([subtrackNormaliseFunc(
        (grammar.getNormalisedDataTrack(subtrackName)
         if (not revComp) else
         grammar.getRevCompedNormalisedDataTrack(subtrackName))
        if (not useSummed) else
        (grammar.getSummedDataTrack(subtrackName)
         if (not revComp) else
         grammar.getRevCompedSummedDataTrack(subtrackName)))
        for subtrackName in subtracksToInclude], axis=0)
    return arr


def oneOverLen(arr):
    return arr / float(np.shape(arr)[1])


def getSummedChannelSignals(seqlets, subtracksToInclude, revComp):
    # not casting to an array immediately as seqlets could be of varying
    # lengths
    listOfChannelTracks = [getArrayForCrossCorrFromGrammar(seqlet, subtracksToInclude, subtrackNormaliseFunc=identity, revComp=revComp, useSummed=False)
                           for seqlet in seqlets]
    # sum them along the length axis, which may be of varying lengths
    summedAlongLen = np.array([x.sum(axis=1) for x in listOfChannelTracks])
    return summedAlongLen


def getChannelSignals(seqlets, subtracksToInclude, revComp):
    channelSignals = getSummedChannelSignals(
        seqlets, subtracksToInclude, revComp)
    return channelSignals


def euclideanSimilarity(twoDVecs1, twoDVecs2):
    return -np.linalg.norm(twoDVecs1[:, None, :]
                           - twoDVecs2[None, :, :], axis=2)


def normaliseTwoDVecsByMagnitude(twoDVecs):
    norms = np.linalg.norm(twoDVecs, axis=1)[:, None]
    norms = np.maximum(norms, 0.0000001)
    return twoDVecs / norms


def cosineSimilarity(twoDVecs1, twoDVecs2):
    # normalise the vectors
    normalisedTwoDVecs1 = normaliseTwoDVecsByMagnitude(twoDVecs1)
    normalisedTwoDVecs2 = normaliseTwoDVecsByMagnitude(twoDVecs2)
    return np.sum(normalisedTwoDVecs1[:, None, :]
                  * normalisedTwoDVecs2[None, :, :], axis=2)


def cosineOnPosSimilarity(twoDVecs1, twoDVecs2):
    twoDVecs1 = twoDVecs1 * (twoDVecs1 > 0.0)
    twoDVecs2 = twoDVecs2 * (twoDVecs2 > 0.0)
    return cosineSimilarity(twoDVecs1, twoDVecs2)

ChannelSimilarityMode = util.enum(cosine=cosineSimilarity,
                                  cosineOnPos=cosineOnPosSimilarity,
                                  euclidean=euclideanSimilarity)


def getChannelSimilarityMatrix(seqlets,
                               subtracksToInclude,
                               channelSimilarityFunc,
                               useRevComp):
    """
        subtracksToInclude refers to subtracts corresponding to a conv
            layer, and the distance matrix will be based on the distance
            between the sum over the conv filters.
    """
    channelSignals = getChannelSignals(seqlets,
                                       subtracksToInclude,
                                       revComp=False)
    # channelSignalsRevComp will be the same as channelSignals if
    #useRevComp is False
    channelSignalsRevComp = getChannelSignals(seqlets,
                                              subtracksToInclude,
                                              revComp=useRevComp)
    similarity_fwd = channelSimilarityFunc(
        twoDVecs1=channelSignals,
        twoDVecs2=channelSignals)
    similarity_rev = channelSimilarityFunc(
        twoDVecs1=channelSignals,
        twoDVecs2=channelSignalsRevComp)
    similarity = np.minimum(similarity_fwd, similarity_rev)
    return similarity


def getCorrelationMatrix(seqlets, subtracksToInclude=[Grammar.coreDeepLIFTtrackName], subtrackNormaliseFunc=util.CROSSC_NORMFUNC.meanAndTwoNorm, normaliseFunc=util.CROSSC_NORMFUNC.none, smallerPerPosNormFuncs=[], largerPerPosNormFuncs=[], accountForRevComp=True, numThreads=1, secondsBetweenUpdates=1, xcorBatchSize=None):
    startTime = time.time()
    print("Num words:", len(seqlets))
    # if the num of underlying observations is not 1, then
    # should use some kind of normalised arra below and not
    # getSummedDataTrack.
    # assert all([seqlet.totalObservationsEver==1 for seqlet in seqlets]);

    listToMakeAnArray = [getArrayForCrossCorrFromGrammar(seqlet, subtracksToInclude, subtrackNormaliseFunc, revComp=False, useSummed=False)
                         for seqlet in seqlets]
    arrays = listToMakeAnArray
    if (xcorBatchSize is not None):
        # we have to call normaliseFunc on each one since won't be done
        # by xcor
        listToMakeAnArray = [normaliseFunc(x) for x in listToMakeAnArray]
    #print("Shape of thing to cross corr:",arrays.shape)
    _computeBestCorrelation_arrays.var = arrays
    _computeBestCorrelation_accountForRevComp.var = accountForRevComp
    if (accountForRevComp):
        revCompArrays = [getArrayForCrossCorrFromGrammar(
            seqlet, subtracksToInclude, subtrackNormaliseFunc, useSummed=False, revComp=True)
            for seqlet in seqlets]
        if (xcorBatchSize is not None):
            revCompArrays = [normaliseFunc(x) for x in revCompArrays]
        _computeBestCorrelation_revCompArrays.var = revCompArrays
    _computeBestCorrelation_normaliseFunc.var = normaliseFunc
    _computeBestCorrelation_smallerPerPosNormFuncs.var = smallerPerPosNormFuncs
    _computeBestCorrelation_largerPerPosNormFuncs.var = largerPerPosNormFuncs
    seqletsCorrMat = np.zeros([len(seqlets)] * 2)
    tuplesToCorrelate = []
    for idx1 in xrange(len(seqlets)):
        for idx2 in xrange(idx1, len(seqlets)):
            tuplesToCorrelate.append((idx1, idx2))
    if (xcorBatchSize is None):
        if (numThreads > 1):
            correlations = util.multiprocessing_map_printProgress(
                secondsBetweenUpdates=secondsBetweenUpdates, numThreads=numThreads, func=computeBestCorrelation, iterable=tuplesToCorrelate)
        else:
            correlations = [computeBestCorrelation(
                x) for x in tuplesToCorrelate]
        for (i, (seqletIdx1, seqletIdx2)) in enumerate(tuplesToCorrelate):
            seqletsCorrMat[seqletIdx1, seqletIdx2] = correlations[i]
            seqletsCorrMat[seqletIdx2, seqletIdx1] = correlations[i]
    else:
        arrays = np.array(listToMakeAnArray)
        if (accountForRevComp):
            revCompArrays = np.array(revCompArrays)
        import modisco
        import modisco.util
        correlationsNoRevComp = modisco.util.get_max_cross_corr(
            filters=arrays.copy(),
            things_to_scan=arrays.copy(),
            verbose=True,
            batch_size=xcorBatchSize)
        if (accountForRevComp):
            correlationsRevComp = modisco.util.get_max_cross_corr(
                filters=arrays.copy(),
                things_to_scan=revCompArrays.copy(),
                verbose=True,
                batch_size=xcorBatchSize)
            correlations = np.maximum(correlationsNoRevComp,
                                      correlationsRevComp)
        else:
            correlations = correlationsNoRevComp
        #import xcor;
        #correlationsNoRevComp, indicesNoRevComp = xcor.crossCorrelateMatrix(arrays.copy(), arrays.copy(), verbose=True, batch_row_size=xcorBatchSize);
        # if (accountForRevComp):
            #correlationsRevComp, indicesRevComp = xcor.crossCorrelateMatrix(arrays.copy(), revCompArrays.copy(), verbose=True, batch_row_size=xcorBatchSize);
            #correlations = np.maximum(correlationsNoRevComp, correlationsRevComp);
        # else:
            # correlations=correlationsNoRevComp;
        count = 0
        for (seqletIdx1, seqletIdx2) in tuplesToCorrelate:
            count += 1
            correlation = correlations[seqletIdx1, seqletIdx2]
            seqletsCorrMat[seqletIdx1, seqletIdx2] = correlation
            seqletsCorrMat[seqletIdx2, seqletIdx1] = correlation
    print("Seconds to compute corr mat:", time.time() - startTime)
    return seqletsCorrMat


def augmentGrammarsWithData(grammars, *args, **kwargs):
    print("Deprecated; use augmentSeqletsWithData")
    return augmentSeqletsWithData(*args, seqlets=grammars, **kwargs)


def augmentSeqletsWithData(seqlets, fullDataArr, keyName, pseudocount, revCompFunc=None, fullRevCompDataArr=None, indicesToSubset=None, effectiveStride=1, effectiveWidth=1, layerFromAbove=False, fillValue=None, minVisibility=None):
    assert revCompFunc is None or fullRevCompDataArr is None,\
        "Exactly one of revCompFunc or fullRevCompDataArr should be None"
    assert revCompFunc is not None or fullRevCompDataArr is not None,\
        "Exactly one of revCompFunc or fullRevCompDataArr should be None"
    if (indicesToSubset is not None):
        fullDataArr = [fullDataArr[i] for i in indicesToSubset]
    for seqlet in seqlets:
        seqlet.extractDataForSummedDataTrack(
            keyName=keyName, fullDataArr=fullDataArr, pseudocount=pseudocount, fullRevCompDataArr=fullRevCompDataArr, revCompFunc=revCompFunc, effectiveStride=effectiveStride, effectiveWidth=effectiveWidth, layerFromAbove=layerFromAbove, fillValue=fillValue, minVisibility=minVisibility)

# create a merged grammar for the clusters


def createMergedGrammars(clusterLabels, grammars, subtracksToInclude=[Grammar.coreDeepLIFTtrackName], subtrackNormaliseFunc=util.CROSSC_NORMFUNC.meanAndSdev, normaliseFunc=util.CROSSC_NORMFUNC.none, smallerPerPosNormFuncs=[], largerPerPosNormFuncs=[], accountForRevComp=True):
    clusterLabelToMergedGrammar = {}
    for clusterLabel, grammar in zip(clusterLabels, grammars):
        if clusterLabel not in clusterLabelToMergedGrammar:
            clusterLabelToMergedGrammar[clusterLabel] = grammar
        else:
            clusterLabelToMergedGrammar[clusterLabel] = clusterLabelToMergedGrammar[clusterLabel]\
                .merge(grammar, subtracksToInclude=subtracksToInclude, subtrackNormaliseFunc=subtrackNormaliseFunc, normaliseFunc=normaliseFunc, smallerPerPosNormFuncs=smallerPerPosNormFuncs, largerPerPosNormFuncs=largerPerPosNormFuncs, revComp=accountForRevComp)
    return clusterLabelToMergedGrammar


def getTsneEmbeddingOfGrammars(grammarsCorrMat, perplexity, verbose=0, random_state=None):
    import sklearn
    from sklearn import manifold
    tsne = manifold.TSNE(metric='precomputed', perplexity=perplexity,
                         verbose=verbose, random_state=random_state)
    grammarsDistMat = np.max(grammarsCorrMat) - grammarsCorrMat
    embedding = tsne.fit_transform(grammarsDistMat)
    return embedding


def colorTSNEembeddingBySpectralClustering(mat, embedding, n_clusters, colors=None, affinity='precomputed', *args, **kwargs):
    if (n_clusters == 1):
        labels = [0 for x in embedding]
    else:
        labels = getSpectralClustering(mat, n_clusters, affinity)
    mplh.scatterPlot(embedding, labels=labels, colors=colors, *args, **kwargs)
    return labels


def colorTSNEembeddingByClusterer(embedding, clusterer, colors=None, *args, **kwargs):
    labels = clusterer.fit_predict(embedding)
    mplh.scatterPlot(embedding, labels=labels, colors=colors, *args, **kwargs)
    return labels


def getSpectralClustering(mat, n_clusters, affinity):
    from sklearn.cluster import SpectralClustering
    spectral = SpectralClustering(n_clusters=n_clusters, affinity=affinity)
    labels = spectral.fit_predict(mat)
    return labels


def getKMeansClustering(mat, **kwargs):
    import sklearn.cluster
    clf = sklearn.cluster.KMeans(**kwargs)
    labels = clf.fit_predict(mat)
    return labels


def getRunningSumOfPositionWeights(array, startFromLeft):
    theLen = array.shape[1]
    # +1 because we want to have the value for an inclusive end
    runningSum = np.zeros(array.shape[1] + 1)
    sumSoFar = 0
    for i in xrange(theLen):
        if (startFromLeft):
            idx = i
        else:
            idx = theLen - i
        if (i > 0):  # aha this fixes all my problems
            sumSoFar += np.sum(array[:, idx - 1 if startFromLeft else idx])
        runningSum[idx] = sumSoFar
    return runningSum


def adjustGrammarUsingTrimmingCriterion(grammar, trimmingFunc):
    (start, end) = trimmingFunc(grammar)
    summedDataTracks = {}
    for key, dataTrack in grammar.summedDataTracks.items():
        if (dataTrack.layerFromAbove == False):
            start_idx = start * dataTrack.effectiveStride
            #(end-1) because "end" is the boundary i.e. it has +1 added
            end_idx = (((end - 1) * dataTrack.effectiveStride)
                       + dataTrack.effectiveWidth)
        elif (dataTrack.layerFromAbove):
            assert dataTrack.effectiveStride == 1,\
                "Have not handled stride != 1 yet for layerFromAbove"
            start_idx = start
            end_idx = dataTrack.data.shape[-1] -\
                (grammar.numUnderlyingObservations.shape[-1] - end)
        data = dataTrack.data[:, start_idx:end_idx]
        revCompData = dataTrack.revCompData[:,
                                            (dataTrack.data.shape[-1] - end_idx):
                                            (dataTrack.data.shape[-1] - start_idx)]
        summedDataTracks[key] = DataTrack(
            data=data, revCompData=revCompData, pseudocount=dataTrack.pseudocount, effectiveStride=dataTrack.effectiveStride, effectiveWidth=dataTrack.effectiveWidth, layerFromAbove=dataTrack.layerFromAbove, fillValue=dataTrack.fillValue, minVisibility=dataTrack.minVisibility)
    return Grammar(numUnderlyingObservations=grammar.numUnderlyingObservations[start:end], totalObservationsEver=grammar.totalObservationsEver, summedDataTracks=summedDataTracks, minPseudocount=grammar.minPseudocount, pseudocountFrac=grammar.pseudocountFrac)


def adjustGrammarsUsingTrimmingCriterion(labelToGrammar, trimmingFunc):
    """
        labelsToGrammar is a dictionary, indented to be the dict
            produced by createMergedGrammars
    """
    toReturn = {}
    for (label, grammar) in labelToGrammar.items():
        toReturn[label] = adjustGrammarUsingTrimmingCriterion(
            grammar, trimmingFunc)
    return toReturn


class TrimmingFunc(object):

    def __call__(self, grammar):
        raise NotImplementedError()


class TrimArrayColumnsToNumUnderlyingObs(TrimmingFunc):

    def __init__(self, percentObs):
        self.percentObs = percentObs

    def __call__(self, grammar):
        """
            Will retain all indices where numUnderlyingObservations
                is at least percentObs of totalObservationsEver
        """
        filteredIndices = [x[0] for x in enumerate(grammar.numUnderlyingObservations)
                           if x[1] >= self.percentObs * grammar.totalObservationsEver]
        return (filteredIndices[0], filteredIndices[-1] + 1)


class TrimArrayColumnsToPercent(TrimmingFunc):

    def __init__(self, percent):
        self.percent = percent
        print("WARNING: this function has not been updated")

    def __call__(self, grammar):
        """
            Will find the smallest subset of the array that retains x% of the signal.
        """
        array = grammar.summedCoreDeepLIFTtrack
        # for now implement brute force thing because not using it for anything intensive but
        # i am pretty sure this can be done more efficiently.
        assert np.sum(np.abs(array) - array) == 0
        totalSum = np.sum(array)
        # compute running sums of what would be excluded from the left and the
        # right
        sumsFromLeft = getRunningSumOfPositionWeights(
            array, startFromLeft=True)
        sumsFromRight = getRunningSumOfPositionWeights(
            array, startFromLeft=False)
        bestTrim = util.GetBest_Min()
        # try all combos of start and end.
        for (leftEdge, sumFromLeft) in enumerate(sumsFromLeft):
            for (rightEdge, sumFromRight) in enumerate(sumsFromRight):
                if (rightEdge > leftEdge):
                    if (totalSum - (sumFromLeft + sumFromRight) >= totalSum * self.percent):
                        bestTrim.process((leftEdge, rightEdge),
                                         (rightEdge - leftEdge))
        (bestLeft, bestRight) = bestTrim.getBestObj()
        return array[:, bestLeft:bestRight], (bestLeft, bestRight)


class TrimArrayColumnsToPeak(TrimmingFunc):

    def __init__(self, slidingWindowSizeForPeak, flanksToExpand, trackNameToUse, useRangeNotSum):
        """
            Will look at the summed version of trackNameToUse (so as to overweight positions
                with more observations). Will find the sliding window of size
                slidingWindowSizeForPeak of the highest weight, and will expand by
                flanksToExpand on either side. useRangeNotSum=True will use the range between
                the values(bases) at each position as the weight (appropriate for, eg, gradients
                on sequence).
            Recommended settings for sequence data:
                trackNameToUse = [name of gradients track, usually "gradients"]
                useRangeNotSum = True
            Recommended settings for all other data:
                trackNameToUse = Grammar.coreDeepLIFTtrackName
                useRangeNotSum = False
        """
        self.slidingWindowSizeForPeak = slidingWindowSizeForPeak
        self.flanksToExpand = flanksToExpand
        self.trackNameToUse = trackNameToUse
        self.useRangeNotSum = useRangeNotSum

    def __call__(self, grammar):
        """
            Using a sliding window of size slidingWindowSizeForPeak,
                will find the peak in the importance of array cols.
                Will then expand
                the sliding window by flanksToExpand, and return that
                as the final array.
        """
        # NOTE the use of the summed data track and not the normalised data
        # track, to overweight those positions with more observations
        array = grammar.getSummedDataTrack(self.trackNameToUse)
        # recall: grammarArray.shape = (4, 61)
        # find the sum at each position
        if (self.useRangeNotSum):
            valPerPosition = np.max(array, axis=0) - np.min(array, axis=0)
        else:
            valPerPosition = np.sum(array, axis=0)
        slidingWindowSums = util.computeRunningWindowSum(
            valPerPosition, self.slidingWindowSizeForPeak)
        maxPos = np.argmax(slidingWindowSums)
        startPos = max(0, maxPos - self.flanksToExpand)
        endPos = min(
            array.shape[1], maxPos + self.slidingWindowSizeForPeak + self.flanksToExpand)
        return (startPos, endPos)


def printLabelAndGrammar(grammars, **kwargs):
    if isinstance(grammars, list):
        grammars = dict(enumerate(grammars))
    for (label, grammar) in grammars.items():
        printGrammar(grammar, title="grammar " + str(label)
                     + ", totalObservationsEver:" + str(grammar.totalObservationsEver), **kwargs)


def printGrammar(grammar, trackNamesToPrint, heightPerTrack=3, minObs=0, plotPosEvery=1, title="default title"):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as grd

    numUnderlyingObservations = grammar.numUnderlyingObservations.astype("int")
    # find first index with minObs observations:
    idxsPassingNumObsThreshold = [i for (i, val) in enumerate(numUnderlyingObservations)
                                  if val >= minObs]
    assert len(idxsPassingNumObsThreshold) >= 1,\
        "only " + str(len(idxsPassingNumObsThreshold))\
        + " positions have at least " + \
        str(minObs) + " underlying observations"
    firstIdx = idxsPassingNumObsThreshold[0]
    lastIdx = idxsPassingNumObsThreshold[-1] + 1
    numUnderlyingObservations = numUnderlyingObservations[firstIdx:lastIdx]

    plt.clf()
    width = lastIdx - firstIdx
    fig_width = 20 + width / 10
    fig = plt.figure(
        figsize=(fig_width, heightPerTrack * len(trackNamesToPrint)))
    for (i, trackName) in enumerate(trackNamesToPrint):
        ax = fig.add_subplot(len(trackNamesToPrint), 1, i + 1)
        arr = grammar.getNormalisedDataTrack(trackName)
        summedDataTrack = grammar.summedDataTracks[trackName]
        arr = arr[:, firstIdx * summedDataTrack.effectiveStride:(
            lastIdx - 1) * summedDataTrack.effectiveStride + summedDataTrack.effectiveWidth]
        if (arr.shape[0] == 4):
            letter_heights = arr.T
            pos_heights = np.copy(letter_heights)
            pos_heights[letter_heights < 0] = 0
            neg_heights = np.copy(letter_heights)
            neg_heights[letter_heights > 0] = 0
            for x_pos, heights in enumerate(letter_heights):
                letters_and_heights = sorted(izip(heights, 'ACGT'))
                y_pos_pos = 0.0
                y_neg_pos = 0.0
                for height, letter in letters_and_heights:
                    if height > 0:
                        plot.add_letter_to_axis(
                            ax, letter, -0.5 + x_pos, y_pos_pos, height)
                        y_pos_pos += height
                    else:
                        plot.add_letter_to_axis(
                            ax, letter, -0.5 + x_pos, y_neg_pos, height)
                        y_neg_pos += height
            if (i == len(trackNamesToPrint)):
                ax.set_xlabel('pos')
            ax.set_aspect(aspect='auto', adjustable='box')
            ax.autoscale_view()
        elif (arr.shape[0] == 1):
            ax.plot(range(arr.shape[1]), arr.squeeze(), 'k', lw=0.5)
            ax.axhline(0, linestyle='dashed', color='black')
        else:
            mplh.plotHeatmapGivenAx(
                ax, data=arr, logTransform=False, zeroCenter=True, cmap=plt.cm.coolwarm)
        # else:
        #    raise RuntimeError("Unsure how to deal with shape "+str(arr.shape));
        ax.set_ylabel(trackName)
        ax.set_xlim(-1, arr.shape[1] + 1)
        xticks_locations = range(-1, arr.shape[1] + 1)
        ax.set_xticks(xticks_locations[::plotPosEvery])
        ax_2 = ax.twiny()
        numObsSpacing = plotPosEvery + 1
        ax_2.set_xticks(xticks_locations[::numObsSpacing])
        ax_2.set_xticklabels(numUnderlyingObservations[::numObsSpacing])
        if (i == 0):
            ax_2.set_xlabel('numObs')
    # plt.title(title)
    plt.show()


def printGrammarWithIdx(grammars, idx, *args, **kwargs):
    """
        convencience function that calls printGrammar on the
            specified index and generates the title accordingly
    """
    printGrammar(grammars[idx], title=idx, *args, **kwargs)
    #printGrammar(grammars[idx].getRevCompGrammar(), title=idx, *args, **kwargs);


def saveGrammarsToPkl(grammars, pklFile):
    import pickle
    reload(pickle)
    toPkl = [(x.summedDataTracks, x.numUnderlyingObservations, x.totalObservationsEver)
             for x in grammars]
    pickle.dump(toPkl, open(pklFile, 'w'))


def loadGrammarsFromPkl(pklFile):
    """
        expecting pickled file that contains a list of
            (numpyArray, numUnderlyingObservations)...
    """
    grammarsRawData = pickle.load(pklFile)
    return [Grammar(summedDataTracks=x[0], numUnderlyingObservations=x[1], totalObservationsEver=x[2])
            for x in grammarsRawData]


def getTopNGreedyNonOverlappingCorrScores(
        largerArr, smallerArr, revCompFunc, N, excludeHitsWithinWindow, normaliseFunc, smallerPerPosNormFuncs, largerPerPosNormFuncs, auxLargerForPerPosNorm, auxLargerPerPosNormFuncs, smallerIsPalindrome):
    """
        will greedily take the best positions
            and will ignore hits within excludePeaksWithinWindow
            of an already-included position
        auxLargerForPerPosNorm is intended for sequence.
            idea is to take [gradientMatch/maxMatch (computable from gradient)]*deepLIFTstrength
            in other words: gradient match is "how well does this motif match what was being looked for"
                            and deepLIFTstrength/maxMatch is "how much of what was being looked for was gotten"
        smallerIsPalindrome: if True, will not distinguish between forward and reverse orientation hits (will
            take the max of both and will declare everything to be in the forward orientation)
        Returns: scores, positions, fwd, leftIdxs, rightIdxs - each is an array of length N
        fwd = 1 if in fwd orientation, -1 if in reverse
        leftIdxs and rightIdxs are the left and right indexes of the hit
        positions = leftIdx + (length of smallerArr) if fwd = -1, else leftIdx; think of it as
            corresponding to where the front of the protein would contact the sequence (where the
            front is the defined as the part of the protein that contacts the left end of the motif
            hit when it is in the fwd orientation). The adjustment is useful for grammar detection.
    """
    #...some inefficiency if the normalisePerPos is being done twice.
    # actually possibly more than a little inefficiency.
    crossCorrelations_fwd, firstIsSmaller, smallerLen =\
        util.crossCorrelateArraysLengthwise(largerArr, smallerArr, normaliseFunc=normaliseFunc, smallerPerPosNormFuncs=smallerPerPosNormFuncs,
                                            largerPerPosNormFuncs=largerPerPosNormFuncs, auxLargerForPerPosNorm=auxLargerForPerPosNorm, auxLargerPerPosNormFuncs=auxLargerPerPosNormFuncs)
    crossCorrelations_rev, firstIsSmaller, smallerLen =\
        util.crossCorrelateArraysLengthwise(largerArr, revCompFunc(smallerArr), normaliseFunc=normaliseFunc, smallerPerPosNormFuncs=smallerPerPosNormFuncs,
                                            largerPerPosNormFuncs=largerPerPosNormFuncs, auxLargerForPerPosNorm=auxLargerForPerPosNorm, auxLargerPerPosNormFuncs=auxLargerPerPosNormFuncs)
    assert firstIsSmaller == False
    crossCorrelations = np.maximum(
        crossCorrelations_fwd, crossCorrelations_rev)
    if (not smallerIsPalindrome):
        hitIsFwd = 1 * (crossCorrelations_fwd >= crossCorrelations_rev) + - \
            1 * (crossCorrelations_fwd < crossCorrelations_rev)
    else:
        hitIsFwd = np.ones(crossCorrelations.shape)

    scores, positions, fwd, leftIdxs, rightIdxs = [], [], [], [], []
    if (N == 1):
        bestIdx = np.argmax(crossCorrelations)
        scores.append(crossCorrelations[bestIdx])
        isFwd = hitIsFwd[bestIdx]
        # This shift is necessary because cross correlation pads in front by
        # smallerLen-1
        actualPos = bestIdx - (smallerLen - 1)
        leftIdxs.append(actualPos)
        rightIdxs.append(actualPos + smallerLen)
        positions.append(actualPos + (0 if isFwd == 1 else smallerLen))
        fwd.append(isFwd)
    else:
        # sort scores
        sortedScores = sorted(
            enumerate(crossCorrelations), key=lambda x: -x[1])
        i = 0
        idxs = []
        while len(scores) < N and i < len(sortedScores):
            # only consider the idx if it is not within
            # excludePeaksWithinWindow of any of the included
            # idxs
            skip = any([abs(sortedScores[i][0] - x) <= excludeHitsWithinWindow
                        for x in idxs])
            if (not skip):
                scores.append(sortedScores[i][1])
                isFwd = hitIsFwd[sortedScores[i][0]]
                idxs.append(sortedScores[i][0])
                fwd.append(isFwd)
            i += 1
        # the shift of (smallerLen-1) is necessary because cross correlation pads by smallerLen-1
        # in front
        positions = [x + (0 if hitIsFwd[x] == 1 else smallerLen) for x in idxs]
        positions = [x - (smallerLen - 1) for x in positions]
        leftIdxs = [x - (smallerLen - 1)
                    if x is not None else None for x in idxs]
        rightIdxs = [x + smallerLen for x in leftIdxs]
        if (len(scores) < N):
            print("Warning: you wanted", N, "scores, but with "
                  "your excludeHitsWithinWindow setting of", excludeHitsWithinWindow, "and largerArr len", largerArr.shape[
                      1], "could not get more than ", len(scores), "peaks at positions", positions,
                  " - in the meantime I will fill the rest with 0")
            leftIdxs.extend([None] * (N - len(scores)))
            rightIdxs.extend([None] * (N - len(scores)))
            positions.extend([None] * (N - len(scores)))
            scores.extend([0] * (N - len(scores)))
            fwd.extend([True] * (N - len(scores)))
    assert firstIsSmaller == False
    # if you change the line below, please also change
    # recastOutputOfTopNGreedy accordingly!
    return scores, positions, fwd, leftIdxs, rightIdxs

###################################################
# I have to endure this nonsense because the function
# pickled by Pool.map has to be accessible at the top level
_topNgreedy_largerArrs = util.VariableWrapper(None)
_topNgreedy_smallerArrs = util.VariableWrapper(None)
_topNgreedy_revCompFunc = util.VariableWrapper(None)
_topNgreedy_N = util.VariableWrapper(None)
_topNgreedy_excludeHitsWithinWindow = util.VariableWrapper(None)
_topNgreedy_normaliseFunc = util.VariableWrapper(None)
_topNgreedy_smallerPerPosNormFuncs = util.VariableWrapper(None)
_topNgreedy_largerPerPosNormFuncs = util.VariableWrapper(None)
_topNgreedy_auxLargerForPerPosNorm = util.VariableWrapper(None)
_topNgreedy_auxLargerPerPosNormFuncs = util.VariableWrapper(None)
_topNgreedy_palindromes = util.VariableWrapper(None)
# Nonsense endured
####################################################


def getTopNGreedyNonOverlappingCorrScores_forParallel(i):
    smallerArrCorrs = [
        getTopNGreedyNonOverlappingCorrScores(
            largerArr=_topNgreedy_largerArrs.var[i], smallerArr=smallerArr, revCompFunc=_topNgreedy_revCompFunc.var, N=_topNgreedy_N.var, excludeHitsWithinWindow=_topNgreedy_excludeHitsWithinWindow.var, normaliseFunc=_topNgreedy_normaliseFunc.var, smallerPerPosNormFuncs=_topNgreedy_smallerPerPosNormFuncs.var, largerPerPosNormFuncs=_topNgreedy_largerPerPosNormFuncs.var, auxLargerForPerPosNorm=None if _topNgreedy_auxLargerForPerPosNorm.var is None
            else _topNgreedy_auxLargerForPerPosNorm.var[i], auxLargerPerPosNormFuncs=_topNgreedy_auxLargerPerPosNormFuncs.var, smallerIsPalindrome=(smallerArrIdx in _topNgreedy_palindromes.var)
        )
        for (smallerArrIdx, smallerArr) in enumerate(_topNgreedy_smallerArrs.var)
    ]
    return smallerArrCorrs

Hit = namedtuple("Hit", ["score", "pos", "fwd", "leftIdx",
                         "rightIdx", "motifIdx", "inputIdx", "grammarIdx"])
# the defaults for leftIdx/rightIdx/motifIdx/grammarIdx/inputIdx are 'None'
Hit.__new__.__defaults__ = (None, None, None, None, None)
#"grammarIdx" is only there for back-compat with a time where "motifs" were called "grammars"
# see documentation of getTopNGreedyNonOverlappingCorrScores for pos vs
# fwd vs leftIdx vs rightIdx


def recastOutputOfTopNGreedy(outputOfTopNgreedy):
    """
        Recasts the output of getTopNGreedyNonOverlappingCorrScores_onFullSet
            input is: [num examples x num motifs x 5 x N]
        to return something like this:
            [num of motifs x num examples x N (as in the "topN" scores)]
        Each entry of the third dimension is a "Hit" object (see above)
    """
    hitsForDifferentMotifs = [[] for i in range(outputOfTopNgreedy.shape[1])]
    for (inputIdx, example) in enumerate(outputOfTopNgreedy):
        for (motifNumber, motifHits) in enumerate(example):
            hitsForThisMotif = []  # will store the motif hits in a nice format
            for hitNumber in range(len(motifHits[0])):
                hitsForThisMotif.append(Hit(score=motifHits[0][hitNumber], pos=motifHits[1][hitNumber], fwd=motifHits[2][
                                        hitNumber], leftIdx=motifHits[3][hitNumber], rightIdx=motifHits[4][hitNumber], motifIdx=motifNumber, inputIdx=inputIdx))
            hitsForDifferentMotifs[motifNumber].append(hitsForThisMotif)
    return hitsForDifferentMotifs


def getTopNGreedyNonOverlappingCorrScores_onFullSet(
        largerArrs, smallerArrs, revCompFunc, N, excludeHitsWithinWindow, normaliseFunc=util.CROSSC_NORMFUNC.none, smallerPerPosNormFuncs=[], largerPerPosNormFuncs=[], auxLargerForPerPosNorm=None, auxLargerPerPosNormFuncs=[], palindromes={}, secondsBetweenUpdates=1, numThreads=1):
    """
        largerArrs: regions to get the corr scores on
        smallerArrs: regions to correlate with largerArrs
        revCompFunc: function for reverse complementation
        N, excludeHitsWithinWindow: see docs for
            getTopNGreedyNonOverlappingCorrScores
        Returns something of the following dimensions:
            [num examples x number of motifs x 5 x N (as in the "topN" scores; the first index is the highest score)]
            Regarding the third dimension which is of length 3, the indexes are as follows:
            Index 0 = the actual score
            Index 1 = left index of the hit if in fwd orientation, left index + motifLen if hit was in reverse orientation. (This adjustment is useful for grammar detection)
            Index 2 = 1 if hit was in forward orientation and -1 if hit was in reverse orientation
            Index 3 = left index of hit
            Index 4 = right index of hit
            The runtime scales linearly with N so I suggest setting N=1 if you can.
    """
    assert auxLargerForPerPosNorm is None or auxLargerForPerPosNorm.shape == largerArrs.shape
    startTime = time.time()
    _topNgreedy_largerArrs.var = largerArrs
    _topNgreedy_smallerArrs.var = smallerArrs
    _topNgreedy_revCompFunc.var = revCompFunc
    _topNgreedy_N.var = N
    _topNgreedy_excludeHitsWithinWindow.var = excludeHitsWithinWindow
    _topNgreedy_normaliseFunc.var = normaliseFunc
    _topNgreedy_smallerPerPosNormFuncs.var = smallerPerPosNormFuncs
    _topNgreedy_largerPerPosNormFuncs.var = largerPerPosNormFuncs
    _topNgreedy_auxLargerForPerPosNorm.var = auxLargerForPerPosNorm
    _topNgreedy_auxLargerPerPosNormFuncs.var = auxLargerPerPosNormFuncs
    _topNgreedy_palindromes.var = palindromes
    if (numThreads > 1):
        toReturn = util.multiprocessing_map_printProgress(
            secondsBetweenUpdates=secondsBetweenUpdates, numThreads=numThreads, func=getTopNGreedyNonOverlappingCorrScores_forParallel, iterable=range(len(largerArrs)))
    else:
        toReturn = []
        for i in range(len(largerArrs)):
            toReturn.append(
                getTopNGreedyNonOverlappingCorrScores_forParallel(i))
            if (i % 1000 == 0):
                print("Done", i)
    print("Time taken:", time.time() - startTime)
    return toReturn


def extractScoresOnlyFromHitsMatrix(hitsMatrix, topNtoKeep):
    """
        hitsMatrix has dimensions:
            numExamples x numMotifs x 2 x N
    """
    return np.array(hitsMatrix)[:, :, 0, :topNtoKeep]


class ReshapeCorrScoresInto2Dmatrix(object):

    def __init__(self, topNtoKeep):
        self.topNtoKeep = topNtoKeep

    def __call__(self, hitsMatrix):
        """
            hitsMatrix has dimensions:
                numExamples x numMotifs x 2 x N
            first index in third dimension corresponds to scores, the
                second index corresponds to the positions of the scores.
            Extract the first index and reshape into numExamples x (numMotifs*N)
        """
        raise NotImplementedError()


class ReshapeCorrScoresInto2Dmatrix_normalisePerMotif(ReshapeCorrScoresInto2Dmatrix):

    def __call__(self, hitsMatrix):
        """
            Normalise each motif's scores by the mean and standard deviation over all
                hits to that motif (even accross multiple ranks)
        """
        scoresOnly = extractScoresOnlyFromHitsMatrix(
            hitsMatrix, self.topNtoKeep)
        matrixToNormaliseByColumns =   np.transpose(scoresOnly, axes=(1, 0, 2))\
            .reshape((len(hitsMatrix[0]), -1))
        stdevPerMotif = np.std(matrixToNormaliseByColumns, axis=1)
        meanPerMotif = np.mean(matrixToNormaliseByColumns, axis=1)
        assert stdevPerMotif.shape == (len(hitsMatrix[0]),)
        # normalise the scores by mean and sdev
        scoresOnly = (
            scoresOnly - meanPerMotif[None, :, None]) / stdevPerMotif[None, :, None]
        #scoresOnly = (scoresOnly) / stdevPerMotif[None,:,None]
        # reshape into 2D matrix
        return np.reshape(scoresOnly, (scoresOnly.shape[0], scoresOnly.shape[1] * scoresOnly.shape[2]))


class ReshapeCorrScoresInto2Dmatrix_normaliseBySdevPerColumn(ReshapeCorrScoresInto2Dmatrix):

    def __call__(hitsMatrix):
        """
            Normalise by the sdev of each column, after the reshape
                to the 2D matrix.
        """
        scoresOnly = extractScoresOnlyFromHitsMatrix(
            hitsMatrix, self.topNtoKeep)
        # reshape into 2D matrix
        scoresOnly_reshaped = np.reshape(
            scoresOnly, (scoresOnly.shape[0], scoresOnly.shape[1] * scoresOnly.shape[2]))
        # normalise columns by mean + stdev
        return (scoresOnly_reshaped - np.mean(scoresOnly_reshaped, axis=1))\
            / np.std(scoresOnly_reshaped, axis=1)


def obtain2DscoresForAllLabelsSatisfying(motifHitsSets, datas, labelCriterion, twoDscoreGetterFunc):
    twoDscores = (twoDscoreGetterFunc(motifHits)
                  for motifHits in motifHitsSets)
    # subset the hits according to labelCriterion and concat
    twoDscores = np.array(list(itertools.chain(*[itertools.compress(hits, (labelCriterion(y) for y in data.Y))
                                                 for hits, data in zip(twoDscores, datas)])))
    ids = list(itertools.chain(*[itertools.compress(data.ids, (labelCriterion(y) for y in data.Y))
                                 for data in datas]))
    return twoDscores, ids


def getSeqletsConsideringFilterSubset(filterArrayOfContribs, rawSequenceArrayOfContribs, indexesOfFiltersToConsider, indicesToGetSeqletsOn, segmentIdentifier                                      # kernelWidthsAndStrideWidths:
                                      # first indices correspond to earlier layers.
                                      # If no earlier conv layers, is a list
                                      # with 1 tuple
                                      , kernelAndStrideWidths, includeNeg, numThreads, secondsBetweenUpdates, revCompFunc=None):
    """
        Is for identifying seqlets associated with specific filters
    """
    if (revCompFunc is None):
        revCompFunc = RevCompWithDNArowsSubset(dnaRowsStart=0, dnaRowsEnd=4)
        print("No reverse comp function provided so assuming you have dna as first 4 rows")

    #"filterArrayOf..." is the same as for getSeqlets. Has shape
    # example x channel x rows x len

    assert len(filterArrayOfContribs.shape) == 4
    assert filterArrayOfContribs.shape[2] == 1
    assert len(rawSequenceArrayOfContribs.shape) == 4
    assert rawSequenceArrayOfContribs.shape[1] == 1
    assert rawSequenceArrayOfContribs.shape[2] == 4
    # reshape to drop out the channel axis from the raw seq contribs
    reshapedRawSequenceContribs = np.squeeze(rawSequenceArrayOfContribs)
    if (includeNeg == False):
        # apply a mask for only positive contribs
        reshapedRawSequenceContribs = reshapedRawSequenceContribs *\
            (reshapedRawSequenceContribs > 0)

    # compute the effective filter width/stride in terms of raw sequence
    kernelAndStrideWidths = kernelAndStrideWidths[::-1]
    effectiveFilterWidth, effectiveFilterStride = kernelAndStrideWidths[0]
    for (kernWidPrevLyr, strideWidPrevLyr) in kernelAndStrideWidths[1:]:
        effectiveFilterWidth = kernWidPrevLyr + \
            (effectiveFilterWidth - 1) * strideWidPrevLyr
        effectiveFilterStride *= strideWidPrevLyr

    # subset to contributions from specific filters of interest
    filterArrayOfContribs = filterArrayOfContribs[
        :, indexesOfFiltersToConsider]

    # find sections of importance
    # filterKeySegments will hold the start and end index in terms of the
    # filter layer's length axis
    print("filterArrayOfContribs.shape", filterArrayOfContribs.shape)
    filterSeqlets, filterSeqletIndices = getSeqlets(
        # not needed if threshold is 1
        # not needed if threshold is 1
        # does not really matter for filters
        rawDeepLIFTContribs=filterArrayOfContribs, indicesToGetSeqletsOn=indicesToGetSeqletsOn, outputsBeforeActivation=None, activation=None, thresholdProb=1.0, segmentIdentifier=segmentIdentifier, revCompFunc=reverseFunc, includeNeg=includeNeg, numThreads=numThreads, secondsBetweenUpdates=secondsBetweenUpdates)
    sequenceSeqlets = []
    for filterSeqlet in filterSeqlets:
        (filterLocStart, filterLocEnd) = filterSeqlet.location
        (seqLocStart, seqLocEnd) = filterLocStart * effectiveFilterStride\
            , filterLocEnd * effectiveFilterStride\
            + effectiveFilterWidth
        assert seqLocStart < seqLocEnd
        assert seqLocEnd <= reshapedRawSequenceContribs.shape[2]
        summedDataTracks = {Grammar.coreDeepLIFTtrackName:
                            DataTrack(data=reshapedRawSequenceContribs
                                      [filterSeqlet.sequenceId, :, seqLocStart:seqLocEnd], pseudocount=0, revCompFunc=revCompFunc)}
        sequenceSeqlet = Seqlet(
            summedDataTracks=summedDataTracks, numUnderlyingObservations=1, totalObservationsEver=1, location=(seqLocStart, seqLocEnd), sequenceId=filterSeqlet.sequenceId)
        sequenceSeqlets.append(sequenceSeqlet)
    return sequenceSeqlets, filterSeqletIndices


def getTopFiltersByImportance(filterScores_forClustering, indicesSubset, topNFilters):
    summedFilterImportancesForIndices =\
        np.sum(np.array([filterScores_forClustering[x]
                         for x in indicesSubset]), axis=0)
    rankedFilterImportances = sorted(
        enumerate(summedFilterImportancesForIndices), key=lambda x: -x[1])
    indexesOfFiltersToConsider = [x[0]
                                  for x in rankedFilterImportances[:topNFilters]]
    return indexesOfFiltersToConsider


def getSeqletsForSpecificFilterSubsets(filtClustLabelToIndicesWithinClusteredArr, correspondingIndicesIntoValidArr, dLValidRawFilterContribs_singleNeuron, dLValidRawSequenceContribs_singleNeuron, filterScores_forClustering, segmentIdentifier, kernelAndStrideWidthsOfPrevLayers, revCompFunc, topNFilters=None, specificFilters=None):
    """
        filterScores_forClustering: 2d matrix of deepLIFT sores on true positives; for each
            region the scores for a particular filter are summed lengthwise
        filtClustLabelToIndicesWithinClusteredArr: dict from filter cluster label to indices within
            filterScores_forClustering
        returns: filtClustLabelToSeqletsAndIndices, which is a dict of
                    label -> (seqletsForFilterCluster, seqletIndices)
                    here seqlet indices refers to index within original valid set
    """
    # topNFilters and specificFilters are mutually exclusive options
    assert topNFilters is None or specificFilters is None
    assert topNFilters is not None or specificFilters is not None

    filtClustLabelToSeqletsAndIndices = OrderedDict()
    for filtClustLabel in sorted(filtClustLabelToIndicesWithinClusteredArr.keys()):
        indicesWithinClusteredArr = filtClustLabelToIndicesWithinClusteredArr[
            filtClustLabel]
        # map the index with respect to truePositiveIndices to the index into the
        # full validation set itself.
        indicesWithinValidArrForCluster = [correspondingIndicesIntoValidArr[x]
                                           for x in indicesWithinClusteredArr]
        topFilterIndices = specificFilters if specificFilters is not None\
            else getTopFiltersByImportance(
                filterScores_forClustering=filterScores_forClustering, indicesSubset=indicesWithinClusteredArr, topNFilters=topNFilters)
        print("Running on filters", topFilterIndices)

        # seqletIndices are into the original validation dataset array
        seqletsForFilterCluster, seqletIndices = getSeqletsConsideringFilterSubset(
            filterArrayOfContribs=dLValidRawFilterContribs_singleNeuron, rawSequenceArrayOfContribs=dLValidRawSequenceContribs_singleNeuron, indexesOfFiltersToConsider=topFilterIndices, indicesToGetSeqletsOn=indicesWithinValidArrForCluster, segmentIdentifier=segmentIdentifier, kernelAndStrideWidths=kernelAndStrideWidthsOfPrevLayers, revCompFunc=revCompFunc, numThreads=2, secondsBetweenUpdates=1
        )
        filtClustLabelToSeqletsAndIndices[filtClustLabel] = (
            seqletsForFilterCluster, seqletIndices)
    return filtClustLabelToSeqletsAndIndices

PairDistance = namedtuple("PairDistance", ["rowIdx", "hit1", "hit2", "sep"])


def obtainPairwiseDistancesBetweenHits(hitsForRows):
    pairwiseDistances = defaultdict(lambda: defaultdict(list))
    for (rowIdx, hitsForRow) in enumerate(hitsForRows):
        for (hit1Idx, hit1) in enumerate(hitsForRow):
            for (hit2Idx, hit2) in enumerate(hitsForRow[hit1Idx + 1:]):
                # the smaller hit will always come first due to the ordering of
                # hitsForRow
                if (hit1.grammarIdx == hit2.grammarIdx):
                    if (hit1.fwd == -1 and hit2.fwd == 1):
                        hit1, hit2 = hit2, hit1  # swap
                pairDistanceObject = PairDistance(
                    rowIdx=rowIdx, hit1=hit1, hit2=hit2, sep=hit2.pos - hit1.pos)
                pairwiseDistances[str(hit1.grammarIdx) + "_" + str(hit1.fwd)][str(
                    hit2.grammarIdx) + "_" + str(hit2.fwd)].append(pairDistanceObject)
    return pairwiseDistances


def obtainHitsForRows(topHitsForEachRow, zScoreThresholdForHit):
    topHitsForEachRow = topHitsForEachRow.copy()
    # topHitsForEachRow is:
    # numExamples x grammar x score,idx,fwd x top N hits
    assert len(topHitsForEachRow.shape) == 4
    meanPerGrammar = np.mean(topHitsForEachRow[:, :, 0, :], axis=(0, -1))
    stdPerGrammar = np.std(topHitsForEachRow[:, :, 0, :], axis=(0, -1))
    topHitsForEachRow[:, :, 0, :] =\
        (topHitsForEachRow[:, :, 0, :] - meanPerGrammar[None,
                                                        :, None]) / stdPerGrammar[None, :, None]
    for i in range(topHitsForEachRow.shape[1]):
        mplh.plotHist(np.ravel((topHitsForEachRow[:, 0, 0, :])), bins=50)
    hitsForRows = []
    for (topHitsRow) in topHitsForEachRow:
        hitsForRow = []
        for (grammarIdx, grammarHits) in enumerate(topHitsRow):
            for (grammarHitIdx, grammarHitScore) in enumerate(grammarHits[0]):
                if (grammarHitScore > zScoreThresholdForHit[grammarIdx]):
                    hitsForRow.append(Hit(grammarIdx=grammarIdx, score=grammarHitScore, pos=grammarHits[
                                      1][grammarHitIdx], fwd=grammarHits[2][grammarHitIdx]))
        hitsForRows.append(hitsForRow)
    return hitsForRows


def compareToKnownMotifs(mergedGrammars, trackNameForComparison=Grammar.coreDeepLIFTtrackName):
    import compare_filters_to_known_motifs
    reload(compare_filters_to_known_motifs)
    pwms = compare_filters_to_known_motifs.load_all_pwms()
    for (i, grammar) in mergedGrammars.items():
        print("We are on grammar", i, "\n")
        for pwmSet in pwms:
            hits = compare_filters_to_known_motifs.get_pwm_matches_for_filter(
                grammar.getNormalisedDataTrack(trackNameForComparison), pwmSet)
            print(hits)
            print("")

ConvLayerTypes = util.enum(conv='conv', maxpool='maxpool')
ConvLayerDetails = namedtuple("ConvLayerDetails", [
                              "layerType", "stride", "width", "weights", "biases", "activation"])


def kerasLayerToConvLayerDetails(kerasLayer, activation):
    config = kerasLayer.get_config()
    if config['custom_name'] == 'convolution2d':
        return ConvLayerDetails(
            layerType=ConvLayerTypes.conv, stride=config['subsample'][1], width=config['nb_col'], weights=kerasLayer.get_weights()[0], biases=kerasLayer.get_weights()[1], activation=activation)
    else:
        raise RuntimeError("Unsupported custom_name: " +
                           str(config['custom_name']))


def determineFinalLayerEffectiveStrideAndWidth(convLayersDetails):
    finalEffectiveStride = convLayersDetails[0].stride
    finalEffectiveWidth = convLayersDetails[0].width
    for convLayerDetails in convLayersDetails[1:]:
        finalEffectiveWidth = finalEffectiveWidth + \
            (convLayerDetails.width - 1) * finalEffectiveStride
        finalEffectiveStride *= convLayerDetails.stride
    return finalEffectiveStride, finalEffectiveWidth


def compileMiniKerasModel(convLayersDetails, inputSize, finalOutputWeights=None):
    assert len(convLayersDetails) == 1  # only handling 1 for now
    # only stride 1 for now
    assert all([x.stride == 1 for x in convLayersDetails])
    assert convLayersDetails[0].weights.shape[1] == 1  # input has 1 channel
    assert convLayersDetails[0].weights.shape[2] == 4  # input has 4 rows

    from keras.models import Sequential
    from keras.layers.core import Flatten
    from keras.layers.core import Dense

    model = Sequential()
    for (i, convLayerDetails) in enumerate(convLayersDetails):
        assert int(inputSize) == inputSize
        input_shape = (1, 4, int(inputSize)) if i == 0 else None
        convLayer = createKerasConvLayerFromConvLayerDetails(
            convLayerDetails, input_shape=input_shape)
        model.add(convLayer)
    if (finalOutputWeights is not None):
        model.add(Flatten())
        denseLayer = Dense(1, activation="linear")
        model.add(denseLayer)
    model.compile(loss='mse', optimizer='sgd')
    for (i, convLayerDetails) in enumerate(convLayersDetails):
        model.layers[i].set_weights(
            [convLayerDetails.weights, convLayerDetails.biases])
    model.layers[-1].set_weights([finalOutputWeights.ravel()
                                  [:, None], np.array([0])])
    return model


def createKerasConvLayerFromConvLayerDetails(convLayerDetails, input_shape=None):
    from keras.layers.convolutional import Convolution2D
    convLayer = Convolution2D(nb_filter=convLayerDetails.weights.shape[0]                              # weights shape = num channels x num channels of previous layer
                              #                x kernel width x kernel height
                              , nb_row=convLayerDetails.weights.shape[2], nb_col=convLayerDetails.width, activation=convLayerDetails.activation, **{} if input_shape is None else {'input_shape': input_shape})
    return convLayer


def agglomerative_clustering(grammars_list,
                             gradient_track,
                             cc_threshold,
                             trimming_func):
    indices_list = [set([i]) for i in range(len(grammars_list))]
    while True:
        # meanAndTwoNorm used for ensuring that the corrrelations are
        # between -1 and 1
        grammars_cc = getCorrelationMatrix(
            grammars_list,
            subtracksToInclude=[gradient_track],
            subtrackNormaliseFunc=util.CROSSC_NORMFUNC.meanAndTwoNorm,
            # smallerPerPosNormFuncs=[util.PERPOS_NORMFUNC.oneOverTwoNorm],
            # largerPerPosNormFuncs=[util.PERPOS_NORMFUNC.oneOverTwoNorm],
            accountForRevComp=True,
            numThreads=1,
            secondsBetweenUpdates=6,
            xcorBatchSize=None)

        np.fill_diagonal(grammars_cc, 0)
        max_grammars_cc = np.max(grammars_cc)

        print("max cc: ", max_grammars_cc)
        if max_grammars_cc < cc_threshold:
            break

        max_cc_idx1, max_cc_idx2 = \
            np.unravel_index(np.argmax(grammars_cc), grammars_cc.shape)

        merged_grammar = grammars_list[max_cc_idx1].merge(
            grammars_list[max_cc_idx2],
            subtracksToInclude=[gradient_track],
            subtrackNormaliseFunc=util.CROSSC_NORMFUNC.meanAndTwoNorm,
            # subtrackNormaliseFunc=util.CROSSC_NORMFUNC.meanAndSdev,
            normaliseFunc=util.CROSSC_NORMFUNC.none,
            smallerPerPosNormFuncs=[],
            largerPerPosNormFuncs=[],
            # smallerPerPosNormFuncs=[util.PERPOS_NORMFUNC.oneOverTwoNorm],
            # largerPerPosNormFuncs=[util.PERPOS_NORMFUNC.oneOverTwoNorm],
            revComp=True)

        removeset = {max_cc_idx1, max_cc_idx2}
        merged_indices = \
            indices_list[max_cc_idx1].union(indices_list[max_cc_idx2])
        indices_list = [indices_set
                        for i, indices_set in enumerate(indices_list)
                        if i not in removeset]
        indices_list.append(merged_indices)
        grammars_list = [grammar
                         for i, grammar in enumerate(grammars_list)
                         if i not in removeset]
        grammars_list.append(adjustGrammarUsingTrimmingCriterion(
            merged_grammar,
            trimmingFunc=trimming_func))
    return grammars_list, indices_list


def mergeAndPrintMotifs(seqlets, labels, numObsFraction, mergingSubtracks, subtracksToPrint, printRevComp=True):
    # The trimming function is optional; it is used to further trim uninformative flanks.
    # TrimArrayColumnsToPercent trims the grammar to the smallest subsequence that has "percent" importance
    # of the original full sequence
    #trimmingFunc = TrimArrayColumnsToPercent(percent=0.95)
    # TrimArrayColumsnToNumUnderlyingObs resticts attention to those positions in the grammar
    # that have at least 20% of the total observations for the grammar
    # supporting them.
    trimmingFunc = TrimArrayColumnsToNumUnderlyingObs(numObsFraction)
    # once again, subtracksToInclude indicates the subtracks to consider for merging. Should be
    # the same as what you supplied for the cross-correlation
    mergedMotifs = createMergedGrammars(
        labels, seqlets, subtracksToInclude=mergingSubtracks, accountForRevComp=True)
    mergedMotifs = adjustGrammarsUsingTrimmingCriterion(
        mergedMotifs, trimmingFunc=trimmingFunc)
    for (motifIdx) in sorted(mergedMotifs.keys()):
        print("index", motifIdx)
        motif = mergedMotifs[motifIdx]
        print("total observations", motif.totalObservationsEver)
        print("fwd")
        printGrammar(motif, trackNamesToPrint=subtracksToPrint)
        if (printRevComp):
            print("rev")
            printGrammar(motif.getRevCompGrammar(),
                         trackNamesToPrint=subtracksToPrint)
    return mergedMotifs
