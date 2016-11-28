from abc import abstractmethod, ABCMeta
from builtins import zip
import collections
import inspect
import json

from genomedatalayer.extractors import (
    MemmappedBigwigExtractor, MemmappedFastaExtractor
)

class Dataset(object):

    @property
    def has_feature_beds_and_regions_or_labels(self):
        return ((self.regions is not None or self.labels is not None) and
                self.feature_beds is not None)

    @property
    def has_raw_and_encoded_dnase(self):
        return self.dnase_bigwig is not None and self.dnase_data_dir is not None

    @property
    def has_raw_and_encoded_fasta(self):
        return self.genome_fasta is not None and self.genome_data_dir is not None

    @property
    def memmaped(self):
        return all([self.dnase_data_dir is not None or self.genome_data_dir is not None,
                    self.dnase_bigwig is None,
                    self.genome_fasta is None])

    @property
    def memmaped_fasta(self):
        return self.genome_data_dir is not None

    @property
    def memmaped_dnase(self):
        return self.dnase_data_dir is not None

    def __init__(self, feature_beds=None,
                 ambiguous_feature_beds=None, region_bed=None,
                 regions=None, labels=None,
                 dnase_bigwig=None, genome_fasta=None,
                 dnase_data_dir=None, genome_data_dir=None):
        self.feature_beds = feature_beds
        self.ambiguous_feature_beds = ambiguous_feature_beds
        self.region_bed = region_bed
        self.regions = regions
        self.labels = labels
        self.dnase_bigwig = dnase_bigwig
        self.genome_fasta = genome_fasta
        self.dnase_data_dir = dnase_data_dir
        self.genome_data_dir = genome_data_dir
        # check that it doesn't use invalid combinations of data
        if self.has_feature_beds_and_regions_or_labels:
            raise ValueError("Invalid Dataset: includes features beds and regions and/or labels!")
        if self.has_raw_and_encoded_dnase:
            raise ValueError("Invalid Dataset: includes raw and encoded dnase!")
        if self.has_raw_and_encoded_fasta:
            raise ValueError("Invalid Dataset: includes raw and encoded fasta!")

    def to_dict(self):
        """
        return dictionary with class attribute names and values.
        """
        return {key:value for key, value in self.__dict__.items()
                if not key.startswith('__') and not callable(key)}


class IntervalDataset(Dataset):
    """
    A Dataset with raw or processed interval files.
    """
    @property
    def has_valid_intervals(self): # cached intervals xor raw intervals
        return (self.regions is not None) != (self.feature_beds is not None or self.region_bed is not None)

    @property
    def has_cached_intervals(self):
        return self.regions is not None

    def __init__(self, **kwargs):
        super(IntervalDataset, self).__init__(**kwargs)
        if not self.has_valid_intervals:
            raise ValueError("Invalid IntervalDataset: must have either cached intervals or raw intervals!")

class LabeledIntervalDataset(IntervalDataset):
    """
    A Dataset with raw or processed interval+label files.
    """
    @property
    def has_valid_labels(self): # cached xor processed labels
        if self.has_cached_intervals:
            return self.labels is not None
        else:
            return self.feature_beds is not None

    def __init__(self, **kwargs):
        super(LabeledIntervalDataset, self).__init__(**kwargs)
        if not self.has_valid_labels:
            raise ValueError("Invalid LabeledIntervalDataset: must have either labels for regions or feature_beds!")

class OrderedLabeledIntervalDataset(LabeledIntervalDataset):
    """
    A LabeledIntervalDataset without use of dictionary of interval files.
    """
    @property
    def has_ordered_labels(self):
        return self.labels is not None or isinstance(self.feature_beds, list)

    def __init__(self, **kwargs):
        super(OrderedLabeledIntervalDataset, self).__init__(**kwargs)
        if not self.has_ordered_labels:
            raise ValueError("Invalid OrderedLabeledIntervalDataset: must have either labels for regions or feature_beds list!")


class Datasets(object):

    def __iter__(self):
        return zip(self.dataset_ids, self.datasets)

    @property
    def has_consistent_datasets(self):
        """
        Checks datasets are of the same type.
        """
        return all(type(dataset) == self.dataset_type for dataset in self.datasets)

    @property
    def has_task_names(self):
        return type(self.task_names) is list

    @property
    def include_regions(self):
        return all(dataset.regions is not None for dataset in self.datasets)

    @property
    def include_regions_or_region_bed(self):
        return all(dataset.region_bed is not None or dataset.regions is not None for dataset in self.datasets)

    @property
    def include_labels(self):
        return all(dataset.labels is not None for dataset in self.datasets)

    @property
    def memmaped(self):
        return all(dataset.memmaped for dataset in self.datasets)

    @property
    def memmaped_fasta(self):
        return all(dataset.genome_data_dir is not None for dataset in self.datasets)

    @property
    def memmaped_dnase(self):
        return all(dataset.dnase_data_dir is not None for dataset in self.datasets)

    def __init__(self, dataset_dict, task_names=None):
        self.dataset_ids = dataset_dict.keys()
        self.datasets = dataset_dict.values()
        self.task_names = task_names
        self.dataset_type = type(self.datasets[0])
        if not self.has_consistent_datasets:
            raise ValueError("Datasets are inconsistent: multiple dataset types are not allowed in the same config file!")
        if self.dataset_type is OrderedLabeledIntervalDataset:
            self.check_ordered_labeled_interval_datasets()
        elif self.dataset_type is LabeledIntervalDataset:
            self.check_nonordered_labeled_interval_datasets()
            self.convert_to_ordered_labeled_interval_datasets()

    def check_ordered_labeled_interval_datasets(self):
        """
        checks number of tasks is consistent across datasets.
        """
        if self.has_task_names:
            pass
        else:
            pass

    def check_nonordered_labeled_interval_datasets(self):
        """
        checks a list of task names is provided and task names in each dataset are a subset of that master list.
        """
        assert self.has_task_names, "task_names list is required when feature_beds are provided as a dictionary!"
        task_names = set(self.task_names)
        assert len(task_names) == len(self.task_names), "task names must be unique!"
        for dataset_id, dataset in self:
            assert type(dataset.feature_beds) is collections.OrderedDict, "feature beds in dataset {} are not a dictionary:\n{}".format(dataset_id, dataset.feature_beds)
            dataset_task_names = set(dataset.feature_beds.keys())
            assert dataset_task_names.issubset(task_names), "Tasks {} in {} are not in task_names!".format(dataset_task_names - task_names, dataset_id)
            if dataset.ambiguous_feature_beds is not None:
                assert type(dataset.ambiguous_feature_beds) is collections.OrderedDict, "ambiguous feature beds in dataset {} are not a dictionary:\n{}".format(dataset_id, dataset.ambiguous_feature_beds)
                dataset_ambiguous_task_names = set(dataset.ambiguous_feature_beds.keys())
                assert dataset_ambiguous_task_names.issubset(task_names), "Tasks {} in {} are not in task_names!".format(dataset_ambiguous_task_names - task_names, dataset_id)

    def convert_to_ordered_labeled_interval_datasets(self):
        """
        Converts dictionaries of feature beds to uniformly ordered lists of feature beds.
        """
        for i, (dataset_id, dataset) in enumerate(self):
            feature_beds_list = []
            for task_name in self.task_names:
                feature_beds_list.append(dataset.feature_beds[task_name] if task_name in dataset.feature_beds.keys() else None)
            self.datasets[i].feature_beds = feature_beds_list
            if dataset.ambiguous_feature_beds is not None:
                ambiguous_feature_beds_list = []
                for task_name in self.task_names:
                    ambiguous_feature_beds_list.append(dataset.ambiguous_feature_beds[task_name] if task_name in dataset.ambiguous_feature_beds.keys() else None)
                self.datasets[i].ambiguous_feature_beds = ambiguous_feature_beds_list
            else:
                self.datasets[i].ambiguous_feature_beds = None

    def to_dict(self):
        datasets_dict = collections.OrderedDict()
        datasets_dict["task_names"] = self.task_names
        for dataset_id, dataset in self:
            datasets_dict[dataset_id] = dataset.to_dict()

        return datasets_dict

    def get_dataset2extractors(self, local_norm_halfwidth=None):
        if self.memmaped:
            sequence_input = False
            dnase_input = False
            dataset2extractors = {dataset_id: [] for dataset_id, _ in self}
            if self.memmaped_fasta:
                sequence_input = True
                # write non redundant mapping from data_dir to extractor
                genome_data_dir2fasta_extractor = {}
                for dataset_id, dataset in self:
                    if dataset.genome_data_dir not in genome_data_dir2fasta_extractor:
                        genome_data_dir2fasta_extractor[dataset.genome_data_dir] = MemmappedFastaExtractor(dataset.genome_data_dir)
                    dataset2extractors[dataset_id].append(genome_data_dir2fasta_extractor[dataset.genome_data_dir])
            if self.memmaped_dnase:
                dnase_input = True
                # write non redundant mapping from data_dir to extractor
                dnase_data_dir2bigwig_extractor = {}
                for dataset_id, dataset in self:
                    if dataset.dnase_data_dir not in dnase_data_dir2bigwig_extractor:
                        dnase_data_dir2bigwig_extractor[dataset.dnase_data_dir] = MemmappedBigwigExtractor(
                            dataset.dnase_data_dir, local_norm_halfwidth=local_norm_halfwidth)
                    dataset2extractors[dataset_id].append(dnase_data_dir2bigwig_extractor[dataset.dnase_data_dir])
        else:
            raise RuntimeError("get_dataset2extractors doesn't support non memmaped inputs!")

        return dataset2extractors


def parse_data_config_file(data_config_file):
    """
    Parses data config file and returns region beds, feature beds, and data files.
    """
    data = json.load(open(data_config_file), object_pairs_hook=collections.OrderedDict)
    task_names = None
    for dataset_id, dataset in data.items():
        if dataset_id == "task_names":
            task_names = dataset
            del data["task_names"]
            continue
        # initialize the appropriate type of Dataset
        try:
            data[dataset_id] = OrderedLabeledIntervalDataset(**dataset)
        except ValueError:
            try:
                data[dataset_id] = LabeledIntervalDataset(**dataset)
            except ValueError:
                try:
                    data[dataset_id] = IntervalDataset(**dataset)
                except ValueError:
                    data[dataset_id] = Dataset(**dataset)

    return Datasets(data, task_names)


class RawInputs(object):
    @property
    def has_dnase_bigwig(self):
        return self.dnase_bigwig is not None

    @property
    def has_genome_fasta(self):
        return self.genome_fasta is not None

    def __init__(self, dnase_bigwig=None, genome_fasta=None):
        self.dnase_bigwig = dnase_bigwig
        self.genome_fasta = genome_fasta


class ProcessedInputs(object):
    @property
    def has_dnase_data_dir(self):
        return self.dnase_data_dir is not None

    @property
    def has_genome_data_dir(self):
        return self.genome_data_dir is not None

    def __init__(self, dnase_data_dir=None, genome_data_dir=None):
        self.dnase_data_dir = dnase_data_dir
        self.genome_data_dir = genome_data_dir

raw_input2processed_input = {"dnase_bigwig": "dnase_data_dir",
                             "genome_fasta": "genome_data_dir"}

def check_property_consistency(config, target_property):
    dataset_id1, dataset_class1 = next(config.__iter__())
    expected_property_value = getattr(dataset_class1, target_property)
    for dataset_id2, dataset_class2 in config:
        if getattr(dataset_class2, target_property) == expected_property_value:
            pass
        else:
            return (dataset_id1, dataset_id2)
    return True


class InputsConfig(object):
     __metaclass__ = ABCMeta

     @abstractmethod
     def __iter__(self):
         pass

     @abstractmethod
     def __init__(self):
         pass

     def check_property_consistency(self, target_property):
         dataset_id1, dataset_class1 = next(self.__iter__())
         expected_property_value = getattr(dataset_class1, target_property)
         for dataset_id2, dataset_class2 in self:
             if getattr(dataset_class2, target_property) == expected_property_value:
                 pass
             else:
                 return (dataset_id1, dataset_id2)
             return True


class RawInputsConfig(InputsConfig):
    """
    A class for the raw inputs config file.

    Parameters
    ----------
    dataset_id2raw_inputs : dict
        keys are data set id strigs, values are instances of RawInputs.
    """
    def __init__(self, dataset_id2raw_inputs):
        self.dataset_ids = dataset_id2raw_inputs.keys()
        self.raw_inputs_list = dataset_id2raw_inputs.values()
        # check for consistent raw inputs
        rv = check_property_consistency(self, "has_genome_fasta")
        if rv is True:
            pass
        else:
            dataset_id1, dataset_id2 = rv
            raise ValueError("Datasets {} and {} have inconsistent genome fasta types!".format(dataset_id1, dataset_id2))
        rv = self.check_property_consistency("has_dnase_bigwig")
        if rv is True:
            pass
        else:
            dataset_id1, dataset_id2 = rv
            raise ValueError("Datasets {} and {} have inconsistent dnase bigwig types!".format(dataset_id1, dataset_id2))

    def __iter__(self):
        return zip(self.dataset_ids, self.raw_inputs_list)


class ProcessedInputsConfig(InputsConfig):
    """
    A class for the processed inputs config file.

    Parameters
    ----------
    dataset_id2processed_inputs : dict
        keys are data set id strigs, values are instances of ProcessedInputs.
    """
    def __init__(self, dataset_id2processed_inputs):
        self.dataset_ids = dataset_id2processed_inputs.keys()
        self.processed_inputs_list = dataset_id2processed_inputs.values()
        # check for consistent processed inputs
        rv = self.check_property_consistency( "has_genome_data_dir")
        if rv is True:
            pass
        else:
            dataset_id1, dataset_id2 = rv
            raise ValueError("Datasets {} and {} have inconsistent genome data dir types!".format(dataset_id1, dataset_id2))
        rv = check_property_consistency(self, "has_dnase_data_dir")
        if rv is True:
            pass
        else:
            dataset_id1, dataset_id2 = rv
            raise ValueError("Datasets {} and {} have inconsistent dnase data dir types!".format(dataset_id1, dataset_id2))

    def __iter__(self):
        return zip(self.dataset_ids, self.raw_inputs_list)


def parse_raw_input_config_file(raw_input_config_fname):
    """
    Returns instance of RawInputsConfig
    """
    dataset_id2raw_inputs = json.load(open(raw_input_config_fname), object_pairs_hook=collections.OrderedDict)
    for dataset_id, raw_inputs in dataset_id2raw_inputs.items():
        dataset_id2raw_inputs[dataset_id] = RawInputs(**raw_inputs)

    return RawInputsConfig(dataset_id2raw_inputs)


def parse_processed_input_config_file(processed_input_config_fname):
    """
    Returns instance of RawInputsConfig
    """
    dataset_id2processed_inputs = json.load(open(processed_input_config_fname), object_pairs_hook=collections.OrderedDict)
    for dataset_id, processed_inputs in dataset_id2processed_inputs.items():
        dataset_id2raw_inputs[dataset_id] = ProcessedInputs(**processed_inputs)

    return ProcessedInputsConfig(dataset_id2raw_inputs)
