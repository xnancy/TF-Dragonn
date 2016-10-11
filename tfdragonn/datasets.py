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
    def is_memmaped(self):
        return all([self.dnase_data_dir is not None or self.genome_data_dir is not None,
                    self.dnase_bigwig is None,
                    self.genome_fasta is None])

    def __init__(self, feature_beds=None, region_bed=None,
                 regions=None, labels=None,
                 dnase_bigwig=None, genome_fasta=None,
                 dnase_data_dir=None, genome_data_dir=None):
        self.feature_beds = feature_beds
        self.region_bed = region_beds
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


def parse_data_config_file(data_config_file):
    """
    Parses data config file and returns region beds, feature beds, and data files.
    """
    data = json.load(open(data_config_file), object_pairs_hook=collections.OrderedDict)
    for dataset_id, dataset in data.items():
        data[dataset_id] = Dataset(**dataset)

    return data


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
        super(IntervalDataset, self).__init__(self, **kwargs)
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
        super(LabeledIntervalDataset, self).__init__(self, **kwargs)
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
        super(OrderedLabeledIntervalDataset, self).__init__(self, **kwargs)
        if not has_ordered_labels:
            raise ValueError("Invalid OrderedLabeledIntervalDataset: must have either labels for regions or feature_beds list!")


class Datasets(object):

    @property
    def has_consistent_datasets(self):
        """
        Checks datasets are of the same type.
        """
        return all(type(dataset) == self.dataset_type for dataset in self.datasets)

    def __init__(self, dataset_dict, task_names=None):
        self.dataset_ids = dataset_dict.keys()
        self.datasets = dataset_dict.items()
        self.task_names = task_names
        self.dataset_type = type(datasets[0])
        if not self.has_consistent_datasets:
            raise ValueError("Datasets are inconsistent: multiple dataset types are not allowed in the same config file!")
        if self.dataset_type is OrderedLabeledIntervalDataset:
            self.check_ordered_labeled_interval_datasets()
        elif self.dataset_type is LabeledIntervalDataset:
            self.check_nonordered_labeled_interval_datasets()

    def check_ordered_labeled_interval_datasets(self):
        pass

    def check_nonordered_labeled_interval_datasets(self):
        assert type(self.task_names) is list, "task_names list is required when feature_beds are provided as a dictionary!"
        task_names = set(self.task_names)
        assert len(task_names) == len(self.task_names), "task names must be unique!"
        for dataset_id, dataset in zip(self.dataset_ids, self.datasets):
            assert type(dataset) is dict
            dataset_task_names = set(dataset.keys())
            assert dataset_task_names,issubset(task_names), "Tasks in {} are not a subset of task_names!".format(dataset_id)
