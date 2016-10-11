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

    def __init__(self, feature_beds=None, region_beds=None,
                 regions=None, labels=None,
                 dnase_bigwig=None, genome_fasta=None,
                 dnase_data_dir=None, genome_data_dir=None):
        self.feature_beds = feature_beds
        self.region_beds = region_beds
        self.regions = regions
        self.labels = labels
        self.dnase_bigwig = dnase_bigwig
        self.genome_fasta = genome_fasta
        self.dnase_data_dir = dnase_data_dir
        self.genome_data_dir = genome_data_dir
        # check that it doesn't use invalid combinations of data
        if self.has_feature_beds_and_regions_or_labels:
            raise ValueError("Invalid dataset: includes features beds and regions and/or labels!")
        if self.has_raw_and_encoded_dnase:
            raise ValueError("Invalid dataset: includes raw and encoded dnase!")
        if self.has_raw_and_encoded_fasta:
            raise ValueError("Invalid dataset: includes raw and encoded fasta!")

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
    data = json.load(open(data_config_file))
    for dataset_id, dataset in data.items():
        data[dataset_id] = Dataset(**dataset)

    return data
