import torch
import torch.utils.data
import torchvision
import pdb
from fer_strong_baseline.datasets.ImbalancedDatasetSampling.ImbalancedRAF_DB import ImbalancedRAF_DB_Dataset
from fer_strong_baseline.datasets.RAF_DB import RafDataSet
from fer_strong_baseline.datasets.AffectNet import AffectNet_Dataset
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from ImbalancedDatasetSampling given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): ImbalancedDatasetSampling list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            # spdb.set_trace()
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        # pdb.set_trace()
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is ImbalancedRAF_DB_Dataset:
            pdb.set_trace()
            return dataset.imgs_first[idx][1]
        elif dataset_type is RafDataSet:
            return dataset.labels[idx]
        elif dataset_type is AffectNet_Dataset:
            return dataset.labels[idx]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples