import math

import torch

class DistributedSampler(torch.utils.data.distributed.Sampler):

    def __init__(self, dataset, shuffle=True, seed=0):
        if not torch.distributed.is_initialized():
            raise Exception("Distributed sampler can only be used in a distributed environment.")

        self.dataset = dataset
        self.num_replicas = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.num_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.random_gen = torch.Generator()
        
        if seed is not None:
            self.random_gen.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), self.random_gen).cuda()

            torch.distributed.broadcast(indices, 0)
            indices = indices.cpu().tolist()
        else:
            indices = list(range(len(self.dataset)))

        indices += indices[:(self.total_size - len(indices))]
        indices = indices[self.rank:self.total_size:self.num_replicas]
        
        assert len(indices) == self.num_samples, "{} != {}".format(len(indices), self.num_samples)

        return iter(indices)

    def __len__(self):
        return self.num_samples


def create_sampler(dataset, shuffle=True, seed=None):
    """Returns a DistributedSampler if running in distributed mode, otherwise a normal sampler

    Args:
        dataset (torch.utils.data.Dataset): The dataset the sampler will work on.
        shuffle (bool): If the sampler should be random or not.
    """

    if torch.distributed.is_initialized():
        return BetterDistributedSampler(
            dataset=dataset, 
            shuffle=shuffle,
            seed=seed
        )

    if shuffle:
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        return torch.utils.data.RandomSampler(data_source=dataset, generator=g)
    
    return torch.utils.data.SequentialSampler(data_source=dataset)


class BetterDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    """This class extends torch's default DistributedSampler but removes the need
    for manually calling the set_epoch method to reseed the random sampler
    """
    def __init__(self, dataset, shuffle=True, seed=None):
        super().__init__(dataset, shuffle=shuffle, seed=seed)
        self.epoch = 0

    def __iter__(self):
        self.set_epoch(self.epoch+1)
        return super().__iter__()


class BetterDataLoader(torch.utils.data.DataLoader):
    """Implements a dataloader data works consistently in both distributed and
    nondistributed runtimes.
    """
    def __init__(self, dataset, shuffle=False, random_seed=None, **kwargs):
        super().__init__(dataset, sampler=create_sampler(dataset, shuffle, random_seed), **kwargs)


