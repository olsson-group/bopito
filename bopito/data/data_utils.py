import torch

def get_sample_batch(dataset, batch_size):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batch = next(iter(dataloader))
    return batch

def get_sample_batch_1d(dataset, batch_size):
    indices = torch.randint(high=len(dataset), size=(batch_size,1)).squeeze(dim=1)
    return dataset[indices]
    
def generate_batches(dataset, batch_size):
    indices = torch.randperm(len(dataset)).type(torch.int)
    
    indices = indices.split(batch_size)
    batches = []
    for indices_batch in indices:
        batch = dataset[indices_batch]
        batches.append(batch)
    return batches