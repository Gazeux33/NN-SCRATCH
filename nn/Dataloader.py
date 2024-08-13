class Dataloader:
    def __init__(self, dataset, targets=None, batch_size: int = 32):
        self.targets = targets
        if self.targets is not None:
            if len(dataset) != len(targets):
                raise ValueError("Dataset and target must have the same length")
        self.dataset = dataset
        self.batch_size = batch_size
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration
        batch_x = self.dataset[self.index:self.index + self.batch_size]
        if self.targets is not None:
            batch_y = self.targets[self.index:self.index + self.batch_size]
            batch_y = batch_y.reshape(-1, 1)
            self.index += self.batch_size
            return batch_x, batch_y
        self.index += self.batch_size
        return batch_x

    def __len__(self):
        return len(self.dataset) // self.batch_size
