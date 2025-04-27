from torchmetrics.image.fid import FrechetInceptionDistance


class FIDWrapper:
    def __init__(self, device):
        self.fid_metric = FrechetInceptionDistance(normalize=True).to(
            device=device, non_blocking=True
        )

    def update(self, samples, real=True):
        if samples.shape[1] == 1:
            # Expand grayscale to RGB
            samples = samples.repeat(1, 3, 1, 1)
        self.fid_metric.update(samples, real=real)

    def compute(self):
        return self.fid_metric.compute()

    def reset(self):
        self.fid_metric.reset()
