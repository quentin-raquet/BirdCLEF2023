import torch
from torchmetrics.classification import MultilabelAveragePrecision


class PaddedCMAP(MultilabelAveragePrecision):
    def __init__(self, num_labels: int, padding_factor: int = 5):
        super().__init__(num_labels=num_labels)
        self.padding_factor = padding_factor

    def compute(self):
        if isinstance(self.preds, torch.Tensor):
            self.preds = torch.cat(
                [
                    self.preds,
                    torch.ones(
                        self.padding_factor, self.num_labels, device=self.device
                    ),
                ]
            )
            self.target = torch.cat(
                [
                    self.target,
                    torch.ones(
                        self.padding_factor, self.num_labels, device=self.device
                    ),
                ]
            )
        elif isinstance(self.preds, list):
            self.preds = [
                torch.cat(
                    [
                        preds,
                        torch.ones(
                            self.padding_factor, self.num_labels, device=self.device
                        ),
                    ]
                )
                for preds in self.preds
            ]
            self.target = [
                torch.cat(
                    [
                        t,
                        torch.ones(
                            self.padding_factor, self.num_labels, device=self.device
                        ),
                    ]
                )
                for t in self.target
            ]

        return super().compute()
