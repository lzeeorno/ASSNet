import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.optim.sgd import SGD
from torch.optim import lr_scheduler
from warmup_scheduler.scheduler import GradualWarmupScheduler


if __name__ == '__main__':
    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optim = torch.optim.SGD(model, lr=0.01, momentum=0.98, weight_decay=3e-5, nesterov=True)

    # scheduler_warmup is chained with schduler_steplr
    scheduler_steplr = lr_scheduler.CosineAnnealingLR(optim, T_max=100, eta_min=1e-5)

    scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1, total_epoch=50, after_scheduler=scheduler_steplr)

    # this zero gradient update is needed to avoid a warning message, issue #8.
    optim.zero_grad()
    optim.step()

    for epoch in range(1, 1000):
        scheduler_warmup.step(epoch)
        print(epoch, optim.param_groups[0]['lr'])

        optim.step()    # backward pass (update network)