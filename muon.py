import torch

@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                momentum_buffer = state['momentum_buffer']
                p.mul_(1 - group['weight_decay'])
                momentum_buffer.lerp_(grad, 1 - momentum)
                grad = grad.lerp_(momentum_buffer, momentum)
                v = zeropower_via_newtonschulz5(grad.bfloat16(), 5)
                p.add_(other=v, alpha=-group['lr'])
