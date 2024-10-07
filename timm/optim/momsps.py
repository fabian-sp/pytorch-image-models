import torch
import warnings

from .types import Params, LossClosure, OptFloat

class MomSPS(torch.optim.Optimizer):
    def __init__(self,
                 params: Params,
                 lr: float=1.0,
                 c: float=0.4,
                 beta: float=0.9,
                 weight_decay: float=0.0,
                 lb: float=0.0,
                 ) -> None:

        defaults = dict(lr=lr,
                        c=c,
                        beta=beta, 
                        weight_decay=weight_decay
        )

        super(MomSPS, self).__init__(params, defaults)
        
        self.lb = lb

        # Initialization
        self.number_steps = 0
        self.state['step_size_list'] = list()

        return
    
    def step(self, closure: LossClosure=None, loss: torch.Tensor=None, lb: torch.Tensor=None) -> OptFloat:
        assert (closure is not None) or (loss is not None), "Either loss tensor or closure must be passed."
        assert (closure is None) or (loss is None), "Pass either the loss tensor or the closure, not both."

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.number_steps += 1
        grad_norm = self.compute_grad_terms()
        
        # use given batch lower bound or not; NOT USED YET
        _this_lb = lb.item() if lb else self.lb

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta'] 
            weight_decay = group['weight_decay']  # NOT USED YET
            c = group["c"]

            for p in group['params']:
                # new_p = x^{t+1}
                # p = x^t
                # old_p = x^{t-1}
                grad = p.grad.data.detach()
                state = self.state[p]
                _momsps_step = (loss/(c*grad_norm**2)).item()
                eta = (1-beta) * min(_momsps_step, lr)

                if self.number_steps == 1:
                    state["p"] = p.detach().clone()
                    new_p = p - eta * grad
                else:
                    old_p = state["p"]
                    state["p"] = p.detach().clone()
                    new_p = p - eta * grad + beta * (p-old_p)

                with torch.no_grad():
                    p.copy_(new_p)
        
        self.state['step_size_list'].append(_momsps_step)
        return loss
    
    @torch.no_grad()   
    def compute_grad_terms(self):
        grad_norm = 0.
        for group in self.param_groups:
            for p in group['params']:
                g = p.grad.data.detach()
                grad_norm += torch.sum(torch.mul(g, g))

        grad_norm = torch.sqrt(grad_norm)
        return grad_norm