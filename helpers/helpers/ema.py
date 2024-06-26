from copy import deepcopy

import torch

def update_ema(msd, esd, decay=0.999):
    # param_keys = [k for k, _ in tmodel.named_parameters()]
    # buffer_keys = [k for k, _ in tmodel.named_buffers()]
    # needs_module = hasattr(tmodel, 'module') and not self.ema_has_module
    # 全部包括bn.running_mean, bn.running_var进行ema update
    # msd = smodel.state_dict()
    # esd = tmodel.state_dict()
    for j in esd.keys():
        model_v = msd[j].detach()
        ema_v = esd[j]
        esd[j].copy_(ema_v * decay + (1. - decay) * model_v)


    

class teacherEMA(object):
    def __init__(self, args, model, decay=0.999):
        self.ema = deepcopy(model)
        self.ema.to(args.device)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        # for p in self.ema.parameters():
        #     p.requires_grad_(False)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])


    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)