import time
import logging

import torch
import torch.optim as optim
import torch.nn.functional as F

from networks.nets import LeNet
from classification import StepClassifier

# Dataset Distillation Module
from src.utils import visualize


class Distiller:
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_data_steps = cfg.DISTILL.d_steps  # how much data we have per epoch
        self.T = cfg.DISTILL.d_steps * cfg.DISTILL.d_epochs  # total number of steps
        # how much data to distill for each step
        self.num_per_step = cfg.DATA_SET.num_classes * cfg.DISTILL.num_per_class

        self.models = []
        self.params = []
        self.labels = []  # labels (no label distillation)
        self.data = []  # distilled data
        self.raw_distill_lrs = []  # learning rate to train task model with distilled data

        self.init_models()
        self.init_data()

        self.optimizer = optim.Adam(self.params, lr=cfg.DISTILL.lr, betas=(0.5, 0.999))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg.DISTILL.decay_epochs,
                                                   gamma=cfg.DISTILL.decay_factor)

    # init models
    def init_models(self):
        cfg = self.cfg
        if cfg.DISTILL.model == 'LeNet':
            for m in range(0, cfg.DISTILL.sample_nets):
                task_model = LeNet(cfg).to(cfg.device)
                self.models.append(task_model)
        else:
            raise RuntimeError("{} Not Implemented".format(cfg.DISTILL.model))

    # initialize label, distilled data and learning rate
    def init_data(self):
        cfg = self.cfg

        # labels
        distill_label = torch.arange(cfg.DATA_SET.num_classes, dtype=torch.long, device=cfg.device) \
            .repeat(cfg.DISTILL.num_per_class, 1)
        distill_label = distill_label.t().reshape(-1)  # [0, 0, ... , 1, 1, ... ]
        for _ in range(self.num_data_steps):
            self.labels.append(distill_label)
            # don't distill labels

        # data
        for _ in range(self.num_data_steps):
            distill_data = torch.randn(self.num_per_step, cfg.DATA_SET.num_channels, cfg.DATA_SET.input_size,
                                       cfg.DATA_SET.input_size, device=cfg.device, requires_grad=True)
            self.data.append(distill_data)
            self.params.append(distill_data)

        # learning rate
        raw_init_distill_lr = torch.tensor(cfg.DISTILL.d_lr, device=cfg.device)
        raw_init_distill_lr = raw_init_distill_lr.repeat(self.T, 1)
        self.raw_distill_lrs = raw_init_distill_lr.expm1_().log_().requires_grad_()
        self.params.append(self.raw_distill_lrs)

        for p in self.params:
            p.grad = torch.zeros_like(p)

    # return label, distilled data and learning rate in a list
    def get_steps(self):
        total_data = (x for _ in range(self.cfg.DISTILL.d_epochs) for x in zip(self.data, self.labels))
        lrs = F.softplus(self.raw_distill_lrs).unbind()

        steps = []
        for (data, label), lr in zip(total_data, lrs):
            steps.append((data, label, lr))

        return steps

    # collect gradient for parameters
    def accumulate_grad(self, grad_infos):
        bwd_out = []
        bwd_grad = []
        for datas, gdatas, lrs, glrs in grad_infos:
            bwd_out += list(lrs)
            bwd_grad += list(glrs)
            for d, g in zip(datas, gdatas):
                d.grad.add_(g)  # add gradient to data
        if len(bwd_out) > 0:
            torch.autograd.backward(bwd_out, bwd_grad)

    # train task model using distilled data
    def forward(self, model, rdata, rlabel, steps):
        # forward distilled dataset
        model.train()
        w = model.get_param()
        params = [w]
        gws = []

        for step, (data, label, lr) in enumerate(steps):
            with torch.enable_grad():
                output = model.forward_with_param(data, w)
                loss = F.cross_entropy(output, label)
            gw, = torch.autograd.grad(loss, w, lr.squeeze(), create_graph=True)

            with torch.no_grad():
                new_w = w.sub(gw).requires_grad_()
                params.append(new_w)
                gws.append(gw)
                w = new_w

        # calculate loss using train data
        model.eval()
        output = model.forward_with_param(rdata, params[-1])
        tloss = F.cross_entropy(output, rlabel)
        return tloss, (tloss, params, gws)

    # update distilled data and lr
    def backward(self, model, steps, saved):
        tloss, params, gws = saved

        # updated values and gradients
        datas = []
        gdatas = []
        lrs = []
        glrs = []

        dw, = torch.autograd.grad(tloss, (params[-1],))

        model.train()
        # back-gradient optimization
        for (data, label, lr), w, gw, in reversed(list(zip(steps, params, gws))):
            # input of autograd
            hvp_in = [w]
            hvp_in.append(data)
            hvp_in.append(lr)
            dgw = dw.neg()  # negate learning rate
            hvp_grad = torch.autograd.grad(
                outputs=(gw,),
                inputs=hvp_in,
                grad_outputs=(dgw,)
            )

            with torch.no_grad():
                datas.append(data)
                gdatas.append(hvp_grad[1])
                lrs.append(lr)
                glrs.append(hvp_grad[2])

                dw.add_(hvp_grad[0])

        return datas, gdatas, lrs, glrs

    # distill dataset
    def distill(self):
        cfg = self.cfg
        device = cfg.device
        num_subnets = cfg.DISTILL.sample_nets
        log_intv = cfg.DISTILL.log_intv
        val_model = StepClassifier(cfg)
        val_intv = cfg.DISTILL.val_intv

        data_t0 = time.time()

        for epoch, it, (rdata, rlabel) in self.prefetch_train_loader_iter():
            data_t = time.time() - data_t0  # data load time

            if it == 0 and not epoch == 0:
                self.scheduler.step()

            if it == 0 and epoch % val_intv == 0:
                # logging.info('Begin of epoch {} validation result'.format(epoch))
                with torch.no_grad():
                    steps = self.get_steps()
                # val_model.set_step(steps)
                # val_model.train_and_evaluate(valid=True)
                visualize(cfg, steps, epoch)

            self.optimizer.zero_grad()
            rdata, rlabel = rdata.to(device, non_blocking=True), rlabel.to(device, non_blocking=True)

            task_models = self.models

            t0 = time.time()
            ls_train = []
            steps = self.get_steps()
            grad_infos = []
            for model in task_models:
                model.reset2(cfg.DISTILL.init, cfg.DISTILL.init_param)

                l_train, saved = self.forward(model, rdata, rlabel, steps)
                ls_train.append(l_train.detach())
                grad_infos.append(self.backward(model, steps, saved))
                del l_train, saved
            self.accumulate_grad(grad_infos)

            # average gradient
            grads = [p.grad for p in self.params]
            for g in grads:
                g.div_(num_subnets)

            self.optimizer.step()
            t = time.time() - t0  # train time

            if it == 0 and epoch % log_intv == 0:
                ls_train = torch.stack(ls_train, 0).sum().item() / num_subnets
                logging.info('Epoch: {:4d}, Train Loss: {:.4f}, '
                             'Data time: {:.2f}, Train time: {:.2f}'
                             .format(epoch, ls_train, data_t, t))
                if ls_train != ls_train:
                    raise RuntimeError('loss became NaN')

            del steps, grad_infos, ls_train, grads

            data_t0 = time.time()

        logging.info('Distillation finished')
        # return results
        with torch.no_grad():
            steps = self.get_steps()
        return steps

    # get epoch, iteration and data from train_loader
    def prefetch_train_loader_iter(self):
        cfg = self.cfg
        train_iter = iter(cfg.train_loader)
        for epoch in range(cfg.DISTILL.epochs):
            niter = len(train_iter)
            prefetch_it = max(0, niter - 2)
            for it, val in enumerate(train_iter):
                # Prefetch (start workers) at the end of epoch BEFORE yielding
                if it == prefetch_it and epoch < cfg.DISTILL.epochs - 1:
                    train_iter = iter(cfg.train_loader)
                yield epoch, it, val
