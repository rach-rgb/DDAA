import time
import logging

import torch
import torch.optim as optim
import torch.nn.functional as F

import augmentation as aug
from utils import visualize
from loss_model import get_loss
from networks.nets import LeNet, AlexCifarNet


# Dataset Distillation Module
class Distiller:
    def __init__(self, cfg):
        self.cfg = cfg
        self.do_val = cfg.DISTILL.validation  # introduce validation
        self.do_raug = cfg.DISTILL.raw_augment  # apply augmentation for raw data
        self.do_daug = cfg.DISTILL.dd_augment  # apply augmentation for distilled data
        self.do_vis = cfg.DISTILL.save_vis_output

        self.dd_step = cfg.DISTILL.d_steps  # data per epoch
        self.dd_epoch = cfg.DISTILL.d_epochs  # number of epoch
        self.num_steps = self.dd_step * self.dd_epoch  # total number of steps
        self.data_per_step = cfg.DATA_SET.num_classes * cfg.DISTILL.num_per_class

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

        # focal loss parameters
        if cfg.DISTILL.rloss_crit == 'BF' or cfg.DISTILL.rloss_crit == 'BCE':
            self.info = (cfg.device,
                         cfg.train_loader.dataset.n_classes, cfg.train_loader.dataset.n_per_classes)
        elif cfg.DISTILL.rloss_crit == 'CE':
            self.info = None
        else:
            logging.error("Loss Model {} not implemented".format(cfg.DISTILL.rloss_crit))
            raise NotImplementedError
        logging.info("Loss: {}".format(cfg.DISTILL.rloss_crit))
        assert cfg.DISTILL.dloss_crit == 'CE'

    # init models
    def init_models(self):
        cfg = self.cfg
        if cfg.DISTILL.model == 'LeNet':
            for m in range(cfg.DISTILL.sample_nets):
                task_model = LeNet(cfg).to(cfg.device)
                self.models.append(task_model)
        elif cfg.DISTILL.model == 'AlexCifarNet':
            for m in range(cfg.DISTILL.sample_nets):
                task_model = AlexCifarNet(cfg).to(cfg.device)
                self.models.append(task_model)
        else:
            logging.error("{} Not Implemented".format(cfg.DISTILL.model))
            raise NotImplementedError

    # initialize label, distilled data and learning rate
    def init_data(self):
        cfg = self.cfg

        # labels
        distill_label = torch.arange(cfg.DATA_SET.num_classes, dtype=torch.long, device=cfg.device) \
            .repeat(cfg.DISTILL.num_per_class, 1)
        distill_label = distill_label.t().reshape(-1)  # [0, 0, ... , 1, 1, ... ]
        for _ in range(self.dd_step):
            self.labels.append(distill_label)
            # don't distill labels

        # data
        for _ in range(self.dd_step):
            distill_data = torch.randn(self.data_per_step, cfg.DATA_SET.num_channels, cfg.DATA_SET.input_size,
                                       cfg.DATA_SET.input_size, device=cfg.device, requires_grad=True)
            self.data.append(distill_data)
            self.params.append(distill_data)

        # learning rate
        raw_init_distill_lr = torch.tensor(cfg.DISTILL.d_lr, device=cfg.device)
        raw_init_distill_lr = raw_init_distill_lr.repeat(self.dd_step * self.dd_epoch, 1)
        self.raw_distill_lrs = raw_init_distill_lr.expm1_().log_().requires_grad_()
        self.params.append(self.raw_distill_lrs)

        for p in self.params:
            p.grad = torch.zeros_like(p)

    # return label, distilled data and learning rate in a list
    def get_steps(self):
        total_data = (x for _ in range(self.dd_epoch) for x in zip(self.data, self.labels))
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
        raw_loss_crit = self.cfg.DISTILL.rloss_crit

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
        tloss = get_loss(output, rlabel, raw_loss_crit, params=self.info)
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
        logging.info('Apply dataset distillation')

        cfg = self.cfg
        device = cfg.device
        task_models = self.models  # classifiers
        n_subnets = len(task_models)
        task_params = [0] * n_subnets  # recent parameters of task_models

        max_it = len(cfg.train_loader) - 1
        unreach_ep = cfg.DISTILL.epochs + 1

        # additional features
        log_intv = cfg.DISTILL.log_intv
        vis_intv = cfg.DISTILL.vis_intv if self.do_vis else unreach_ep
        val_intv = cfg.DISTILL.val_intv if self.do_val else unreach_ep

        # augmentation
        augmentor = None
        p_optimizer = None
        exp_intv = unreach_ep
        start_aug = cfg.RAUG.start_ep
        aug_online = cfg.RAUG.online
        do_autoaug = True if self.do_raug and cfg.RAUG.aug_type == 'Auto' else False
        if self.do_raug:
            if cfg.RAUG.aug_type == 'Random':
                augmentor = aug.AugModule(device, cfg.RAUG)
            elif cfg.RAUG.aug_type == 'Auto':
                assert not self.do_val
                if cfg.RAUG.load:
                    augmentor, p_optimizer = aug.autoaug_load(device, cfg, cfg.RAUG)
                else:
                    augmentor, p_optimizer = aug.autoaug_creator(device, cfg.RAUG, task_models[0])
                exp_intv = cfg.RAUG.search_intv
                if not aug_online:
                    cfg.train_loader.dataset.augment_dataset(augmentor, 3)
            else:
                logging.error("{} Augmentation for raw data not implemented".format(cfg.RAUG.aug_type))
                raise NotImplementedError

        if self.do_daug:
            logging.error("Augmentation for distilled data during distillation not implemented")
            raise NotImplementedError

        data_t0 = time.time()
        for epoch, it, (rdata, rlabel) in self.prefetch_train_loader_iter():
            rdata, rlabel = rdata.to(device), rlabel.to(device)
            data_t = time.time() - data_t0  # data load time

            if it == 0 and epoch != 0:
                self.scheduler.step()

            if self.do_raug and aug_online and epoch == start_aug and it == 0:
                cfg.train_loader.dataset.add_augmentation(1, augmentor)  # Tensor -> Aug -> Normalize

            # explore auto-aug strategy
            if do_autoaug and epoch % exp_intv == 0 and it == 0 and epoch != 0:
                if not augmentor.loaded:
                    ex_t0 = time.time()
                    for idx, model in enumerate(task_models):
                        model.unflatten_weight(task_params[idx])
                        aug.autoaug_update(device, augmentor, p_optimizer, cfg.val_loader)
                    ex_t = time.time() - ex_t0
                    logging.info('Epoch: {:4d}, Iteration: {:4d}, Search time: {:.2f}'.format(epoch, it, ex_t))

            train_t0 = time.time()

            rlosses = []
            grad_infos = []
            steps = self.get_steps()
            self.optimizer.zero_grad()
            for mid, model in enumerate(task_models):
                model.reset2(cfg.DISTILL.init, cfg.DISTILL.init_param)

                rloss, saved = self.forward(model, rdata, rlabel, steps)
                rlosses.append(rloss.detach())
                grad_infos.append(self.backward(model, steps, saved))

                task_params[mid] = saved[1][-1]
                del rloss, saved
            self.accumulate_grad(grad_infos)

            # average gradient
            grads = [p.grad for p in self.params]
            for g in grads:
                g.div_(n_subnets)
            self.optimizer.step()

            train_t = time.time() - train_t0  # train time

            if it == max_it and epoch % log_intv == 0:
                rlosses = torch.stack(rlosses, 0).sum().item() / n_subnets
                logging.info('Epoch: {:4d}, Iteration: {:4d}, Train Loss: {:.4f}, Data time: {:.2f}, Train time: {:.2f}'
                             .format(epoch, it, rlosses, data_t, train_t))
                if rlosses != rlosses:
                    logging.error("loss became NAN")
                    raise

            del steps, rlosses, grad_infos, grads

            # save visualized intermediate result
            if self.do_vis and epoch % vis_intv == 0 and it == max_it and epoch != 0:
                logging.info("Epoch: {:4d}: save visualized result".format(epoch))
                with torch.no_grad():
                    steps = self.get_steps()
                    visualize(cfg, steps, epoch)

            # validation
            if self.do_val and epoch % val_intv == 0 and it == max_it and epoch != 0:
                logging.info('Epoch: {:4d}: validation'.format(epoch))
                avg_loss = 0
                avg_accu = 0
                with torch.no_grad():
                    for w, m in zip(task_params, task_models):
                        m.unflatten_weight(w)
                        l, a = self.validate(device, m, cfg.val_loader)
                        avg_loss += l
                        avg_accu += avg_accu
                    avg_loss = l / n_subnets
                    avg_accu = avg_accu / n_subnets
                    logging.info("Average loss: {:.4f}, average accuracy: {:.0f}".format(avg_loss, avg_accu))

            data_t0 = time.time()
            # end of for loop
        logging.info('Distillation finished')

        # model save
        if do_autoaug and cfg.RAUG.save:
            aug.autoaug_save(cfg.RAUG, augmentor)

        with torch.no_grad():
            steps = self.get_steps()
        return steps

    # validation
    def validate(self, device, model, dataloader):
        avg_loss = 0
        accuracy = 0
        model.eval()

        with torch.no_grad():
            for data, label in dataloader:
                data, label = data.to(device), label.to(device)
                output = model(data)
                avg_loss += F.cross_entropy(output, label, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                accuracy += pred.eq(label.view_as(pred)).sum().item()

        avg_loss /= len(dataloader.dataset)
        accuracy = accuracy * 100. / len(dataloader.dataset)

        return avg_loss, accuracy

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
