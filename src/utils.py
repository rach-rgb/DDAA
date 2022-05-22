import os
import logging
import warnings
from pathlib import Path

import torch
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


# source: https://github.com/SsnL/dataset-distillation/blob/master/utils/io.py
# load distilled dataset
def load_results(cfg):
    load_dir = os.path.join(Path(os.getcwd()).parent, cfg.DISTILL.load_dir)
    load_path = os.path.join(load_dir, 'result.pth')
    device = cfg.device

    # change to tensor
    np_steps = torch.load(load_path, map_location=device)
    if not isinstance(np_steps[0][0], torch.Tensor) or not (np_steps[0][0]).device == device:
        steps = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for step in np_steps:
                steps.append(tuple(torch.as_tensor(t, device=device) for t in step))
    else:   # already tensor
        steps = np_steps

    logging.info('Distilled data loaded from {}'.format(load_path))
    return steps


# save distilled dataset
def save_results(cfg, steps):
    steps = [(d.detach().cpu(), l.detach().cpu(), lr) for (d, l, lr) in steps]

    output_dir = os.path.join(Path(os.getcwd()).parent, cfg.DISTILL.output_dir)
    output = os.path.join(output_dir, 'result.pth')

    torch.save(steps, output)
    logging.info('Distilled data saved to {}'.format(output))

    if cfg.DISTILL.save_vis_output:
        visualize(cfg, steps)
        logging.info('Visualized data saved to {}'.format(output_dir))


# save visualized distilled dataset
def visualize(cfg, steps, epoch=None):
    if isinstance(steps[0][0], torch.Tensor):   # change to ndarray
        np_steps = []
        for data, label, lr in steps:
            np_data = data.detach().permute(0, 2, 3, 1).to('cpu').numpy()
            np_label = label.detach().to('cpu').numpy()
            if lr is not None:
                lr = lr.detach().cpu().numpy()
            np_steps.append((np_data, np_label, lr))
        steps = np_steps

    dataset_vis_info = (cfg.DATA_SET.name, cfg.DATA_SET.num_channels, cfg.DATA_SET.input_size,
                        cfg.DATA_SET.mean, cfg.DATA_SET.std, cfg.DATA_SET.labels)
    if epoch is None:
        vis_dir = os.path.join(Path(os.getcwd()).parent, cfg.DISTILL.output_dir)
    else:
        vis_dir = os.path.join(Path(os.getcwd()).parent, cfg.DISTILL.output_dir, 'epoch'+str(epoch))
        os.mkdir(vis_dir)
    vis_args = (steps, cfg.DISTILL.num_per_class, dataset_vis_info, vis_dir)
    _vis_results_fn(*vis_args)


def _vis_results_fn(np_steps, img_per_class, dataset_info,
                    vis_dir=None, vis_name_fmt='visuals_step{step:03d}', supertitle=True, subtitle=True,
                    reuse_axes=True):

    vis_name_fmt += '.png'

    dataset, nc, input_size, mean, std, label_names = dataset_info

    N = len(np_steps[0][0])
    nrows = max(2, img_per_class)
    grid = (nrows, np.ceil(N / float(nrows)).astype(int))
    plt.rcParams["figure.figsize"] = (grid[1] * 1.5 + 1, nrows * 1.5 + 1)

    plt.close('all')
    fig, axes = plt.subplots(nrows=grid[0], ncols=grid[1])
    axes = axes.flatten()
    if supertitle:
        fmts = [
            'Dataset: {dataset}',
        ]
        if len(np_steps) > 1:
            fmts.append('Step: {{step}}')
        if np_steps[0][-1] is not None:
            fmts.append('LR: {{lr:.4f}}')
        supertitle_fmt = ', '.join(fmts).format(dataset=dataset)

    plt_images = []
    first_run = True
    for i, (data, labels, lr) in enumerate(np_steps):
        for n, (img, label, axis) in enumerate(zip(data, labels, axes)):
            if nc == 1:
                img = img[..., 0]
            img = (img * std + mean).clip(0, 1)
            if first_run:
                plt_images.append(axis.imshow(img, interpolation='nearest'))
            else:
                plt_images[n].set_data(img)
            if first_run:
                axis.axis('off')
                if subtitle:
                    axis.set_title('Label {}'.format(label_names[label]))
        if supertitle:
            if lr is not None:
                lr = lr.sum().item()
            plt.suptitle(supertitle_fmt.format(step=i, lr=lr))
            if first_run:
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0, rect=[0, 0, 1, 0.95])
        fig.canvas.draw()

        plt.savefig(os.path.join(vis_dir, vis_name_fmt.format(step=i)))
        if reuse_axes:
            first_run = False
        else:
            fig, axes = plt.subplots(nrows=grid[0], ncols=grid[1])
            axes = axes.flatten()
            plt.show()()


# Args: steps(list): list of distilled images, labels and lrs
# Returns: images(tensor list), labels(int list)
def step_to_tensor(steps):
    x = []
    y = []
    for data, labels, lr in steps:
        for img, label in zip(data, labels):
            x.append(img)
            y.append(label.item())
    return x, y


# PIL Image Methods
def tensor_to_pil(x):
    n_channels = int(x.shape[0])
    np_x = x.detach().permute(1, 2, 0).numpy()
    np_x = (np_x * 255).astype(np.uint8)
    if n_channels == 1:
        im_x = Image.fromarray(np.squeeze(np_x, axis=2), mode='L')
    else:
        im_x = Image.fromarray(np_x)
    return im_x


def concat_row(img_list):
    w, h = zip(*(i.size for i in img_list))
    result = Image.new('RGB', (sum(w), max(h)))
    off = 0
    for im in img_list:
        result.paste(im, (off, 0))
        off += im.size[0]
    return result


def concat_col(img_list):
    w, h = zip(*(i.size for i in img_list))
    result = Image.new('RGB', (max(w), sum(h)))
    off = 0
    for im in img_list:
        result.paste(im, (0, off))
        off += im.size[1]
    return result
