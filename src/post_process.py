# source: https://github.com/SsnL/dataset-distillation/blob/master/utils/io.py
import torch, os, logging
import numpy as np
import matplotlib
matplotlib.use('agg')  # this needs to be before the next line
import matplotlib.pyplot as plt


def save_results(cfg, steps):
    steps = [(d.detach().cpu(), l.detach().cpu(), lr) for (d, l, lr) in steps]

    result_path = cfg.OUTPUT.dir
    torch.save(steps, result_path)
    logging.info('Results saved to {}'.format(result_path))

    if cfg.OUTPUT.visualize is True:
        visualize(cfg, steps)


def visualize(cfg, steps):
    logging.info("Visualize Distilled Result")
    if isinstance(steps[0][0], torch.Tensor):   # change to ndarray
        np_steps = []
        for data, label, lr in steps:
            np_data = data.detach().permute(0, 2, 3, 1).to('cpu').numpy()
            np_label = label.detach().to('cpu').numpy()
            if lr is not None:
                lr = lr.detach().cpu().numpy()
            np_steps.append((np_data, np_label, lr))
        steps = np_steps

    dataset_vis_info = ('MNIST', 1, 28, 0.1307, 0.3081, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    vis_args = (steps, cfg.DISTILL.num_per_class, dataset_vis_info, cfg.OUTPUT.vis_dir)
    _vis_results_fn(*vis_args)

    logging.info("Visualize Done")


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
