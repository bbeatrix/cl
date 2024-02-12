import torch
from torch.func import functional_call, vmap, grad
import wandb


def flatten_grads(grad_dict):
    # flatten all but batch dim, concat grads
    flattened_grads = torch.cat([grad.reshape(grad.shape[0], -1) for grad in grad_dict.values()], axis=1)
    return flattened_grads

def get_batch_grad_stats(model, input_images, target, function):
    model.eval()

    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    ft_compute_grad = grad(function)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0), randomness="different")

    ft_per_sample_grads_dict = ft_compute_sample_grad(params, buffers, input_images, target)
    ft_per_sample_grads_flattened = flatten_grads(ft_per_sample_grads_dict)

    batch_mean_grad = torch.mean(ft_per_sample_grads_flattened, axis=0)
    batch_mean_grad_norm = torch.linalg.norm(batch_mean_grad, ord=2)

    per_sample_grad_norms = torch.linalg.norm(ft_per_sample_grads_flattened, ord=2, axis=1)
    batch_grad_norm_mean = torch.mean(per_sample_grad_norms, axis=0)
    batch_grad_norm_std = torch.std(per_sample_grad_norms, axis=0)

    batch_sum_grad_norm = torch.linalg.norm(torch.sum(ft_per_sample_grads_flattened, axis=0), ord=2)

    model.train()
    return batch_sum_grad_norm, batch_mean_grad_norm, batch_grad_norm_mean, batch_grad_norm_std

def log_batch_grad_stats(model, loss_function, global_iters, input_images, target, batch_type="train", output_index=None):
    model.eval()

    def model_output_func(params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)

        output = functional_call(model, (params, buffers), (batch, output_index))
        target_output = torch.gather(output, 1, targets.view(-1, 1))
        return target_output.squeeze()
    
    def loss_func(params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)

        predictions = functional_call(model, (params, buffers), (batch, output_index))
        loss_per_sample = loss_function(predictions, targets)
        return loss_per_sample.mean()

    batch_out_sum_grad_norm, batch_out_mean_grad_norm, batch_out_grad_norm_mean, batch_out_grad_norm_std = get_batch_grad_stats(model, input_images, target, model_output_func)
    batch_loss_sum_grad_norm, batch_loss_mean_grad_norm, batch_loss_grad_norm_mean, batch_loss_grad_norm_std = get_batch_grad_stats(model, input_images, target, loss_func)

    wandb.log({
            f"fgrad/{batch_type} batch out mean grad norm": batch_out_mean_grad_norm,
            f"fgrad/{batch_type} batch out sum grad norm": batch_out_sum_grad_norm,
            f"fgrad/{batch_type} batch out grad norm mean": batch_out_grad_norm_mean,
            f"fgrad/{batch_type} batch out grad norm std": batch_out_grad_norm_std,

            f"lossgrad/{batch_type} batch loss sum grad norm": batch_loss_sum_grad_norm,
            f"lossgrad/{batch_type} batch loss mean grad norm": batch_loss_mean_grad_norm,
            f"lossgrad/{batch_type} batch loss grad norm mean": batch_loss_grad_norm_mean,
            f"lossgrad/{batch_type} batch loss grad norm std": batch_loss_grad_norm_std},
            step=global_iters)

    model.train()
    return

def log_margin_stats(global_iters, batch_model_output, target, batch_type="train"):
    batch_correct_output = batch_model_output.gather(1, target.unsqueeze(dim=1)).squeeze(dim=1).detach()

    batch_correct_output_mean = torch.mean(batch_correct_output)
    batch_correct_output_std = torch.std(batch_correct_output)
    batch_correct_output_abs = torch.abs(batch_correct_output)

    batch_model_output.scatter_(1, target.unsqueeze(1), float('-inf'))
    max_output_of_rest, _ = torch.max(batch_model_output, dim=1)

    batch_out_margin = batch_correct_output - max_output_of_rest
    batch_out_margin_abs = torch.abs(batch_out_margin)
    batch_out_margin_mean = torch.mean(batch_out_margin)
    batch_out_margin_std = torch.std(batch_out_margin)

    batch_out_margin_abs_mean = torch.mean(batch_out_margin)
    batch_out_margin_abs_std = torch.std(batch_out_margin)

    batch_normalized_margin_abs = batch_out_margin_abs / batch_correct_output_abs

    batch_normalized_margin_abs_mean = torch.mean(batch_normalized_margin_abs)
    batch_normalized_margin_abs_std = torch.std(batch_normalized_margin_abs)

    wandb.log({
        f"margin/{batch_type} batch correct output mean": batch_correct_output_mean,
        f"margin/{batch_type} batch correct output std": batch_correct_output_std,
        f"margin/{batch_type} batch out margin mean": batch_out_margin_mean,
        f"margin/{batch_type} batch out margin std": batch_out_margin_std,
        f"margin/{batch_type} batch out margin abs mean": batch_out_margin_abs_mean,
        f"margin/{batch_type} batch out margin abs std": batch_out_margin_abs_std,
        f"margin/{batch_type} batch normalized margin abs mean": batch_normalized_margin_abs_mean,
        f"margin/{batch_type} batch normalized margin abs std": batch_normalized_margin_abs_std},
        step=global_iters)
    return
