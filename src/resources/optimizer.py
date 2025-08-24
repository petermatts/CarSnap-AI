import torch

options = ['Adadelta', 'Adafactor', 'Adagrad', 'Adam', 'AdamW', 'SparseAdam',
           'Adamax', 'ASGD', 'LBFGS', 'NAdam', 'RMSprop', 'Rprop', 'SGD']


def getOptimizer(optimizer: str, model: torch.nn.Module, **kwargs) -> torch.optim.Optimizer:
    """
    Returns the specified optimizer on the given model with the specified key word arguments
    """
    if optimizer not in options:
        raise ValueError(
            f"Did not find torch optimizer for {optimizer}. Expeceted one of {options}")

    params = model.parameters()

    # unpacking common variables, with pytorch defaults if not specified
    foreach = kwargs.get('foreach', None)
    maximize = kwargs.get('maximize', False)
    capturable = kwargs.get('capturable', False)
    differentiable = kwargs.get('differentiable', False)
    fused = kwargs.get('fused', None)

    match optimizer:
        case 'Adadelta':
            lr = kwargs.get('lr', 1.0)
            rho = kwargs.get('rho', 0.9)
            eps = kwargs.get('eps', 1e-6)
            weight_decay = kwargs.get('weight_decay', 0)

            return torch.optim.Adadelta(params, lr=lr, rho=rho, eps=eps, weight_decay=weight_decay, foreach=foreach, capturable=capturable, maximize=maximize, differentiable=differentiable)

        case 'Adafactor':
            lr = kwargs.get('lr', 0.01)
            beta2_decay = kwargs.get('beta2_decay', -0.8)
            eps = kwargs.get('eps', (None, 0.001))
            d = kwargs.get('d', 1.0)
            weight_decay = kwargs.get('weight_decay', 0.0)
            return torch.optim.Adafactor(params, lr=lr, beta2_decay=beta2_decay, eps=eps, d=d, weight_decay=weight_decay, foreach=foreach, maximize=maximize)

        case 'Adagrad':
            lr = kwargs.get('lr', 0.01)
            lr_decay = kwargs('lr_decay', 0)
            weight_decay = kwargs.get('weight_decay', 0)
            initial_accumulator_value = kwargs.get(
                'initial_accumulator_value', 0)
            eps = kwargs.get('eps', 1e-10)

            return torch.optim.Adagrad(params, lr=lr, lr_decay=lr_decay, weight_decay=weight_decay, initial_accumulator_value=initial_accumulator_value, eps=eps, foreach=foreach, maximize=maximize, differentiable=differentiable, fused=fused)

        case 'Adam':
            lr = kwargs.get('lr', 0.01)
            betas = kwargs.get('betas', (0.9, 0.999))
            eps = kwargs.get('eps', 1e-8)
            weight_decay = kwargs.get('weight_decay', 0)
            amsgrad = kwargs.get('amsgrad', False)

            return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, foreach=foreach, maximize=maximize, capturable=capturable, differentiable=differentiable, fused=fused)

        case 'AdamW':
            lr = kwargs.get('lr', 0.001)
            betas = kwargs.get('betas', (0.9, 0.999))
            eps = kwargs.get('eps', 1e-8)
            weight_decay = kwargs.get('weight_decay', 0.01)
            amsgrad = kwargs.get('amsgrad', False)

            return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, foreach=foreach, maximize=maximize, capturable=capturable, differentiable=differentiable, fused=fused)

        case 'SparseAdam':
            lr = kwargs.get('lr', 0.001)
            betas = kwargs.get('betas', (0.9, 0.999))
            eps = kwargs.get('eps', 1e-8)

            return torch.optim.SparseAdam(params, lr=lr, betas=betas, eps=eps)

        case 'Adamax':
            lr = kwargs.get('lr', 0.002)
            betas = kwargs.get('betas', (0.9, 0.999))
            eps = kwargs.get('eps', 1e-8)
            weight_decay = kwargs.get('weight_decay', 0)
            amsgrad = kwargs.get('amsgrad', False)

            return torch.optim.Adamax(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, foreach=foreach, maximize=maximize, differentiable=differentiable, capturable=capturable)

        case 'ASGD':
            lr = kwargs.get('lr', 0.01)
            lambd = kwargs.get('lambd', 0.0001)
            alpha = kwargs.get('alpha', 0.75)
            t0 = kwargs.get('t0', 1000000.0)
            weight_decay = kwargs.get('weight_decay', 0)

            return torch.optim.ASGD(params, lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay, foreach=foreach, maximize=maximize, differentiable=differentiable, capturable=capturable)

        case 'LBFGS':
            lr = kwargs.get('lr', 1)
            max_iter = kwargs.get('max_iter', 20)
            max_eval = kwargs.get('max_eval', None)
            tolerance_grad = kwargs.get('tolerance_grad', 1e-7)
            tolerange_change = kwargs.get('tolerance_change', 1e-9)
            history_size = kwargs.get('history_size', 100)
            line_search_fn = kwargs.get('line_search_fn', None)

            return torch.optim.LBFGS(params, lr=lr, max_iter=max_iter, max_eval=max_eval, tolerance_grad=tolerance_grad, tolerance_change=tolerange_change, history_size=history_size, line_search_fn=line_search_fn)

        case 'NAdam':
            lr = kwargs.get('lr', 0.002)
            betas = kwargs.get('betas', (0.9, 0.999))
            eps = kwargs.get('eps', 1e-8)
            weight_decay = kwargs.get('weight_decay', 0)
            momentum_decay = kwargs.get('momentum_decay', 0.004)
            decoupled_weight_decay = kwargs.get(
                'decoupled_weigtht_decay', False)

            return torch.optim.NAdam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, momentum_decay=momentum_decay, decoupled_weight_decay=decoupled_weight_decay, foreach=foreach, maximize=maximize, differentiable=differentiable, capturable=capturable)

        case 'RAdam':
            lr = kwargs.get('lr', 0.001)
            betas = kwargs.get('betas', (0.9, 0.999))
            eps = kwargs.get('eps', 1e-8)
            weight_decay = kwargs.get('weight_decay', 0)
            decoupled_weight_decay = kwargs.get(
                'decoupled_weigtht_decay', False)

            return torch.optim.RAdam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, decoupled_weight_decay=decoupled_weight_decay, foreach=foreach, maximize=maximize, differentiable=differentiable, capturable=capturable)

        case 'RMSprop':
            lr = kwargs.get('lr', 0.01)
            alpha = kwargs.get('alpha', 0.99)
            eps = kwargs.get('eps', 1e-8)
            weight_decay = kwargs.get('weight_decay', 0)
            momentum = kwargs.get('momentum', 0)
            centered = kwargs.get('centered', False)

            return torch.optim.RMSprop(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered, capturable=capturable, foreach=foreach, maximize=maximize, differentiable=differentiable)

        case 'Rprop':
            lr = kwargs.get('lr', 0.01)
            etas = kwargs.get('etas', (0.5, 1.2))
            step_sizes = kwargs.get('step_sizes', (1e-6, 50))

            return torch.optim.Rprop(params, lr=lr, etas=etas, step_sizes=step_sizes, capturable=capturable, foreach=foreach, maximize=maximize, differentiable=differentiable)

        case 'SGD':
            lr = kwargs.get('lr', 0.001)
            momentum = kwargs.get('momentum', 0)
            dampening = kwargs.get('dampening', 0)
            weight_decay = kwargs.get('weight_decay', 0)
            nesterov = kwargs.get('nesterov', False)

            return torch.optim.SGD(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov, maximize=maximize, foreach=foreach, differentiable=differentiable, fused=fused)
