import os
from dataclasses import dataclass

import dill
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax import value_and_grad, vmap, jit, lax


def glue_params(params):
    return jnp.concatenate(params)


def unglue_params(glued_params, params_sample):
    sizes = [len(ip) for ip in params_sample]
    slice_indices = [sum(sizes[:i]) for i in range(len(sizes)+1)]

    return params_sample._make([glued_params[i0:i1] for i0, i1 in zip(slice_indices, slice_indices[1:])])


def unglue_params_history(params_glued_history, params_sample):
    return [unglue_params(params, params_sample) for params in params_glued_history]


def mynimize(loss, initial_params_batch, opt_options):

    params_sample = initial_params_batch[0]

    regloss = opt_options.regloss(loss)
    regloss_glued = lambda params_glued: regloss(unglue_params(params_glued, params_sample))
    loss_and_grad_glued = value_and_grad(regloss_glued)

    @jit
    def single_minimize(initial_params_glued):
        return single_mynimize(loss_and_grad_glued, initial_params_glued, opt_options)

    initial_params_glued_batch = jnp.array([glue_params(params) for params in initial_params_batch])
    params_glued_histories, loss_histories = vmap(single_minimize)(initial_params_glued_batch)

    # print('done minimizing')
    results = [[ph, lh] for ph, lh in zip(params_glued_histories, loss_histories)]  # This step takes unreasonably long
    # print('done unglueing')
    return OptMultiResult(results, loss, opt_options, params_sample)


def single_mynimize(loss_and_grad, initial_params, opt_options):
    opt = opt_options.optimizer()
    opt_state = opt.init(initial_params)

    def iteration_with_history(i, carry):
        params_history, loss_history, opt_state = carry
        params = params_history[i]
        params, loss, opt_state = update_step(loss_and_grad, opt, opt_state, params)
        params_history = params_history.at[i+1].set(params)
        loss_history = loss_history.at[i].set(loss)
        return params_history, loss_history, opt_state

    def iteration_without_history(i, carry):
        params, best_params, previous_loss, best_loss, opt_state = carry
        params, loss, opt_state = update_step(loss_and_grad, opt, opt_state, params)
        best_loss, best_params = lax.cond(loss < best_loss,
                                          lambda x: [loss, params],
                                          lambda x: [best_loss, best_params],
                                          None)
        return params, best_params, previous_loss, best_loss, opt_state

    if opt_options.keep_history:
        params_history = jnp.zeros((opt_options.num_iterations, len(initial_params))).at[0].set(initial_params)
        loss_history = jnp.zeros((opt_options.num_iterations,))

        params_history, loss_history, _ = lax.fori_loop(
            0,
            opt_options.num_iterations,
            iteration_with_history,
            (params_history, loss_history, opt_state))

    else:
        initial_loss, _ = loss_and_grad(initial_params)
        params, best_params, previous_loss, best_loss, opt_state = lax.fori_loop(
            0,
            opt_options.num_iterations,
            iteration_without_history,
            (initial_params, initial_params, initial_loss, initial_loss, opt_state))

        params_history = jnp.array([initial_params, best_params])
        loss_history = jnp.array([initial_loss, best_loss])

    return params_history, loss_history


def update_step(loss_and_grad, opt, opt_state, params):
    loss, grads = loss_and_grad(params)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, loss, opt_state


@dataclass
class OptOptions:
    learning_rate: float = 0.01
    num_iterations: int = 100
    random_seed: int = 0
    regularization_func: callable = None
    keep_history: bool = True

    def __post_init__(self):
        if self.regularization_func is not None:
            self.regularization_func = jit(self.regularization_func)

    def optimizer(self):
        return optax.adam(self.learning_rate)

    def regloss(self, loss):
        if self.regularization_func is None:
            return loss
        else:
            return lambda x: (loss(x) + self.regularization_func(x))


class OptMultiResult:
    def __init__(self, raw_results, loss_func, opt_options, params_sample):
        self.all_results = [OptResult(r, loss_func, opt_options, params_sample) for r in raw_results]
        self.all_best_losses = [res.best_loss for res in self.all_results]
        self.best_loss = min(self.all_best_losses)
        self.best_result = min(self.all_results, key=lambda res: res.best_loss)

    def success_ratio(self, percentage=0.1):
        best_overall_loss = min(self.all_best_losses)
        losses_within_margin = [loss <= best_overall_loss*(1+percentage) for loss in self.all_best_losses]
        return sum(losses_within_margin) / len(losses_within_margin)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            dill.dump(self, f)

    def __repr__(self):
        return f'OptMultiResult(best_loss {self.best_loss}, success_ratio {self.success_ratio()}, num_samples {len(self.all_best_losses)})'

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            results = dill.load(f)

        return results


class OptResult:
    def __init__(self, raw_result, loss_func, opt_options, params_sample, best='loss'):

        self.loss_func = loss_func
        self.opt_options = opt_options
        self.glued_params_history = raw_result[0]
        self.regloss_history = raw_result[1]
        self._params_sample = params_sample

        if opt_options.regularization_func is not None:
            self.reg_history = vmap(opt_options.regularization_func)(unglue_params_history(self.glued_params_history))
            self.loss_history = self.regloss_history - self.reg_history
        else:
            self.reg_history = None
            self.loss_history = self.regloss_history

        assert best in ['loss', 'regloss'], 'best flag must be `loss` or `regloss`'
        if best == 'loss':
            self._best_i = jnp.argmin(self.loss_history)
        else:
            self._best_i = jnp.argmin(self.regloss_history)

        self.best_params = unglue_params(self.glued_params_history[self._best_i], self._params_sample)
        self.best_loss = self.loss_history[self._best_i]

    def params(self, i):
        return unglue_params(self.glued_params_history[i], self._params_sample)

    def _i_or_best_i(self, i):
        if i is None:
            return self._best_i
        else:
            return i

    def __repr__(self):
        return f'OptResult(best_loss {self.best_loss})'

    def plot_loss_history(self):
        plt.plot(self.loss_history)
        plt.yscale('log')
