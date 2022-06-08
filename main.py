import os
from dataclasses import dataclass

import dill
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax import value_and_grad, vmap, jit, lax


def mynimize(loss, initial_params_batch, opt_options):
    regloss = opt_options.regloss(loss)
    loss_and_grad = value_and_grad(regloss)

    @jit
    def single_minimize(initial_params):
        return single_mynimize(loss_and_grad, initial_params, opt_options)

    params_sample = initial_params_batch[0]

    threaded_initial_params_batch = params_sample._make([jnp.array(p) for p in zip(*initial_params_batch)])
    params_histories, loss_histories = vmap(single_minimize)(threaded_initial_params_batch)

    num_params = len(params_sample)
    num_samples = len(initial_params_batch)

    reordered_params_histories = [
        [params_sample._make([params_histories[num_param][num_sample][num_iteration] for num_param in range(num_params)])
         for num_iteration in range(opt_options.num_iterations)] for num_sample in range(num_samples)]

    results = [[ph, lh] for ph, lh in zip(reordered_params_histories, loss_histories)]

    return OptMultiResult(results, loss, opt_options)


def single_mynimize(loss_and_grad, initial_params, opt_options):
    opt = opt_options.optimizer()
    opt_state = opt.init(initial_params)
    sizes = [len(ip) for ip in initial_params]
    slice_indices = [sum(sizes[:i]) for i in range(len(sizes)+1)]

    def iteration_with_history(i, carry):
        glued_params_history, loss_history, opt_state = carry
        glued_params = glued_params_history[i]
        params = unglue_params(glued_params, slice_indices)
        params = initial_params._make(params)
        params, loss, opt_state = update_step(loss_and_grad, opt, opt_state, params)
        glued_params = glue_params(params)
        glued_params_history = glued_params_history.at[i + 1].set(glued_params)
        loss_history = loss_history.at[i].set(loss)
        return glued_params_history, loss_history, opt_state

    def iteration_without_history(i, carry):
        params, best_params, previous_loss, best_loss, opt_state = carry
        params, loss, opt_state = update_step(loss_and_grad, opt, opt_state, params)
        best_loss, best_params = lax.cond(loss < best_loss,
                                          lambda x: [loss, params],
                                          lambda x: [best_loss, best_params],
                                          None)
        return params, best_params, loss, best_loss, opt_state

    if opt_options.keep_history:
        glued_initial_params = glue_params(initial_params)
        glued_params_history = jnp.zeros((opt_options.num_iterations, len(glued_initial_params))).at[0].set(
            glued_initial_params)
        loss_history = jnp.zeros((opt_options.num_iterations,))

        glued_params_history, loss_history, _ = lax.fori_loop(
            0,
            opt_options.num_iterations,
            iteration_with_history,
            (glued_params_history, loss_history, opt_state))

        params_history = vmap(lambda gp: unglue_params(gp, slice_indices))(glued_params_history)
    else:
        initial_loss, _ = loss_and_grad(initial_params)
        params, best_params, loss, best_loss, _ = lax.fori_loop(
            0,
            opt_options.num_iterations,
            iteration_without_history,
            [initial_params, initial_params, initial_loss, initial_loss, opt_state])

        params_history = jnp.array([initial_params, best_params])
        loss_history = jnp.array([initial_loss, best_loss])

    return params_history, loss_history


def update_step(loss_and_grad, opt, opt_state, params):
    loss, grads = loss_and_grad(params)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, loss, opt_state


def glue_params(params):
    return jnp.concatenate(params)


def unglue_params(glued_params, slice_indices):
    return [glued_params[i0:i1] for i0, i1 in zip(slice_indices, slice_indices[1:])]


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
    def __init__(self, raw_results, loss_func, opt_options):
        self.all_results = [OptResult(r, loss_func, opt_options) for r in raw_results]
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
    def __init__(self, raw_result, loss_func, opt_options, best='loss'):

        self.loss_func = loss_func
        self.opt_options = opt_options
        self.params_history = raw_result[0]
        self.regloss_history = raw_result[1]

        if opt_options.regularization_func is not None:
            sample_params = self.params_history[0]
            self.reg_history = vmap(opt_options.regularization_func)(sample_params._make([jnp.array(p) for p in zip(*self.params_history)]))
            self.loss_history = self.regloss_history - self.reg_history
        else:
            self.reg_history = None
            self.loss_history = self.regloss_history

        assert best in ['loss', 'regloss'], 'best flag must be `loss` or `regloss`'
        if best == 'loss':
            self._best_i = jnp.argmin(self.loss_history)
        else:
            self._best_i = jnp.argmin(self.regloss_history)

        self.best_params = self.params_history[self._best_i]
        self.best_loss = self.loss_history[self._best_i]

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
