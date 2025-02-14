import math

import matplotlib.pyplot as plt
import numpy as np
import torch


class BetaSchedulerBase:
    def __init__(self, diffusion_steps, beta_min=None, beta_max=None):
        self.diffusion_steps = diffusion_steps
        self.beta_min = beta_min
        self.beta_max = beta_max

    def get_alpha_bars(self):
        betas = self.get_betas()
        alpha_bars = []
        alpha_bar = 1
        for beta in betas:
            alpha_bars.append(alpha_bar)
            alpha = 1 - beta
            alpha_bar = alpha * alpha_bar

        return torch.Tensor(alpha_bars)

    def get_alphas(self):
        return 1 - self.get_betas()

    def get_snr_weight(self):
        return self.get_snr()[:-1] - self.get_snr()[1:]

    def get_betas(self):
        raise NotImplementedError

    def get_snr(self):
        alpha_bars_squared = self.get_alpha_bars()  ** 2
        sigma_bars_squared = 1 - alpha_bars_squared

        snr = alpha_bars_squared / sigma_bars_squared
        return snr

    def plot(self):
        betas = self.get_betas()
        alpha_bars = self.get_alpha_bars()
        snr = self.get_snr()

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs = axs.flatten()
        fig.suptitle(self.__class__.__name__)

        axs[0].plot(betas)
        axs[0].set_title(r"$\beta_t$ schedule")
        axs[0].set_xlabel(r"$t$")
        axs[0].set_ylabel(r"$\beta$")

        axs[1].plot(alpha_bars)
        axs[1].set_title(r"$\bar\alpha_t$ schedule")
        axs[1].set_xlabel(r"$t$")
        axs[1].set_ylabel(r"$\bar\alpha_t$")
        axs[1].text(0, alpha_bars[-1], r"$\bar\alpha_T=$" + f"{alpha_bars[-1]:.3f}")

        axs[2].plot(snr)
        axs[2].set_title(r"SNR schedule")
        axs[2].set_xlabel(r"$t$")
        axs[2].set_ylabel(r"SNR")
        axs[2].text(0, snr[-1], r"$SNR_T=$" + f"{snr[-1]:.3f}" + "$SNR_0=$" + f"{snr[0]:.3f}")

        axs[3].plot((snr[:-1] - snr[1:]))
        axs[3].set_title(r'SNR "gradient"')
        axs[3].set_xlabel(r"$t$")

        fig.set_tight_layout(True)

        plt.show()


class LinearBetaScheduler(BetaSchedulerBase):
    def get_betas(self):
        betas = [
            self.beta_min + (t / self.diffusion_steps) * (self.beta_max - self.beta_min)
            for t in range(self.diffusion_steps)
        ]
        return torch.Tensor(betas)


class ExponentialBetaScheduler(BetaSchedulerBase):
    def get_betas(self):
        base = 1.1
        betas = [
            base
            ** (
                math.log(self.beta_min, base)
                + (t / self.diffusion_steps)
                * (math.log(self.beta_max, base) - math.log(self.beta_min, base))
            )
            for t in range(self.diffusion_steps)
        ]
        return torch.Tensor(betas)


class SigmoidalBetaScheduler(BetaSchedulerBase):
    def get_betas(self):
        ts = torch.linspace(-8, -4, self.diffusion_steps)
        betas = torch.sigmoid(ts)
        return betas


## below is lifted from Emiel
def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)
    alphas_step = alphas2[1:] / alphas2[:-1]
    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)
    return alphas2


class PolynomialBetaScheduler(BetaSchedulerBase):
    def __init__(self, diffusion_steps, beta_min=None, beta_max=None, power=2, s=1e-4):
        super().__init__(diffusion_steps, beta_min, beta_max)
        self.power = power
        self.s = s

    def get_betas(self):
        x = np.linspace(0, self.diffusion_steps, self.diffusion_steps)
        alphas2 = (1 - np.power(x / self.diffusion_steps, self.power)) ** 2
        alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)
        precision = 1 - 2 * self.s
        alphas2 = precision * alphas2 + self.s
        return 1.0 - torch.from_numpy(alphas2.astype(np.float32) ** 0.5)


if __name__ == "__main__":
    diffusion_steps = 10
    # beta_min = 1e-6
    beta_min = .00001
    beta_max = .02

    linear_beta_scheduler = LinearBetaScheduler(
        diffusion_steps, beta_min=beta_min, beta_max=beta_max
    )
    exponential_beta_scheduler = ExponentialBetaScheduler(
        diffusion_steps, beta_min=beta_min, beta_max=beta_max
    )
    sigmoidal_beta_scheduler = SigmoidalBetaScheduler(
        diffusion_steps, beta_min=beta_min, beta_max=beta_max
    )
    polynomial_beta_scheduler = PolynomialBetaScheduler(
        diffusion_steps, beta_min=beta_min, beta_max=beta_max
    )

    polynomial_beta_scheduler.plot()
    sigmoidal_beta_scheduler.plot()
    # exponential_beta_scheduler.plot()[ts]

    # linear_beta_scheduler.plot()

