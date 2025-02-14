import numpy as np
import torch


class NoiseSchedulerBase:
    def __init__(self, diffusion_steps, beta_min=None, beta_max=None):
        self.diffusion_steps = diffusion_steps
        self.beta_min = beta_min
        self.beta_max = beta_max

    def get_alphas(self):
        return 1 - self.get_betas()

    def get_betas(self):
        raise NotImplementedError


class AlphaBarScheduler(NoiseSchedulerBase):
    def get_betas(self):
        alpha_bars = self.get_alpha_bars()
        betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

        return betas

    def get_alpha_bars(self):
        raise NotImplementedError


class BetaScheduler(NoiseSchedulerBase):
    def get_alpha_bars(self):
        betas = self.get_betas()
        alpha_bars = []
        alpha_bar = 1
        for beta in betas:
            alpha_bars.append(alpha_bar)
            alpha = 1 - beta
            alpha_bar = alpha * alpha_bar

        return torch.Tensor(alpha_bars)

    def get_betas(self):
        raise NotImplementedError


class LinearBetaScheduler(BetaScheduler):
    def get_betas(self):
        betas = [
            self.beta_min + (t / self.diffusion_steps) * (self.beta_max - self.beta_min)
            for t in range(self.diffusion_steps)
        ]
        return torch.Tensor(betas)


class SigmoidalBetaScheduler(BetaScheduler):
    def get_betas(self):
        ts = torch.linspace(-8, -4, self.diffusion_steps)
        betas = torch.sigmoid(ts)
        return betas


class CosineScheduler(AlphaBarScheduler):
    def get_alpha_bars(self):
        s = 0.008
        nu = 1

        import matplotlib.pyplot as plt

        t = torch.linspace(0, self.diffusion_steps, self.diffusion_steps + 1)
        arg = ((t / self.diffusion_steps + s) ** nu) / (1 + s) * torch.pi / 2 / torch.pi

        return torch.cos(arg) ** 2


class SigmoidalScheduler(AlphaBarScheduler):
    def get_alpha_bars(self):
        t = torch.linspace(-5, 5, self.diffusion_steps + 1)
        alpha_bars = torch.sigmoid(-t)
        return alpha_bars


class PolynomialScheduler(AlphaBarScheduler):
    def __init__(self, diffusion_steps, power=2):
        super().__init__(diffusion_steps)
        self.power = power

    def get_alpha_bars(self):
        s = 1e-4
        steps = self.diffusion_steps + 1
        x = np.linspace(0, steps, steps)
        alphas2 = (1 - np.power(x / steps, self.power)) ** 2

        alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

        precision = 1 - 2 * s

        alphas2 = precision * alphas2 + s

        return torch.tensor(alphas2, dtype=torch.float32)


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


# pylint: disable=import-outside-toplevel, unused-variable
def main():
    import matplotlib.pyplot as plt

    s500 = PolynomialScheduler(1000, 3)
    s1000 = SigmoidalBetaScheduler(1000)

    alpha_bars500 = s500.get_alpha_bars()
    alpha_bars1000 = s1000.get_alpha_bars()

    import numpy as np

    plt.plot(np.linspace(0, 1, len(alpha_bars500)), alpha_bars500, label="500")
    plt.plot(np.linspace(0, 1, len(alpha_bars1000)), alpha_bars1000, label="1000")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
