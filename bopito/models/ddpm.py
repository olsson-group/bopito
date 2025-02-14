import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from torch import nn
from torch_scatter import scatter
from tqdm import tqdm

import wandb
from bopito.models import beta_schedule, dpm_solve, ema
from bopito.utils import mlops


class DDPMBase(pl.LightningModule):
    def __init__(
        self,
        score_model,
        diffusion_steps=1000,
        wandb_logger=None,
        lr=1e-3,
        noise_schedule="sigmoid",
        alpha_bar_weight=True,
        dont_evaluate=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.score_model = score_model
        self.diffusion_steps = diffusion_steps
        self.alpha_bar_weight = alpha_bar_weight
        self.dont_evaluate = dont_evaluate
        self.fails = 0

        if noise_schedule == "sigmoid":
            self.beta_scheduler = beta_schedule.SigmoidalBetaScheduler(diffusion_steps)
        if noise_schedule == "cosine":
            self.beta_scheduler = beta_schedule.CosineScheduler(diffusion_steps)
        if noise_schedule.startswith("polynomial"):
            split = noise_schedule.split("_")
            if len(split) == 1:
                power = 2
            else:
                power = float(split[1])

            self.beta_scheduler = beta_schedule.PolynomialScheduler(
                diffusion_steps, float(power)
            )

        self.register_buffer("betas", self.beta_scheduler.get_betas())
        self.register_buffer("alphas", self.beta_scheduler.get_alphas())
        self.register_buffer("alpha_bars", self.beta_scheduler.get_alpha_bars())

        self.ema = ema.ExponentialMovingAverage(
            self.score_model.parameters(), decay=0.99
        )
        self.wandb_logger = wandb_logger
        self.lr = lr
        self.last_evaluation = 10

    def on_after_backward(self):
        for _, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(
                        "detected inf or nan values in gradients. not updating model parameters",
                        flush=True,
                    )
                    self.fails += 1
                    wandb.log({"fails": self.fails})
                    if self.fails > 1000:
                        raise ValueError("Too many fails, stopping training")

                    self.zero_grad()
                    break

    def forward(self, batch):
        return self.score_model(batch)

    def training_step(self, batch, _):
        global_step = self.trainer.global_step

        loss, batch = self.get_loss(batch, return_batch=True)

        loss = loss.mean()

        if torch.isnan(loss):
            print(f"Loss is NaN at global_step {global_step}", flush=True)
            self.fails += 1
            wandb.log({"fails": self.fails})
            return None
            #  raise ValueError(f"Loss is NaN at global_step {global_step}")

        wandb.log({"global_step": global_step, "loss": loss})
        self.log("loss", loss, prog_bar=True)

        return loss


    def save_checkpoint(self, path):
        trainer = Trainer()
        trainer.strategy.connect(self)
        trainer.save_checkpoint(path)

    def on_before_zero_grad(self, *args, **kwargs):  # pylint: disable=unused-argument
        self.ema.update(self.score_model.parameters())

    def prepare_batch(self, batch):
        ts_diff = torch.randint(
            1, self.diffusion_steps, [len(batch["target"]), 1], device=self.device
        )

        batch["alpha_bars"] = self.alpha_bars[ts_diff]
        batch["ts_diff"] = ts_diff

        batch["epsilon"] = self.get_epsilon(batch)
        batch["corr"] = self.get_corrupted(batch)

        return batch

    def get_loss(self, batch, return_batch=False):
        batch = self.prepare_batch(batch)

        epsilon_hat = self.forward(batch)
        loss = self.calculate_loss(epsilon_hat, batch)

        if self.alpha_bar_weight:
            loss *= self.alpha_bars[batch["ts_diff"]].squeeze()

        if return_batch:
            return loss, batch
        return loss

    def get_epsilon(self, batch):
        raise NotImplementedError

    def get_corrupted(self, batch):
        raise NotImplementedError

    def calculate_loss(self, epsilon_hat, batch):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def log_error(self, batch, loss):
        checkpoint_path = "model.pt"
        batch_path = "batch.pt"

        self.save_checkpoint(checkpoint_path)
        torch.save(batch, batch_path)

        model_artifact = wandb.Artifact(name=f"{wandb.run.id}-last_model", type="model")
        model_artifact.add_file(checkpoint_path)
        wandb.run.log_artifact(model_artifact)

        batch_artifact = wandb.Artifact(name=f"{wandb.run.id}-last_batch", type="batch")
        batch_artifact.add_file(batch_path)
        wandb.run.log_artifact(batch_artifact)

        runid = wandb.run.id
        wandb.run.finish()

        epsilon_hat_ = self(batch)
        loss_ = self.calculate_loss(epsilon_hat_, batch)

        print("psum", sum(p.sum() for p in self.parameters()))
        print("run_id", runid)
        print(mlops.hash_obj(batch))
        print(mlops.hash_model(self))
        print(batch)
        print("loss", loss)
        print("loss_", loss_)
        print("NaN detected, logging error and stopping training")
        __import__("sys").exit()


class GeometricDDPM(DDPMBase):
    def get_epsilon(self, batch):
        epsilon = batch["target"].clone()
        epsilon.x = torch.randn(batch["target"].x.shape, device=self.device)
        return epsilon

    def get_epsilon_like(self, conf):
        epsilon = conf.clone()
        epsilon.x = torch.randn(conf.x.shape, device=self.device)
        return epsilon

    def get_corrupted(self, batch):
        alpha_bars = batch["alpha_bars"][batch["target"].batch]
        corrupted = batch["target"].clone()

        corrupted.x = (
            torch.sqrt(alpha_bars) * batch["target"].x
            + torch.sqrt(1 - alpha_bars) * batch["epsilon"].x
        )
        return corrupted

    def calculate_loss(self, epsilon_hat, batch):
        loss = nn.functional.mse_loss(
            epsilon_hat.x, batch["epsilon"].x, reduction="none"
        ).sum(-1)
        return scatter(loss, batch["target"].batch, reduce="mean")

    def sample_like(self, conf, ode_steps=0):
        corr = self.get_epsilon_like(conf)
        batch = {"corr": corr}
        return self.sample(batch, ode_steps)

    def sample_cond(self, cond, lag=1, ode_steps=0):
        corr = self.get_epsilon_like(cond)
        lag = torch.ones(len(cond), device=self.device) * lag
        batch = {"corr": corr, "cond": cond, "lag": lag}
        return self.sample(batch, ode_steps)

    def sample(self, batch, ode_steps=0):
        if ode_steps:
            return self._ode_sample(batch, ode_steps=ode_steps)
        return self._sample(batch)

    def _sample(self, batch):
        with torch.no_grad():
            for t_diff in tqdm(
                torch.arange(self.diffusion_steps - 1, 0, -1, dtype=torch.int64)
            ):
                ts_diff = (
                    torch.ones(
                        [len(batch["corr"].batch), 1],
                        device=self.device,
                        dtype=torch.int32,
                    )
                    * t_diff
                )

                batch["ts_diff"] = ts_diff
                batch = self.denoise_sample(batch)
        return batch["corr"]

    def denoise_sample(self, batch):  # , t, x, epsilon_hat):
        epsilon_hat = self.forward(batch)
        epsilon = self.get_epsilon_like(batch["corr"])

        preepsilon_scale = (self.alphas[batch["ts_diff"]]) ** (-0.5)
        epsilon_scale = (1 - self.alphas[batch["ts_diff"]]) / (
            1 - self.alpha_bars[batch["ts_diff"]]
        ) ** 0.5
        post_sigma = ((self.betas[batch["ts_diff"]]) ** 0.5) * epsilon.x
        batch["corr"].x = (
            preepsilon_scale * (batch["corr"].x - epsilon_scale * epsilon_hat.x)
            + post_sigma
        )

        return batch

    def _ode_sample(self, batch, ode_steps=100):
        noise_schedule = dpm_solve.NoiseScheduleVP(
            "discrete",
            betas=self.betas,
        )

        def t_diff_and_forward(x, t):
            t = t[0]
            batch["ts_diff"] = (
                torch.ones_like(batch["corr"].batch, device=self.device) * t
            )
            batch["corr"].x = x
            epsilon_hat = self.forward(batch)
            return epsilon_hat.x

        wrapped_model = dpm_solve.model_wrapper(t_diff_and_forward, noise_schedule)
        dpm_solver = dpm_solve.DPM_Solver(wrapped_model, noise_schedule)

        batch["corr"].x = dpm_solver.sample(batch["corr"].x, ode_steps)
        return batch["corr"]



class TensorDDPM(DDPMBase):
    def get_loss(self, batch):
        noise_batch, epsilon = self.get_noise_img_and_epsilon(batch)
        epsilon_hat = self.score_model(noise_batch)
        loss = nn.functional.mse_loss(epsilon_hat, epsilon)
        return loss

    def get_noise_img_and_epsilon(self, batch):
        ts = torch.randint(
            1, self.diffusion_steps, [len(batch["target"]), 1], device=self.device
        )
        
        epsilon = self.get_epsilon(batch)
        alpha_bars = self.alpha_bars[ts]

        noise_batch = dict(batch)
        noise_batch["t_diff"] = ts
        noise_batch["corr"] = (
            torch.sqrt(alpha_bars) * batch["target"]
            + torch.sqrt(1 - alpha_bars) * epsilon
        )
        return noise_batch, epsilon

    def get_epsilon(self, batch):
        epsilon = torch.randn(batch["target"].size(), device=self.device)
        return epsilon
    
    def get_epsilon_like(self, conf):
        epsilon = conf.clone()
        epsilon = torch.randn(conf.shape, device=self.device)
        return epsilon

    def sample(self, init_batch, ode_steps=0):
        #  batch = {}
        #  batch["corr"] = torch.randn((n_samples, 1), device=self.device)
        if ode_steps:
            return self._ode_sample(init_batch, ode_steps=ode_steps)
        return self._sample(init_batch)
    
    def sample_nested(self, init_batch, lag, step_lag=None, ode_steps=0):
        with torch.no_grad():
            print("Using nested sampling since lag is greater than max_lag. Lag schedule is ", end="", flush=True)
            #if step_lag is None:
            #    step_lag = self.score_model.max_lag
            #lag_schedule = [step_lag for _ in range(lag//step_lag)]
            #if lag%step_lag != 0:
            #    lag_schedule.append(lag%step_lag)
            lag_schedule = [lag-self.score_model.max_lag, self.score_model.max_lag]
            print(lag_schedule, flush=True)
            
            for lag_i in lag_schedule:
                init_batch["lag"] = torch.ones_like(init_batch["lag"], device=init_batch["lag"].device)*lag_i
                samples = self.sample(init_batch, ode_steps=ode_steps)
                init_batch["corr"] = torch.randn_like(init_batch["target"], device=samples.device)
                init_batch["cond"] = samples
        return samples
    
    def training_step(self, batch, _):
        loss = self.get_loss(batch)
        self.log("train/loss", loss, prog_bar=True)

        if torch.isnan(loss):
            raise ValueError("Loss is NaN")

        return loss
    
    def _ode_sample(self, batch, ode_steps=100):
        noise_schedule = dpm_solve.NoiseScheduleVP(
            "discrete",
            betas=self.betas,
        )

        def t_diff_and_forward(x, t):
            t = t[0]
            batch["t_diff"] = (
                torch.ones_like(batch["corr"], device=self.device) * t
            )
            batch["corr"]= x
            epsilon_hat = self.forward(batch)
            return epsilon_hat

        wrapped_model = dpm_solve.model_wrapper(t_diff_and_forward, noise_schedule)
        dpm_solver = dpm_solve.DPM_Solver(wrapped_model, noise_schedule)

        batch["corr"] = dpm_solver.sample(batch["corr"], ode_steps)
        return batch["corr"]
    
    def _sample(self, batch):
        with torch.no_grad():
            for t_diff in tqdm(
                torch.arange(self.diffusion_steps - 1, 0, -1, dtype=torch.int64)
            ):
                ts_diff = (
                    torch.ones(
                        [len(batch["corr"]), 1],
                        device=self.device,
                        dtype=torch.int32,
                    )
                    * t_diff
                )
                batch["t_diff"] = ts_diff
                batch = self.denoise_sample(batch)
        return batch["corr"]
    
    def denoise_sample(self, batch):  # , t, x, epsilon_hat):
        epsilon_hat = self.forward(batch)
        epsilon = self.get_epsilon_like(batch["corr"])

        preepsilon_scale = (self.alphas[batch["t_diff"]]) ** (-0.5)
        epsilon_scale = (1 - self.alphas[batch["t_diff"]]) / (
            1 - self.alpha_bars[batch["t_diff"]]
        ) ** 0.5
        post_sigma = ((self.betas[batch["t_diff"]]) ** 0.5) * epsilon
        batch["corr"] = (
            preepsilon_scale * (batch["corr"] - epsilon_scale * epsilon_hat)
            + post_sigma
        )

        return batch

        


#  class TensorTLDDPM():
#      def get_loss(self, batch):
#          noise_batch, epsilon = self.get_noise_img_and_epsilon(batch)
#          epsilon_hat = self.forward(noise_batch)
#          loss = nn.functional.mse_loss(epsilon_hat.x, epsilon.x, reduction="none").sum(
#              -1
#          )
#          loss = scatter(loss, batch.batch, reduce="mean").mean()
#
#          if torch.isnan(loss):
#              raise ValueError("Loss is NaN")
#
#          return loss
#


class DDPM(DDPMBase):
    def sample(self, n_samples, atom_number, ode_steps=0):
        batch = utils.get_batch_from_atom_number(atom_number, n_samples)
        batch = batch.to(self.device)

        if ode_steps:
            return self._ode_sample(
                batch, forward_callback=self.forward, ode_steps=ode_steps
            )
        return self._sample(batch, forward_callback=self.forward)

    def get_loss(self, batch):
        noise_batch, epsilon = self.get_noise_img_and_epsilon(batch)
        epsilon_hat = self.forward(noise_batch)
        loss = nn.functional.mse_loss(epsilon_hat.x, epsilon.x, reduction="none").sum(
            -1
        )
        loss = scatter(loss, batch.batch, reduce="mean").mean()

        if torch.isnan(loss):
            raise ValueError("Loss is NaN")

        return loss


class TLDDPM(DDPMBase):
    def sample(self, cond_batch, ode_steps=0):
        cond_batch.to(self.device)

        batch = cond_batch.clone()
        batch.x = torch.randn_like(batch.x, device=self.device)

        def forward_callback(batch):
            return self.forward(batch, cond_batch)

        if ode_steps:
            return self._ode_sample(batch, forward_callback, ode_steps=ode_steps)
        return self._sample(batch, forward_callback=forward_callback)

    def get_loss(self, batch):
        batch_0 = batch["batch_0"]
        batch_t = batch["batch_t"]

        noise_batch, epsilon = self.get_noise_img_and_epsilon(batch_t)
        epsilon_hat = self.forward(noise_batch, batch_0)
        loss = nn.functional.mse_loss(epsilon_hat.x, epsilon.x, reduction="none").sum(
            -1
        )
        loss = scatter(loss, noise_batch.batch, reduce="mean").mean()
        return loss


def mocklog(*args, **kwargs):
    return None
