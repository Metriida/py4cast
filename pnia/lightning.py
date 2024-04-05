from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Tuple, Union

import einops
import pytorch_lightning as pl
import torch
from lightning.pytorch.utilities import rank_zero_only
from torchinfo import summary

from pnia.datasets.base import DatasetInfo, ItemBatch
from pnia.losses import ScaledL1Loss, ScaledRMSELoss, WeightedL1Loss, WeightedMSELoss
from pnia.models import build_model_from_settings, get_model_kls_and_settings
from pnia.models.base import expand_to_batch
from pnia.observer import (
    PredictionPlot,
    SpatialErrorPlot,
    StateErrorPlot,
    val_step_log_errors,
)


@dataclass
class ArLightningHyperParam:
    """
    Settings and hyperparameters for the lightning AR model.
    """

    dataset_info: DatasetInfo
    batch_size: int

    model_conf: Union[Path, None] = None
    model_name: str = "halfunet"

    lr: float = 0.1
    loss: str = "mse"

    num_input_steps: int = 2

    num_pred_steps_train: int = 2
    num_pred_steps_val_test: int = 2

    num_samples_to_plot: int = 2


@rank_zero_only
def rank_zero_init(model_kls, model_settings, statics):
    if hasattr(model_kls, "rank_zero_setup"):
        model_kls.rank_zero_setup(model_settings, statics)


class AutoRegressiveLightning(pl.LightningModule):
    """
    Auto-regressive lightning module for predicting meteorological fields.
    """

    def __init__(self, hparams: ArLightningHyperParam, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()
        hparams.dataset_info.summary()
        # Load static features for grid/data
        # We do not want to change dataset statics inplace
        # Otherwise their is some problem with transform_statics and parameters_saving
        # when relaoding from checkpoint
        statics = deepcopy(hparams.dataset_info.statics)

        # Keeping track of grid shape and N_interior
        self.grid_shape = statics.grid_shape
        self.N_interior = statics.N_interior

        # For making restoring of optimizer state optional (slight hack)
        self.opt_state = None

        # For example plotting
        self.plotted_examples = 0

        # For storing spatial loss maps during evaluation
        self.spatial_loss_maps = []

        # Set model input/output grid features based on dataset tensor shapes
        num_grid_static_features = statics.grid_static_features.values.shape[-1]

        # Compute the number of input features for the neural network
        # Should be directly supplied by datasetinfo ?

        num_input_features = (
            hparams.num_input_steps * hparams.dataset_info.weather_dim
            + num_grid_static_features
            + hparams.dataset_info.forcing_dim
        )

        num_output_features = hparams.dataset_info.weather_dim

        model_kls, model_settings = get_model_kls_and_settings(
            hparams.model_name, hparams.model_conf
        )

        rank_zero_init(model_kls, model_settings, statics)

        self.model, model_settings = build_model_from_settings(
            hparams.model_name,
            num_input_features,
            num_output_features,
            hparams.model_conf,
        )

        summary(self.model)

        # We transform and register the statics after the model has been set up
        # This change the dimension of all statics
        statics = self.model.transform_statics(statics)
        statics.register_buffers(self)

        self.register_buffer(
            "grid_static_features",
            expand_to_batch(statics.grid_static_features.values, hparams.batch_size),
            persistent=False,
        )
        # We need to instantiate the loss after statics had been transformed.
        # Indeed, the statics used should be in the right dimensions.
        # MSE loss, need to do reduction ourselves to get proper weighting
        if hparams.loss == "mse":
            self.loss = WeightedMSELoss(reduction="none")
        elif hparams.loss == "mae":
            self.loss = WeightedL1Loss(reduction="none")
        else:
            raise TypeError(f"Unknown loss function: {hparams.loss}")
        self.loss.prepare(statics)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams["hparams"].lr, betas=(0.9, 0.95)
        )
        if self.opt_state:
            opt.load_state_dict(self.opt_state)

        return opt

    def common_step(self, batch: ItemBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict on single batch
        init_states: (B,num_input_steps, N_grid, d_features)
        or (B, num_input_steps, Nlat,Nlon, d_features) depend on the grid
        target_states: (B, num_pred_steps, N_grid, d_features) or (B, num_pred_steps, Nlat,Nlon,, dfeatures)
        forcing_features: (B, num_pred_steps, N_grid, d_forcing), where index 0
            corresponds to index 1 of init_states -> Ngrid could also be (Nlat,Nlon)
        """
        (
            init_states,
            target_states,
            forcing_features,
        ) = self.model.transform_batch(batch)

        prev_states = init_states

        prediction_list = []

        # Here we do the autoregressive prediction looping
        # for the desired number of ar steps.
        for i in range(batch.num_pred_steps):

            forcing = forcing_features[:, i]
            border_state = target_states[:, i]

            x = torch.cat(
                [prev_states[:, idx] for idx in range(batch.num_input_steps)]
                + [
                    self.grid_static_features[: batch.batch_size],
                    forcing,
                ],
                dim=-1,
            )

            # Graph (B, N_grid, d_f) or Conv (B, N_lat,N_lon d_f)
            y = self.model(x)

            # We update the latest of our prev_states with the network output
            predicted_state = (
                prev_states[:, -1] + y * self.step_diff_std + self.step_diff_mean
            )

            # Overwrite border with true state
            new_state = (
                self.border_mask * border_state + self.interior_mask * predicted_state
            )
            prediction_list.append(new_state)

            # Only update the prev_states if we are not at the last step
            if i < batch.num_pred_steps - 1:
                # Update input states for next iteration: drop oldest, append new_state
                prev_states = torch.cat(
                    [prev_states[:, 1:], new_state.unsqueeze(1)], dim=1
                )

        prediction = torch.stack(
            prediction_list, dim=1
        )  # Stacking is done on time step. (B, pred_steps, N_grid, d_f) or (B, pred_steps, N_lat, N_lon, d_f)

        return prediction, target_states

    def on_train_start(self):
        self.train_plotters = []

    def training_step(self, batch: ItemBatch) -> torch.Tensor:
        """
        Train on single batch
        """
        prediction, target = self.common_step(batch)

        # Compute loss
        batch_loss = torch.mean(
            self.loss(prediction, target)
        )  # mean over unrolled times and batch

        log_dict = {"train_loss": batch_loss}
        self.log_dict(
            log_dict, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        # Notify every plotters
        for plotter in self.train_plotters:
            plotter.update(self, prediction=prediction, target=target)

        return batch_loss

    def on_validation_start(self):
        """
        Add some observers when starting validation
        """
        statics = deepcopy(self.hparams["hparams"].dataset_info.statics)
        statics = self.model.transform_statics(statics)
        l1_loss = ScaledL1Loss(reduction="none")
        l1_loss.prepare(statics)
        metrics = {"mae": l1_loss}
        self.valid_plotters = [StateErrorPlot(metrics, kind="Validation")]

    def validation_step(self, batch, batch_idx):
        """
        Run validation on single batch
        """
        prediction, target = self.common_step(batch)

        time_step_loss = torch.mean(
            self.loss(prediction, target), dim=0
        )  # (time_steps-1)

        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        val_log_dict = {
            f"val_loss_unroll{step}": time_step_loss[step - 1]
            for step in val_step_log_errors
        }
        val_log_dict["val_mean_loss"] = mean_loss
        self.log_dict(val_log_dict, on_step=False, on_epoch=True, sync_dist=True)

        # Notify every plotters
        for plotter in self.valid_plotters:
            plotter.update(self, prediction=prediction, target=target)

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch
        """
        # Notify every plotter that this is the end
        for plotter in self.valid_plotters:
            plotter.on_step_end(self)

    def on_test_start(self):
        """
        Attach observer when starting test
        """
        # this is very strange that we need to do that
        # Think at this statics problem (may be we just need to have a self.statics).
        statics = deepcopy(self.hparams["hparams"].dataset_info.statics)
        statics = self.model.transform_statics(statics)
        l1_loss = ScaledL1Loss(reduction="none")
        rmse_loss = ScaledRMSELoss(reduction="none")
        l1_loss.prepare(statics)
        rmse_loss.prepare(statics)
        metrics = {"mae": l1_loss, "rmse": rmse_loss}
        # I do not like settings things outside the __init__
        self.test_plotters = [
            StateErrorPlot(metrics),
            SpatialErrorPlot(),
            PredictionPlot(self.hparams["hparams"].num_samples_to_plot),
        ]

    def test_step(self, batch: ItemBatch, batch_idx):
        """
        Run test on single batch
        """
        prediction, target = self.common_step(batch)

        # Computing basic scores
        time_step_loss = torch.mean(
            self.loss(prediction, target), dim=0
        )  # (time_steps-1)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        test_log_dict = {
            f"test_loss_unroll{step}": time_step_loss[step - 1]
            for step in val_step_log_errors
        }
        test_log_dict["test_mean_loss"] = mean_loss
        self.log_dict(test_log_dict, on_step=False, on_epoch=True, sync_dist=True)

        # Notify plotters
        for plotter in self.test_plotters:
            plotter.update(self, prediction=prediction, target=target)

    @cached_property
    def interior_2d(self) -> torch.Tensor:
        """
        Get the interior mask as a 2d mask.
        Usefull when stored as 1D in statics.
        """
        if self.model.info.output_dim == 1:
            return einops.rearrange(
                self.interior_mask, "(x y) h -> x y h", x=self.grid_shape[0]
            )
        return self.interior_mask

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch.
        """
        # Notify plotters that the test epoch end
        for plotter in self.test_plotters:
            plotter.on_step_end(self)
