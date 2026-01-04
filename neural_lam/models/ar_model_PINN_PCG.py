# Standard library
import os
import pickle
from typing import List, Union

# Third-party
import matplotlib.pyplot as plt
import numcodecs
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
import xarray as xr
from loguru import logger

# Local
from .. import metrics, vis
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from ..datastore.base import BaseRegularGridDatastore
from ..loss_weighting import get_state_feature_weighting
from ..weather_dataset import WeatherDataset


class ARModel(pl.LightningModule):
    """
    Generic auto-regressive weather model.
    Abstract class that can be extended.
    """

    # pylint: disable=arguments-differ
    # Disable to override args/kwargs from superclass

    def __init__(
        self,
        args,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
        datastore_boundary: Union[BaseDatastore, None],
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["datastore"])
        self.args = args
        self._datastore = datastore
        num_state_vars = datastore.get_num_data_vars(category="state")
        num_forcing_vars = datastore.get_num_data_vars(category="forcing")
        self.automatic_optimization = False

        num_past_forcing_steps = args.num_past_forcing_steps
        num_future_forcing_steps = args.num_future_forcing_steps

        # Load static features for interior
        da_static_features = datastore.get_dataarray(
            category="static", split=None, standardize=True
        )
        self.register_buffer(
            "interior_static_features",
            torch.tensor(da_static_features.values, dtype=torch.float32),
            persistent=False,
        )

        # Load stats for rescaling and weights
        da_state_stats = datastore.get_standardization_dataarray(
            category="state"
        )
        state_stats = {
            "state_mean": torch.tensor(
                da_state_stats.state_mean.values, dtype=torch.float32
            ),
            "state_std": torch.tensor(
                da_state_stats.state_std.values, dtype=torch.float32
            ),
            # Change stats below to be for diff of standardized variables
            "diff_mean": torch.tensor(
                da_state_stats.state_diff_mean.values
                / da_state_stats.state_std.values,
                dtype=torch.float32,
            ),
            "diff_std": torch.tensor(
                da_state_stats.state_diff_std.values
                / da_state_stats.state_std.values,
                dtype=torch.float32,
            ),
        }

        for key, val in state_stats.items():
            self.register_buffer(key, val, persistent=False)

        state_feature_weights = get_state_feature_weighting(
            config=config, datastore=datastore
        )
        self.feature_weights = torch.tensor(
            state_feature_weights, dtype=torch.float32
        )

        # Double grid output dim. to also output std.-dev.
        self.output_std = bool(args.output_std)
        if self.output_std:
            # Pred. dim. in grid cell
            self.grid_output_dim = 2 * num_state_vars
        else:
            # Pred. dim. in grid cell
            self.grid_output_dim = num_state_vars
            # Store constant per-variable std.-dev. weighting
            # NOTE that this is the inverse of the multiplicative weighting
            # in wMSE/wMAE
            self.register_buffer(
                "per_var_std",
                self.diff_std / torch.sqrt(self.feature_weights),
                persistent=False,
            )

        # interior from data + static
        (
            self.num_interior_nodes,
            interior_static_dim,
        ) = self.interior_static_features.shape
        self.num_total_grid_nodes = self.num_interior_nodes
        self.interior_dim = (
            2 * self.grid_output_dim
            + interior_static_dim
            + num_forcing_vars
            * (num_past_forcing_steps + num_future_forcing_steps + 1)
        )

        self.datastore_boundary = datastore_boundary

        # If datastore_boundary is given, the model is forced from the boundary
        self.boundary_forced = datastore_boundary is not None

        if self.boundary_forced:
            # Load static features for boundary
            da_boundary_static_features = datastore_boundary.get_dataarray(
                category="static", split=None, standardize=True
            )



            self.register_buffer(
                "boundary_static_features",
                torch.tensor(
                    da_boundary_static_features.values, dtype=torch.float32
                ),
                persistent=False,
            )

            # Compute dimensionalities (e.g. to instantiate MLPs)
            (
                self.num_boundary_nodes,
                boundary_static_dim,
            ) = self.boundary_static_features.shape

            # Compute boundary input dim separately
            num_boundary_forcing_vars = datastore_boundary.get_num_data_vars(
                category="forcing"
            )

            # Dimensionality of encoded time deltas
            self.time_delta_enc_dim = (
                args.hidden_dim
                if args.time_delta_enc_dim is None
                else args.time_delta_enc_dim
            )
            assert self.time_delta_enc_dim % 2 == 0, (
                "Number of dimensions to use for time delta encoding must be "
                "even (sin and cos)"
            )

            num_past_boundary_steps = args.num_past_boundary_steps
            num_future_boundary_steps = args.num_future_boundary_steps
            self.boundary_dim = (
                boundary_static_dim
                # Time delta counts as one additional forcing_feature
                + (num_boundary_forcing_vars + self.time_delta_enc_dim)
                * (num_past_boundary_steps + num_future_boundary_steps + 1)
            )
            # How many of the last boundary forcing dims contain time-deltas
            self.boundary_time_delta_dims = (
                num_past_boundary_steps + num_future_boundary_steps + 1
            )

            self.num_total_grid_nodes += self.num_boundary_nodes

        # Instantiate loss function
        self.loss = metrics.get_metric(args.loss)

        self.val_metrics = {
            "mse": [],
        }
        self.test_metrics = {
            "mse": [],
            "mae": [],
        }
        if self.output_std:
            self.test_metrics["output_std"] = []  # Treat as metric

        # For example plotting
        self.n_example_pred = args.n_example_pred
        self.plotted_examples = 0

        # For storing spatial loss maps during evaluation
        self.spatial_loss_maps = []

        # Set if grad checkpointing function should be used during rollout
        if args.grad_checkpointing:
            # Perform gradient checkpointing at each unrolling step
            self.unroll_ckpt_func = (
                lambda f, *args: torch.utils.checkpoint.checkpoint(
                    f, *args, use_reentrant=False
                )
            )
        else:
            self.unroll_ckpt_func = lambda f, *args: f(*args)

        # Store step length (h), taking subsampling into account
        self.step_length = datastore.step_length

        # Make WeatherDataset:s for being able to make tensor into xr.DA
        # Note: Unclear if it is actually necessary to make one per split?
        # TODO: creating an instance of WeatherDataset here on is
        # not how this should be done but whether WeatherDataset should be
        # provided to ARModel or where to put plotting still needs discussion
        self.wds_for_da = {
            split: WeatherDataset(
                datastore=self._datastore,
                datastore_boundary=None,
                split=split,
            )
            for split in ("train", "val", "test")
        }

        # Which variables to plot during eval
        self.plot_vars = args.plot_vars
        for var in self.plot_vars:
            assert var in self._datastore.get_vars_names(
                "state"
            ), f"Can not plot variable {var}: not in datastore"

    def _create_dataarray_from_tensor(
        self,
        tensor: torch.Tensor,
        time: Union[int, List[int]],
        split: str,
        category: str,
    ) -> xr.DataArray:
        """
        Create an `xr.DataArray` from a tensor, with the correct dimensions and
        coordinates to match the datastore used by the model. This function in
        in effect is the inverse of what is returned by
        `WeatherDataset.__getitem__`.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to convert to a `xr.DataArray` with dimensions [time,
            grid_index, feature]. The tensor will be copied to the CPU if it is
            not already there.
        time : Union[int,List[int]]
            The time index or indices for the data, given as integers or a list
            of integers representing epoch time in nanoseconds. The ints will be
            copied to the CPU memory if they are not already there.
        split : str
            The split of the data, either 'train', 'val', or 'test'
        category : str
            The category of the data, either 'state' or 'forcing'
        """
        # Move to CPU if on GPU
        time = time.detach().cpu()
        time = np.array(time, dtype="datetime64[ns]")

        tensor = tensor.detach().cpu()
        weather_dataset = self.wds_for_da[split]
        da = weather_dataset.create_dataarray_from_tensor(
            tensor=tensor, time=time, category=category
        )
        return da

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.args.lr, betas=(0.9, 0.95)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.args.epochs,
            eta_min=self.args.min_lr if hasattr(self.args, "min_lr") else 0.0,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    @staticmethod
    def expand_to_batch(x, batch_size):
        """
        Expand tensor with initial batch dimension
        """
        return x.unsqueeze(0).expand(batch_size, -1, -1)

    def predict_step(
        self, prev_state, prev_prev_state, forcing, boundary_forcing
    ):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_interior_nodes, feature_dim), X_t
        prev_prev_state: (B, num_interior_nodes, feature_dim), X_{t-1}
        forcing: (B, num_interior_nodes, forcing_dim)
        boundary_forcing: (B, num_boundary_nodes, boundary_forcing_dim)
        """
        raise NotImplementedError("No prediction step implemented")

    def unroll_prediction(self, init_states, forcing, boundary_forcing):
        """
        Roll out prediction taking multiple autoregressive steps with model
        init_states: (B, 2, num_interior_nodes, d_f)
        forcing: (B, pred_steps, num_interior_nodes, d_static_f)
        boundary_forcing: (B, pred_steps, num_boundary_nodes, d_boundary_f)
        """
        prev_prev_state = init_states[:, 0]
        prev_state = init_states[:, 1]
        prediction_list = []
        pred_std_list = []
        pred_steps = forcing.shape[1]

        for i in range(pred_steps):
            forcing_step = forcing[:, i]

            if self.boundary_forced:
                boundary_forcing_step = boundary_forcing[:, i]
            else:
                boundary_forcing_step = None

            pred_state, pred_std = self.unroll_ckpt_func(
                self.predict_step,
                prev_state,
                prev_prev_state,
                forcing_step,
                boundary_forcing_step,
            )
            # state: (B, num_interior_nodes, d_f)
            # pred_std: (B, num_interior_nodes, d_f) or None

            prediction_list.append(pred_state)

            if self.output_std:
                pred_std_list.append(pred_std)

            # Update conditioning states
            prev_prev_state = prev_state
            prev_state = pred_state

        prediction = torch.stack(
            prediction_list, dim=1
        )  # (B, pred_steps, num_interior_nodes, d_f)
        if self.output_std:
            pred_std = torch.stack(
                pred_std_list, dim=1
            )  # (B, pred_steps, num_interior_nodes, d_f)
        else:
            pred_std = self.per_var_std  # (d_f,)

        return prediction, pred_std

    def common_step(self, batch):
        """
        Predict on single batch
        batch consists of:
        init_states: (B, 2, num_interior_nodes, d_features)
        target_states: (B, pred_steps, num_interior_nodes, d_features)
        forcing: (B, pred_steps, num_interior_nodes, d_forcing),
        boundary_forcing:
            (B, pred_steps, num_boundary_nodes, d_boundary_forcing),
            where index 0 corresponds to index 1 of init_states
        """
        (
            init_states,
            target_states,
            forcing,
            boundary_forcing,
            batch_times,
        ) = batch

        prediction, pred_std = self.unroll_prediction(
            init_states, forcing, boundary_forcing
        )  # (B, pred_steps, num_interior_nodes, d_f)
        # prediction: (B, pred_steps, num_interior_nodes, d_f) pred_std: (B,
        # pred_steps, num_interior_nodes, d_f) or (d_f,)

        return prediction, target_states, pred_std, batch_times

    def training_step(self, batch, batch_idx):

        opt = self.optimizers()
        opt.zero_grad()
        
        prediction, target, pred_std, batch_times = self.common_step(batch)

        # your existing data loss (e.g. standardized MSE)
        time_step_loss = torch.mean(
            self.loss(prediction, target, pred_std), dim=0
        )
        data_loss = torch.mean(time_step_loss)

        # NEW: physics loss
        phys_loss = self.navier_stokes_loss(
            prediction=prediction,
            batch_times=batch_times,
            split="train",
            nu=0.01,
        )

        params = [p for p in self.parameters() if p.requires_grad]


        self.manual_backward(data_loss, retain_graph=True)
        grads_data = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) 
                    for p in params]
        for p in params:
            p.grad = None

        
        self.manual_backward(phys_loss, retain_graph=True)
        grads_phys = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) 
                    for p in params]
        for p in params:
            p.grad = None

        
        gD_flat = torch.cat([g.view(-1) for g in grads_data])
        gP_flat = torch.cat([g.view(-1) for g in grads_phys])

        dot = torch.dot(gD_flat, gP_flat)

        if dot < 0:
            if torch.rand(()) < 0.5:
                denom = (gP_flat.pow(2).sum() + 1e-12)
                proj = dot / denom
                grads_data = [gD - proj * gP for gD, gP in zip(grads_data, grads_phys)]
            else:
                denom = (gD_flat.pow(2).sum() + 1e-12)
                proj = dot / denom
                grads_phys = [gP - proj * gD for gD, gP in zip(grads_data, grads_phys)]

        final_grads = [gD + gP for gD, gP in zip(grads_data, grads_phys)]


        for p, g in zip(params, final_grads):
            p.grad = g


        opt.step()


        total_loss_for_logging = data_loss + phys_loss

        self.log("train_data_loss", data_loss,
                on_step=True, on_epoch=True, prog_bar=True, batch_size=batch[0].shape[0])
        self.log("train_phys_loss", phys_loss,
                on_step=True, on_epoch=True, prog_bar=True, batch_size=batch[0].shape[0])
        self.log("train_total_loss", total_loss_for_logging,
                on_step=True, on_epoch=True, prog_bar=True, batch_size=batch[0].shape[0])

        return total_loss_for_logging.detach()

    def _build_physics_grid(self, split: str = "train"):
        """
        Build and cache:
          - x, y coordinates in meters (1D)
          - a mapping from flattened (y, x) -> original grid_index

        This uses xarray only once, on a dummy tensor, and does NOT touch
        the actual predictions (so it doesn't affect gradients).
        """
        if hasattr(self, "_phys_idx_flat"):
            # Already built
            return

        num_nodes = self.num_interior_nodes
        var_names = self._datastore.get_vars_names(category="state")
        d_f = len(var_names)

        # Dummy tensor: encode grid_index in channel 0
        dummy_tensor = torch.zeros(1, num_nodes, d_f, dtype=torch.float32)
        dummy_tensor[0, :, 0] = torch.arange(num_nodes, dtype=torch.float32)

        # Dummy time (epoch ns) – content doesn't matter, just shape
        dummy_time = torch.tensor([0], dtype=torch.int64)

        # Create DataArray with correct coords
        da = self._create_dataarray_from_tensor(
            tensor=dummy_tensor,
            time=dummy_time,
            split=split,
            category="state",
        )

        # Unstack grid_index -> (x, y)
        if isinstance(self._datastore, BaseRegularGridDatastore):
            da = self._datastore.unstack_grid_coords(da)

        # At this point dims are like: (time, state_feature, x, y)
        # 1) Coordinates in meters
        deg_to_m = 111_000.0
        y_deg = da.coords["y"].values
        x_deg = da.coords["x"].values

        y_m = y_deg * deg_to_m
        x_m = x_deg * deg_to_m * np.cos(np.deg2rad(y_deg.mean().item()))

        # 2) Index map: value we stored in channel 0 is the original grid_index
        idx_xy = da.isel(time=0, state_feature=0).values.astype(np.int64)  # shape (Nx, Ny)
        # We want (Ny, Nx) with y as first spatial dim
        idx_yx = idx_xy.T  # (Ny, Nx)

        # Cache as torch tensors (no grad needed)
        self._phys_y_m = torch.tensor(y_m, dtype=torch.float32)      # (Ny,)
        self._phys_x_m = torch.tensor(x_m, dtype=torch.float32)      # (Nx,)
        self._phys_idx_flat = torch.tensor(
            idx_yx.reshape(-1), dtype=torch.long
        )
    
    @staticmethod
    def _time_derivative(u: torch.Tensor, t_s: torch.Tensor) -> torch.Tensor:
        """
        u: (T, Ny, Nx)
        t_s: (T,) time in seconds (float32) on same device as u
        """
        T = u.shape[0]
        ut = torch.zeros_like(u)

        if T <= 1:
            return ut

        # assume (roughly) uniform dt
        dt = (t_s[1] - t_s[0])

        # central diff for interior, forward/backward for ends
        ut[1:-1] = (u[2:] - u[:-2]) / (2.0 * dt)
        ut[0] = (u[1] - u[0]) / dt
        ut[-1] = (u[-1] - u[-2]) / dt

        return ut

    @staticmethod
    def _spatial_derivatives(u: torch.Tensor, dx: float, dy: float):
        """
        u: (T, Ny, Nx)

        Returns:
            u_x, u_xx, u_y, u_yy  (each (T, Ny, Nx))
        """
        u_x = torch.zeros_like(u)
        u_y = torch.zeros_like(u)
        u_xx = torch.zeros_like(u)
        u_yy = torch.zeros_like(u)

        # ----- derivative in x (axis=2) -----
        # first derivative
        u_x[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) / (2.0 * dx)
        u_x[:, :, 0] = (u[:, :, 1] - u[:, :, 0]) / dx
        u_x[:, :, -1] = (u[:, :, -1] - u[:, :, -2]) / dx

        # second derivative
        u_xx[:, :, 1:-1] = (u[:, :, 2:] - 2.0 * u[:, :, 1:-1] + u[:, :, :-2]) / (
            dx * dx
        )
        u_xx[:, :, 0] = (u[:, :, 1] - 2.0 * u[:, :, 0] + u[:, :, 0]) / (dx * dx)
        u_xx[:, :, -1] = (
            u[:, :, -1] - 2.0 * u[:, :, -1] + u[:, :, -2]
        ) / (dx * dx)

        # ----- derivative in y (axis=1) -----
        u_y[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) / (2.0 * dy)
        u_y[:, 0, :] = (u[:, 1, :] - u[:, 0, :]) / dy
        u_y[:, -1, :] = (u[:, -1, :] - u[:, -2, :]) / dy

        u_yy[:, 1:-1, :] = (u[:, 2:, :] - 2.0 * u[:, 1:-1, :] + u[:, :-2, :]) / (
            dy * dy
        )
        u_yy[:, 0, :] = (u[:, 1, :] - 2.0 * u[:, 0, :] + u[:, 0, :]) / (dy * dy)
        u_yy[:, -1, :] = (
            u[:, -1, :] - 2.0 * u[:, -1, :] + u[:, -2, :]
        ) / (dy * dy)

        return u_x, u_xx, u_y, u_yy
    
    def navier_stokes_loss(
    self,
    prediction: torch.Tensor,
    batch_times: torch.Tensor,
    split: str = "train",
    nu: float = 0.01,
    ) -> torch.Tensor:
        """
        Torch-differentiable NV loss using *all* forecast frames.

        prediction : (B, pred_steps, num_interior_nodes, d_f), standardized
        batch_times : (B, pred_steps), epoch time in ns (int64)
        split : which split's grid/coords to use for building x,y

        Returns
        -------
        phys_loss : torch.Tensor scalar (requires_grad=True)
        """
        device = prediction.device
        dtype = prediction.dtype

        # 1) Ensure we have grid geometry & index mapping
        self._build_physics_grid(split=split)

        # Move cached coords/mapping to the right device
        x_m = self._phys_x_m.to(device=device, dtype=dtype)  # (Nx,)
        y_m = self._phys_y_m.to(device=device, dtype=dtype)  # (Ny,)
        idx_flat = self._phys_idx_flat.to(device=device)     # (Ny*Nx,)

        Ny = y_m.shape[0]
        Nx = x_m.shape[0]

        
        dx = float(x_m[1] - x_m[0]) if Nx > 1 else 1.0
        dy = float(y_m[1] - y_m[0]) if Ny > 1 else 1.0

        
        pred_rescaled = prediction * self.state_std + self.state_mean
        B, T, _, _ = pred_rescaled.shape

        
        external_states_std = self._get_external_states_for_batch(
            batch_times=batch_times,
            split=split,
        )
        if external_states_std is not None:
            external_rescaled = external_states_std * self.state_std + self.state_mean
        else:
            external_rescaled = None


        var_names = self._datastore.get_vars_names(category="state")
        u_idx = var_names.index("U_10M")
        v_idx = var_names.index("V_10M")
        p_idx = var_names.index("PS")

        nu_torch = torch.tensor(nu, dtype=dtype, device=device)
        
        batch_losses = []

        for b in range(B):

            pred_b = pred_rescaled[b]      # (T, G, F)
            times_b = batch_times[b]       # (T,)
            if external_rescaled is not None:
                ext_b = external_rescaled[b]  # (T, G, F)
            else:
                ext_b = None

            t_s = (times_b - times_b[0]).to(device=device, dtype=dtype) * 1e-9

            
            u_nodes = pred_b[:, :, u_idx]
            v_nodes = pred_b[:, :, v_idx]
            p_nodes = pred_b[:, :, p_idx]

            
            u_grid = u_nodes[:, idx_flat].view(T, Ny, Nx)
            v_grid = v_nodes[:, idx_flat].view(T, Ny, Nx)
            p_grid = p_nodes[:, idx_flat].view(T, Ny, Nx)

           
            if ext_b is not None:
                u_ext_nodes = ext_b[:, :, u_idx]
                v_ext_nodes = ext_b[:, :, v_idx]
                p_ext_nodes = ext_b[:, :, p_idx]

                u_ext_grid = u_ext_nodes[:, idx_flat].view(T, Ny, Nx)
                v_ext_grid = v_ext_nodes[:, idx_flat].view(T, Ny, Nx)
                p_ext_grid = p_ext_nodes[:, idx_flat].view(T, Ny, Nx)
            else:
                u_ext_grid = v_ext_grid = p_ext_grid = None

           
            u_t = self._time_derivative(u_grid, t_s)
            v_t = self._time_derivative(v_grid, t_s)
            u_x, u_xx, u_y, u_yy = self._spatial_derivatives(u_grid, dx, dy)
            v_x, v_xx, v_y, v_yy = self._spatial_derivatives(v_grid, dx, dy)
            p_x, _, p_y, _ = self._spatial_derivatives(p_grid, dx, dy)

            
            f_u = u_t + (u_grid * u_x + v_grid * u_y) + p_x - nu_torch * (u_xx + u_yy)
            f_v = v_t + (u_grid * v_x + v_grid * v_y) + p_y - nu_torch * (v_xx + v_yy)
            f_e = u_x + v_y

            #Lateral boundary relaxation conditions
            if u_ext_grid is not None:
                
                N_field_2d, D_field_2d, mask_core, mask_buffer, edge_mask = \
                    self._build_relaxation_profiles(
                        Ny=Ny,
                        Nx=Nx,
                        device=device,
                        dtype=dtype,
                    )
                
                N_field = N_field_2d.unsqueeze(0).expand(T, -1, -1)
                D_field = D_field_2d.unsqueeze(0).expand(T, -1, -1)

                
                delta_u = u_grid - u_ext_grid         # (T, Ny, Nx)
                delta_v = v_grid - v_ext_grid

                
                _, delta_uxx, _, delta_uyy = self._spatial_derivatives(
                    delta_u, dx, dy
                )
                lap_delta_u = delta_uxx + delta_uyy

                _, delta_vxx, _, delta_vyy = self._spatial_derivatives(
                    delta_v, dx, dy
                )
                lap_delta_v = delta_vxx + delta_vyy

               
                f_u = f_u + N_field * delta_u - D_field * lap_delta_u
                f_v = f_v + N_field * delta_v - D_field * lap_delta_v

            
            f_u_loss = (f_u ** 2).mean()
            f_v_loss = (f_v ** 2).mean()
            f_e_loss = (f_e ** 2).mean()
            loss_b = f_u_loss + f_v_loss + f_e_loss
            batch_losses.append(loss_b)


        # Mean over batch
        phys_loss = torch.stack(batch_losses).mean()
        return phys_loss
    
    def _build_external_state_cache(self, split: str = "train"):
        """
        Loads large-scale fields (datastore_boundary) and remaps them to the internal grid. Caches the result for a given split.

        After calling, we will have, for example:
        self._ext_state_da[split]: xr.DataArray
        dims: (time, grid_index, state_feature)
        on the same internal grid as self._datastore
        """
        if not self.boundary_forced:
            return

        
        if hasattr(self, "_ext_state_da") and split in self._ext_state_da:
            return

        if not hasattr(self, "_ext_state_da"):
            self._ext_state_da = {}

        
        da_int_state = self._datastore.get_dataarray(
            category="state", split=split, standardize=False
        )
        da_ext = self.datastore_boundary.get_dataarray(
            category="forcing",  
            split=split,
            standardize=False,
        )


        if isinstance(self._datastore, BaseRegularGridDatastore):
            da_int_unstack = self._datastore.unstack_grid_coords(da_int_state)
        else:
            raise RuntimeError("internal datastore is not a BaseRegularGridDatastore")

        if isinstance(self.datastore_boundary, BaseRegularGridDatastore):
            da_ext_unstack = self.datastore_boundary.unstack_grid_coords(da_ext)
        else:
            raise RuntimeError("datastore_boundary is not a BaseRegularGridDatastore")

        
        da_int_unstack = da_int_unstack.transpose("time", "state_feature", "x", "y")
        x_int = da_int_unstack.coords["x"]
        y_int = da_int_unstack.coords["y"]


        if "state_feature" in da_ext_unstack.dims:
            ext_feature_dim = "state_feature"
        elif "forcing_feature" in da_ext_unstack.dims:
            ext_feature_dim = "forcing_feature"
        elif "feature" in da_ext_unstack.dims:
            ext_feature_dim = "feature"
        else:

            ext_feature_dim = [d for d in da_ext_unstack.dims if d != "time"][0]


        if "x" in da_ext_unstack.dims and "y" in da_ext_unstack.dims:
            ext_x_dim, ext_y_dim = "x", "y"
        elif "longitude" in da_ext_unstack.dims and "latitude" in da_ext_unstack.dims:

            ext_x_dim, ext_y_dim = "longitude", "latitude"
        else:

            ext_y_dim, ext_x_dim = da_ext_unstack.dims[-2], da_ext_unstack.dims[-1]


        da_ext_unstack = da_ext_unstack.transpose("time", ext_feature_dim, ext_x_dim, ext_y_dim)


        da_ext_on_int = da_ext_unstack.interp(
            {ext_x_dim: x_int, ext_y_dim: y_int},
            method="nearest",
        )

        da_ext_on_int = da_ext_on_int.stack(grid_index=("x", "y"))
        da_ext_on_int = da_ext_on_int.transpose("time", "grid_index", ext_feature_dim)


        int_var_names = list(self._datastore.get_vars_names(category="state"))
        ext_var_names = list(self.datastore_boundary.get_vars_names(category="forcing"))


        name_map_int_to_ext = {
            "U_10M": "10m_u_component_of_wind",
            "V_10M": "10m_v_component_of_wind",
            "PS": "surface_pressure",
        }

        int_idx = []
        ext_idx = []
        for int_name, ext_name in name_map_int_to_ext.items():
            if int_name in int_var_names and ext_name in ext_var_names:
                int_idx.append(int_var_names.index(int_name))
                ext_idx.append(ext_var_names.index(ext_name))

        import numpy as np
        from loguru import logger

        if len(int_idx) == 0:

            logger.warning(
                "Aucun recouvrement de variables entre datastore interne "
                "et boundary pour la relaxation."
            )
            da_ext_remapped = xr.DataArray(
                np.zeros(
                    (
                        da_ext_on_int.sizes["time"],
                        da_ext_on_int.sizes["grid_index"],
                        len(int_var_names),
                    ),
                    dtype=np.float32,
                ),
                dims=("time", "grid_index", "state_feature"),
                coords={
                    "time": da_ext_on_int.coords["time"],
                    "grid_index": da_int_state.coords["grid_index"],
                    "state_feature": int_var_names,
                },
            )
            self._ext_state_da[split] = da_ext_remapped
            return


        da_ext_sel = da_ext_on_int.isel({ext_feature_dim: ext_idx}) 


        d_f_int = len(int_var_names)
        data = np.zeros(
            (
                da_ext_sel.sizes["time"],
                da_ext_sel.sizes["grid_index"],
                d_f_int,
            ),
            dtype=np.float32,
        )

        data[:, :, int_idx] = da_ext_sel.values

        da_ext_remapped = xr.DataArray(
            data,
            dims=("time", "grid_index", "state_feature"),
            coords={
                "time": da_ext_sel.coords["time"],
                "grid_index": da_int_state.coords["grid_index"],
                "state_feature": int_var_names,
            },
        )

        self._ext_state_da[split] = da_ext_remapped


    

    def _get_external_states_for_batch(
    self,
    batch_times: torch.Tensor,  # (B, T) en ns
    split: str = "train",
) -> Union[None, torch.Tensor]:
        """
        Renvoie external_states : (B, T, num_interior_nodes, d_f), standardisé.

        Si self.has_boundary_states == False -> retourne None.
        """
        if not self.boundary_forced:
            return None

        self._build_external_state_cache(split=split)
        da_ext = self._ext_state_da[split]  # (time, grid_index, state_feature)


        times_np = batch_times.detach().cpu().numpy().astype("datetime64[ns]")
        B, T = times_np.shape

        ext_list = []
        for b in range(B):

            da_sel = da_ext.sel(time=times_np[b], method="nearest")  # (T, grid_index, state_feature)
            ext_list.append(torch.from_numpy(da_sel.values))  # (T, G, F)

        external = torch.stack(ext_list, dim=0)  # (B, T, G, F)
        external = external.to(batch_times.device).to(torch.float32)


        external = torch.nan_to_num(
            external,
            nan=0.0,    
            posinf=0.0,
            neginf=0.0,
        )


        state_std_safe = self.state_std.clone()
        state_std_safe[state_std_safe == 0] = 1.0  

        external_std = (external - self.state_mean) / state_std_safe


        external_std = torch.nan_to_num(
            external_std,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return external_std

    def _build_relaxation_profiles(
        self,
        Ny: int,
        Nx: int,
        device: torch.device,
        dtype: torch.dtype,
        ):
        """
        Builds the 2D fields:
        - N_field_2d(y,x): Newtonian relaxation coefficient (s⁻¹)
        - D_field_2d(y,x): diffusive coefficient (m² s⁻¹)
        as well as the core/buffer/edge masks.

                The idea:
                * buffer zone with thickness W = self.args.relax_width
                * N(x) = 0 in the core, increases towards the edge (linear or exp profile)
                * option: stronger factor on the edge row (edge_factor)
                * D(x) proportional to N(x) (reasonable simplification)

        Translated with DeepL.com (free version)
        """
        W = 10


        mask_core = torch.zeros((Ny, Nx), dtype=torch.bool, device=device)
        if W > 0 and 2 * W < min(Ny, Nx):
            mask_core[W:Ny - W, W:Nx - W] = True
        mask_buffer = ~mask_core


        iy = torch.arange(Ny, device=device).view(Ny, 1)
        ix = torch.arange(Nx, device=device).view(1, Nx)
        dist_y = torch.minimum(iy, Ny - 1 - iy)
        dist_x = torch.minimum(ix, Nx - 1 - ix)
        dist_edge = torch.minimum(dist_y, dist_x)  # (Ny, Nx)

        edge_mask = (dist_edge == 0)


        if W > 0:
            dist_buffer = dist_edge.clamp(max=W - 1).float()

            r = (W - 1 - dist_buffer) / max(W - 1, 1)
            r = r * mask_buffer.float()
        else:
            r = torch.zeros((Ny, Nx), dtype=torch.float32, device=device)


        profile_type = getattr(self.args, "relax_profile", "linear")
        if profile_type == "exp":

            gamma = float(getattr(self.args, "relax_exp_gamma", 3.0))

            num = torch.expm1(gamma * r)
            den = torch.expm1(torch.tensor(gamma, device=device))
            shape_N = num / (den + 1e-12)
        else:

            shape_N = r


        alpha_max = 0.7  # ex: 1 / tau
        N_field_2d = alpha_max * shape_N  # (Ny, Nx)


        edge_factor = float(getattr(self.args, "relax_edge_factor", 1.0))
        if edge_factor != 1.0:
            N_field_2d = torch.where(
                edge_mask,
                edge_factor * alpha_max,
                N_field_2d,
            )


        D_max = float(getattr(self.args, "relax_diff_coef", 0.0))
        D_field_2d = D_max * shape_N  # (Ny, Nx)

        N_field_2d = N_field_2d.to(dtype=dtype)
        D_field_2d = D_field_2d.to(dtype=dtype)

        return N_field_2d, D_field_2d, mask_core, mask_buffer, edge_mask



    def all_gather_cat(self, tensor_to_gather):
        """
        Gather tensors across all ranks, and concatenate across dim. 0 (instead
        of stacking in new dim. 0)

        tensor_to_gather: (d1, d2, ...), distributed over K ranks

        returns: (K*d1, d2, ...)
        """
        return self.all_gather(tensor_to_gather).flatten(0, 1)

    # newer lightning versions requires batch_idx argument, even if unused
    # pylint: disable-next=unused-argument
    def validation_step(self, batch, batch_idx):
        """
        Run validation on single batch
        """
        prediction, target, pred_std, _ = self.common_step(batch)

        time_step_loss = torch.mean(
            self.loss(
                prediction,
                target,
                pred_std,
            ),
            dim=0,
        )  # (time_steps-1)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        val_log_dict = {
            f"val_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.args.val_steps_to_log
            if step <= len(time_step_loss)
        }
        val_log_dict["val_mean_loss"] = mean_loss
        self.log_dict(
            val_log_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )

        # Store MSEs
        entry_mses = metrics.mse(
            prediction,
            target,
            pred_std,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.val_metrics["mse"].append(entry_mses)

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch
        """
        # Create error maps for all test metrics
        self.aggregate_and_plot_metrics(self.val_metrics, prefix="val")

        # Clear lists with validation metrics values
        for metric_list in self.val_metrics.values():
            metric_list.clear()

    def _save_predictions_to_zarr(
        self,
        batch_times: torch.Tensor,
        batch_predictions: torch.Tensor,
        batch_idx: int,
        zarr_output_path: str,
    ):
        """
        Save state predictions for single batch to zarr dataset. Will append to
        existing dataset for batch_idx > 0. Resulting dataset will contain a
        variable named `state` with coordinates (start_time,
        elapsed_forecast_duration, grid_index, state_feature).
        Parameters
        ----------
        batch_times : torch.Tensor[int]
            The times for the batch, given as epoch time in nanoseconds. Shape
            is (B, args.pred_steps) where B is the batch size and
            args.pred_steps is the number of prediction steps.
        batch_predictions : torch.Tensor[float]
            The predictions for the batch, given as (B, args.pred_steps,
            num_grid_nodes, d_f) where B is the batch size, args.pred_steps is
            the number of prediction steps, num_grid_nodes is the number of
            grid nodes, and d_f is the number of state features.
        batch_idx : int
            The index of the batch in the current epoch.
        """
        # Scale predictions back to original data scale
        batch_predictions_rescaled = (
            batch_predictions * self.state_std + self.state_mean
        )

        # Convert predictions to DataArray using _create_dataarray_from_tensor
        das_pred = []
        for i in range(len(batch_times)):
            da_pred = self._create_dataarray_from_tensor(
                tensor=batch_predictions_rescaled[i],
                time=batch_times[i],
                split="test",
                category="state",
            )
            # Unstack grid coords if necessary, this also avoids the need to
            # try to store a MultiIndex zarr dataset which is not supported by
            # xarray
            if isinstance(self._datastore, BaseRegularGridDatastore):
                da_pred = self._datastore.unstack_grid_coords(da_pred)

            # First entry in da_pred.coords["time"] is time of first prediction,
            # so init time of forecast is one time step before
            t0 = da_pred.coords["time"].values[0] - np.array(
                self.step_length, dtype="timedelta64[h]"
            )
            da_pred.coords["start_time"] = t0
            da_pred.coords["elapsed_forecast_duration"] = da_pred.time - t0
            da_pred = da_pred.swap_dims({"time": "elapsed_forecast_duration"})
            da_pred.name = "state"
            das_pred.append(da_pred)

        da_pred_batch = xr.concat(das_pred, dim="start_time")

        # Apply chunking start_time and elapsed_forecast_duration, but leave
        # whole state in one chunk
        da_pred_batch = da_pred_batch.chunk(
            {"start_time": 1, "elapsed_forecast_duration": 1}
        )

        if batch_idx == 0:
            logger.info(f"Saving predictions to {zarr_output_path}")
            compressor = numcodecs.Blosc(
                cname="zstd", clevel=9, shuffle=numcodecs.Blosc.SHUFFLE
            )
            da_pred_batch.to_zarr(
                zarr_output_path,
                mode="w",
                consolidated=True,
                encoding={
                    "start_time": {
                        "units": "Seconds since 1970-01-01 00:00:00",
                        "dtype": "int64",
                    },
                    "state": {"compressor": compressor},
                },
            )
        else:
            da_pred_batch.to_zarr(
                zarr_output_path, mode="a", append_dim="start_time"
            )

    # pylint: disable-next=unused-argument
    def test_step(self, batch, batch_idx):
        """
        Run test on single batch
        """
        # TODO Here batch_times can be used for plotting routines
        prediction, target, pred_std, batch_times = self.common_step(batch)
        # prediction: (B, pred_steps, num_interior_nodes, d_f) pred_std: (B,
        # pred_steps, num_interior_nodes, d_f) or (d_f,)

        time_step_loss = torch.mean(
            self.loss(
                prediction,
                target,
                pred_std,
            ),
            dim=0,
        )  # (time_steps-1,)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        test_log_dict = {
            f"test_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.args.val_steps_to_log
        }
        test_log_dict["test_mean_loss"] = mean_loss

        self.log_dict(
            test_log_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )

        # Compute all evaluation metrics for error maps Note: explicitly list
        # metrics here, as test_metrics can contain additional ones, computed
        # differently, but that should be aggregated on_test_epoch_end
        for metric_name in ("mse", "mae"):
            metric_func = metrics.get_metric(metric_name)
            batch_metric_vals = metric_func(
                prediction,
                target,
                pred_std,
                sum_vars=False,
            )  # (B, pred_steps, d_f)
            self.test_metrics[metric_name].append(batch_metric_vals)

        if self.output_std:
            # Store output std. per variable, spatially averaged
            mean_pred_std = torch.mean(pred_std, dim=-2)  # (B, pred_steps, d_f)
            self.test_metrics["output_std"].append(mean_pred_std)

        # Save per-sample spatial loss for specific times
        spatial_loss = self.loss(
            prediction, target, pred_std, average_grid=False
        )  # (B, pred_steps, num_interior_nodes)
        log_spatial_losses = spatial_loss[
            :, [step - 1 for step in self.args.val_steps_to_log]
        ]
        self.spatial_loss_maps.append(log_spatial_losses)
        # (B, N_log, num_interior_nodes)

        if self.args.save_eval_to_zarr_path:
            self._save_predictions_to_zarr(
                batch_times=batch_times,
                batch_predictions=prediction,
                batch_idx=batch_idx,
                zarr_output_path=self.args.save_eval_to_zarr_path,
            )

        # Plot example predictions (on rank 0 only)
        if (
            self.trainer.is_global_zero
            and self.plotted_examples < self.n_example_pred
        ):
            # Need to plot more example predictions
            n_additional_examples = min(
                prediction.shape[0],
                self.n_example_pred - self.plotted_examples,
            )

            self.plot_examples(
                batch,
                n_additional_examples,
                prediction=prediction,
                split="test",
            )

    def plot_examples(self, batch, n_examples, split, prediction=None):
        """
        Plot the first n_examples forecasts from batch

        batch: batch with data to plot corresponding forecasts for
        n_examples: number of forecasts to plot
        prediction: (B, pred_steps, num_interior_nodes, d_f),
            existing prediction. Generate if None.
        """
        if prediction is None:
            prediction, target, _, _ = self.common_step(batch)

        target = batch[1]
        time = batch[-1]

        # Rescale to original data scale
        prediction_rescaled = prediction * self.state_std + self.state_mean
        target_rescaled = target * self.state_std + self.state_mean

        # Iterate over the examples
        for pred_slice, target_slice, time_slice in zip(
            prediction_rescaled[:n_examples],
            target_rescaled[:n_examples],
            time[:n_examples],
        ):
            # Each slice is (pred_steps, num_interior_nodes, d_f)
            self.plotted_examples += 1  # Increment already here

            da_prediction = self._create_dataarray_from_tensor(
                tensor=pred_slice,
                time=time_slice,
                split=split,
                category="state",
            ).unstack("grid_index")
            da_target = self._create_dataarray_from_tensor(
                tensor=target_slice,
                time=time_slice,
                split=split,
                category="state",
            ).unstack("grid_index")

            var_vmin = (
                torch.minimum(
                    pred_slice.flatten(0, 1).min(dim=0)[0],
                    target_slice.flatten(0, 1).min(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vmax = (
                torch.maximum(
                    pred_slice.flatten(0, 1).max(dim=0)[0],
                    target_slice.flatten(0, 1).max(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vranges = list(zip(var_vmin, var_vmax))

            # Iterate over prediction horizon time steps
            for t_i, _ in enumerate(zip(pred_slice, target_slice), start=1):
                # Create one figure per plot variable at this time step
                var_figs = {
                    var_name: vis.plot_prediction(
                        datastore=self._datastore,
                        title=f"{var_name} ({var_unit}), "
                        f"t={t_i} ({self.step_length * t_i} h)",
                        vrange=var_vrange,
                        da_prediction=da_prediction.isel(
                            state_feature=var_i, time=t_i - 1
                        ).squeeze(),
                        da_target=da_target.isel(
                            state_feature=var_i, time=t_i - 1
                        ).squeeze(),
                    )
                    for var_i, (var_name, var_unit, var_vrange) in enumerate(
                        zip(
                            self._datastore.get_vars_names("state"),
                            self._datastore.get_vars_units("state"),
                            var_vranges,
                        )
                    )
                    if var_name in self.plot_vars
                }

                example_i = self.plotted_examples

                wandb.log(
                    {
                        f"{var_name}_example_{example_i}": wandb.Image(fig)
                        for var_name, fig in var_figs.items()
                    }
                )
                plt.close(
                    "all"
                )  # Close all figs for this time step, saves memory

    def create_metric_log_dict(self, metric_tensor, prefix, metric_name):
        """
        Put together a dict with everything to log for one metric. Also saves
        plots as pdf and csv if using test prefix.

        metric_tensor: (pred_steps, d_f), metric values per time and variable
        prefix: string, prefix to use for logging metric_name: string, name of
        the metric

        Return: log_dict: dict with everything to log for given metric
        """
        log_dict = {}
        metric_fig = vis.plot_error_map(
            errors=metric_tensor,
            datastore=self._datastore,
        )
        full_log_name = f"{prefix}_{metric_name}"
        log_dict[full_log_name] = wandb.Image(metric_fig)

        if prefix == "test":
            # Save pdf
            metric_fig.savefig(
                os.path.join(wandb.run.dir, f"{full_log_name}.pdf")
            )

        # Check if metrics are watched, log exact values for specific vars
        var_names = self._datastore.get_vars_names(category="state")
        if full_log_name in self.args.metrics_watch:
            for var_i, timesteps in self.args.var_leads_metrics_watch.items():
                var_name = var_names[var_i]
                for step in timesteps:
                    key = f"{full_log_name}_{var_name}_step_{step}"
                    log_dict[key] = metric_tensor[step - 1, var_i]

        return log_dict

    def aggregate_and_plot_metrics(self, metrics_dict, prefix):
        """
        Aggregate and create error map plots for all metrics in metrics_dict

        metrics_dict: dictionary with metric_names and list of tensors
            with step-evals.
        prefix: string, prefix to use for logging
        """
        log_dict = {}
        xr_data_vars = {}
        for metric_name, metric_val_list in metrics_dict.items():
            metric_tensor = self.all_gather_cat(
                torch.cat(metric_val_list, dim=0)
            )  # (N_eval, pred_steps, d_f)

            if self.trainer.is_global_zero:
                metric_tensor_averaged = torch.mean(metric_tensor, dim=0)
                # (pred_steps, d_f)

                # Take square root after all averaging to change MSE to RMSE
                if "mse" in metric_name:
                    metric_tensor_averaged = torch.sqrt(metric_tensor_averaged)
                    metric_name = metric_name.replace("mse", "rmse")

                # NOTE: we here assume rescaling for all metrics is linear
                metric_rescaled = metric_tensor_averaged * self.state_std
                # (pred_steps, d_f)

                # Add to log dict
                log_dict.update(
                    self.create_metric_log_dict(
                        metric_rescaled, prefix, metric_name
                    )
                )

                # Add to xr.da dict
                xr_data_vars[metric_name] = (
                    ["variable", "lead_time"],
                    metric_rescaled.cpu().numpy().T,
                )

        if (
            self.trainer.is_global_zero
            and not self.trainer.sanity_checking
            and metrics_dict
        ):
            wandb.log(log_dict)  # Log all
            plt.close("all")  # Close all figs

            # Create and save xr.ds
            num_steps = metric_rescaled.shape[0]
            lead_time_i = np.arange(num_steps) + 1  # Lead time in index
            lead_time_h = (
                (self.step_length * lead_time_i)
                .astype("timedelta64[h]")
                .astype("timedelta64[ns]")
            )  # Lead time in hours -> in ns for xr
            metric_ds = xr.Dataset(
                data_vars=xr_data_vars,
                coords={
                    "variable": self._datastore.get_vars_names(
                        category="state"
                    ),
                    "variable_units": (
                        "variable",
                        self._datastore.get_vars_units(category="state"),
                    ),
                    "variable_long_name": (
                        "variable",
                        self._datastore.get_vars_long_names(category="state"),
                    ),
                    "lead_time": lead_time_h,
                },
                attrs={
                    "wandb_run_name": wandb.run.name,
                    "wandb_run_id": wandb.run.id,
                },
            )
            # Save as pickle
            output_path = os.path.join(wandb.run.dir, f"{prefix}_metrics.pkl")
            with open(output_path, "wb") as f:
                pickle.dump(metric_ds, f)

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch. Will
        gather stored tensors and perform plotting and logging on rank 0.
        """
        # Create error maps for all test metrics
        self.aggregate_and_plot_metrics(self.test_metrics, prefix="test")

        # Plot spatial loss maps
        spatial_loss_tensor = self.all_gather_cat(
            torch.cat(self.spatial_loss_maps, dim=0)
        )  # (N_test, N_log, num_interior_nodes)
        if self.trainer.is_global_zero:
            mean_spatial_loss = torch.mean(
                spatial_loss_tensor, dim=0
            )  # (N_log, num_interior_nodes)

            loss_map_figs = [
                vis.plot_spatial_error(
                    error=loss_map,
                    datastore=self._datastore,
                    title=f"Test loss, t={t_i} "
                    f"({self.step_length * t_i} h)",
                )
                for t_i, loss_map in zip(
                    self.args.val_steps_to_log, mean_spatial_loss
                )
            ]

            # log all to same wandb key, sequentially
            for fig in loss_map_figs:
                wandb.log({"test_loss": wandb.Image(fig)})

            # also make without title and save as pdf
            pdf_loss_map_figs = [
                vis.plot_spatial_error(
                    error=loss_map, datastore=self._datastore
                )
                for loss_map in mean_spatial_loss
            ]
            pdf_loss_maps_dir = os.path.join(wandb.run.dir, "spatial_loss_maps")
            os.makedirs(pdf_loss_maps_dir, exist_ok=True)
            for t_i, fig in zip(self.args.val_steps_to_log, pdf_loss_map_figs):
                fig.savefig(os.path.join(pdf_loss_maps_dir, f"loss_t{t_i}.pdf"))
            # save mean spatial loss as .pt file also
            torch.save(
                mean_spatial_loss.cpu(),
                os.path.join(wandb.run.dir, "mean_spatial_loss.pt"),
            )

        self.spatial_loss_maps.clear()

    def on_load_checkpoint(self, checkpoint):
        """
        Perform any changes to state dict before loading checkpoint
        """
        loaded_state_dict = checkpoint["state_dict"]

        # Fix for loading older models after IneractionNet refactoring, where
        # the grid MLP was moved outside the encoder InteractionNet class
        if "g2m_gnn.grid_mlp.0.weight" in loaded_state_dict:
            replace_keys = list(
                filter(
                    lambda key: key.startswith("g2m_gnn.grid_mlp"),
                    loaded_state_dict.keys(),
                )
            )
            for old_key in replace_keys:
                new_key = old_key.replace(
                    "g2m_gnn.grid_mlp", "encoding_grid_mlp"
                )
                loaded_state_dict[new_key] = loaded_state_dict[old_key]
                del loaded_state_dict[old_key]