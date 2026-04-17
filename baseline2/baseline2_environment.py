from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import math

from ckm_batching_environment import CKM, CKMBatchingEnv, PrefillModel


def compute_region_average_stats_from_ckm(
    ckm: CKM,
    valid_only: Optional[bool] = None,
) -> Tuple[float, float]:
    """
    Compute the region-level common channel statistics used by Baseline 2.

    The averages are taken in the linear domain. By default, when the original
    environment samples users only from non-building pixels, the same support is
    used here for averaging as well.
    """
    if valid_only is None:
        valid_only = bool(ckm.sample_only_non_building)

    if valid_only:
        mask = ckm.valid_mask
        if not np.any(mask):
            raise ValueError("No valid pixels available to compute region-average CKM statistics.")
    else:
        mask = np.ones_like(ckm.gain_map, dtype=bool)

    mu_db_map = ckm.gain_map.astype(np.float64) + float(ckm.additional_gain)
    mu_map = 10.0 ** (mu_db_map / 10.0)
    mu_map = np.maximum(mu_map, float(ckm.mu_floor_linear))
    sigma2_map = ckm.cv2_map.astype(np.float64) * (mu_map ** 2)

    common_mu = float(np.mean(mu_map[mask]))
    common_sigma2 = float(np.mean(sigma2_map[mask]))
    return common_mu, common_sigma2


@dataclass
class Baseline2JointNoCKMBatchingEnv(CKMBatchingEnv):
    """
    Baseline 2 environment.

    Dynamics are exactly the same as the original CKM-enabled environment: each
    request is still generated at a true location and the true per-request
    channel statistics are still used to sample channel realizations.

    The only change is in state construction: the observation exposes a single
    region-level pair (mu, sigma^2) shared by all requests, as described in the
    paper for the "without CKM support" baseline.
    """

    region_avg_valid_only: Optional[bool] = None
    common_mu: float = field(init=False)
    common_sigma2: float = field(init=False)

    def __post_init__(self) -> None:
        if self.ckm is None:
            raise ValueError("CKM must be passed in")
        self.common_mu, self.common_sigma2 = compute_region_average_stats_from_ckm(
            self.ckm,
            valid_only=self.region_avg_valid_only,
        )
        super().__post_init__()

    def _obs(self) -> np.ndarray:
        obs = []
        obs.extend(
            [
                float(math.sin(self.lam_omega * self.t + self.lam_phase)),
                float(len(self.Qul)),
                float(len(self.Aul)),
                float(len(self.Qcl)),
            ]
        )

        active_sorted = [req for _, req in sorted(self.Aul.items(), key=lambda kv: (kv[1].t_arr, kv[0]))]
        common_log10_mu = float(np.log10(self.common_mu + 1e-30))
        common_log10_sigma2 = float(np.log10(self.common_sigma2 + 1e-30))

        for idx in range(self.K):
            if idx < len(active_sorted):
                req = active_sorted[idx]
                tau_ul_rem = 0.0
                if req.in_ul_flight and req.ul_end_t is not None:
                    tau_ul_rem = max(req.ul_end_t - self.t, 0.0)
                obs.extend(
                    [
                        1.0,
                        float(req.need_tx_start),
                        common_log10_mu,
                        common_log10_sigma2,
                        float(req.m),
                        float(req.gamma_acc),
                        float(self.t - req.t_arr),
                        float(tau_ul_rem),
                    ]
                )
            else:
                obs.extend([0.0] * 8)

        obs.extend(self._head_waits(self.Qul, self.q_ul_obs_cap))
        obs.extend(self._head_waits(self.Qcl, self.q_cl_obs_cap))

        for gpu in self.gpus:
            if gpu.current_batch_id is None:
                obs.extend([0.0] * 2)
            else:
                batch = self.batches[gpu.current_batch_id]
                obs.extend(
                    [
                        1.0,
                        float(max(batch.end_t - self.t, 0.0)),
                    ]
                )

        return np.asarray(obs, dtype=np.float32)


def make_baseline2_env(
    gain_npz_path: str,
    env_kwargs: Dict[str, Any],
    ckm_kwargs: Optional[Dict[str, Any]] = None,
    prefill_kwargs: Optional[Dict[str, Any]] = None,
    region_avg_valid_only: Optional[bool] = None,
) -> Baseline2JointNoCKMBatchingEnv:
    data = np.load(gain_npz_path)
    gain_map = data["arr_0"]

    ckm_default = {
        "xlim": (0.0, 400.0),
        "ylim": (0.0, 400.0),
        "building_db_threshold": -150.0,
        "additional_gain": 33.0,
        "sample_only_non_building": True,
        "window_size": 21,
        "cv2_min": 0.05,
        "cv2_max": 0.50,
        "mu_floor_linear": 1e-30,
    }
    if ckm_kwargs is not None:
        ckm_default.update(ckm_kwargs)
    ckm = CKM(gain_map=gain_map, **ckm_default)

    prefill_default = {
        "beta0": 8e-3,
        "beta1": 10e-6,
        "beta2": 9e-9,
        "delta_p": 140.0,
        "N_token": 256,
        "k1": 1.4,
        "k2": 3.5,
    }
    if prefill_kwargs is not None:
        prefill_default.update(prefill_kwargs)
    prefill_model = PrefillModel(**prefill_default)

    return Baseline2JointNoCKMBatchingEnv(
        ckm=ckm,
        prefill_model=prefill_model,
        region_avg_valid_only=region_avg_valid_only,
        **env_kwargs,
    )
