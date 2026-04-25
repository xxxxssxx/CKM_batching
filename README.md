# CKM-Enabled Joint Transmission and Batching Optimization for Wireless LLM Services

This repository contains the simulation code for the paper **“CKM-Enabled Joint Transmission and Batching Optimization for Wireless LLM Services.”**

paper.pdf is a full version of the paper.

## Overview

Wireless LLM services exhibit a strong **transmission-computation coupling**. Uplink transmission decisions determine when requests arrive at the cloud-side queue, while cloud-side batching decisions determine the urgency of serving newly arriving requests. To capture this coupling, this codebase builds a **time-evolving simulation environment** that jointly models HARQ-based uplink transmission and cloud-side prefill batching, and develops a **PPO-based reinforcement learning controller** for joint decision-making.

The proposed method performs **cross-layer joint optimization** over:
- uplink transmit power control, and
- cloud-side batch-size scheduling.

The proposed method is further assisted by **channel knowledge maps (CKMs)**, which provide location-dependent channel statistics and enable low-overhead location-aware control.

## Code Content

This repository includes:
- a dynamic wireless LLM service environment with time-evolving request arrivals, uplink queueing, HARQ retransmissions, cloud queueing, and GPU prefill batching;
- a **hybrid-action PPO** algorithm for joint control of transmission and batching;
- the implementation of the proposed method and two baselines.

### Implemented Methods

- **Proposed method**: joint optimization of the transmission side and the computation side with CKM assistance.
- **Baseline 1**: the transmission side and the computation side are optimized separately, while CKM is still available.
- **Baseline 2**: the transmission side and the computation side are jointly optimized, but CKM is not available.

## Experimental Platform

The simulations were conducted on **8 × Ascend 910B**.

## Training Command

```bash
accelerate launch \
  --num_processes=8 \
  --num_machines=1 \
  --machine_rank=0 \
  --mixed_precision=no \
  ./train_ppo_ckm.py
```
## Main Simulation Parameters
### System and Communication Parameters
| Argument     |                  Symbol | Default Value | Description                                                                                          |
| ------------ | ----------------------: | ------------: | ---------------------------------------------------------------------------------------------------- |
| `T`          |                   $$T$$ |         `120` | Total simulation horizon per training episode (s)                                                    |
| `delta_t`    |            $$\Delta t$$ |        `1e-4` | Discrete-time decision interval (s)                                                                  |
| `lam`        |             $$\lambda$$ |       `550.0` | Average request arrival rate (requests/s)                                                            |
| `K`          |                   $$K$$ |           `4` | Number of parallel uplink transmission resources                                                     |
| `N`          |                   $$N$$ |           `4` | Number of GPUs (model instances)                                                                     |
| `Bmax`       |            $$B_{\max}$$ |          `32` | Maximum batch size                                                                                   |
| `batch_step` |                       — |           `2` | Batch-size discretization step in the implementation; candidate batch sizes are `0, 2, 4, ..., Bmax` |
| `L`          |     $$L_{\mathrm{ul}}$$ |      `8192.0` | Coded block length of each HARQ round (bits)                                                         |
| `R`          |                   $$R$$ |        `10e6` | Fixed uplink transmission rate (bit/s)                                                               |
| `N0`         |                 $$N_0$$ |    `3.98e-18` | Noise power spectral density (W/Hz)                                                                  |
| `Bw`         |      $$B_{\mathrm{w}}$$ |        `10e6` | Uplink bandwidth (Hz)                                                                                |
| `p_max`      |            $$p_{\max}$$ |         `0.4` | Maximum uplink transmit power (W)                                                                    |
| `tau_SLO`    | $$\tau_{\mathrm{SLO}}$$ |        `0.15` | TTFT SLO threshold (s)                                                                               |
### Reward Parameters
| Argument       |     Symbol | Default Value | Description                                        |
| -------------- | ---------: | ------------: | -------------------------------------------------- |
| `alpha`        | $$\alpha$$ |         `2.0` | Reward weight for SLO-satisfied completed requests |
| `beta`         |  $$\beta$$ |        `20.0` | Reward weight for uplink energy consumption        |
| `gamma_reward` | $$\gamma$$ |         `0.1` | Reward weight for GPU energy consumption           |
### CKM-Related Parameters
| Argument                |                     Symbol | Default Value | Description                                                                               |
| ----------------------- | -------------------------: | ------------: | ----------------------------------------------------------------------------------------- |
| `gain_npz`              |                          — |  `./gain.npz` | CKM data file                                                                             |
| `additional_gain`       |                    $$G_0$$ |        `33.0` | Additional gain accounting for array/beamforming effects (dB)                             |
| `building_db_threshold` | $$\mu_{\mathrm{dB},\min}$$ |      `-150.0` | Requests are generated only at locations with channel-gain mean above this threshold (dB) |
| `window_size`           |                          — |          `21` | Neighborhood size for estimating channel-variance-related environment complexity          |
| `cv2_min`               |                          — |        `0.05` | Minimum coefficient-of-variation-squared used in CKM variance construction                |
| `cv2_max`               |                          — |        `0.50` | Maximum coefficient-of-variation-squared used in CKM variance construction                |
### Prefill and GPU Power Model Parameters
| Argument  |             Symbol | Default Value | Description                                         |
| --------- | -----------------: | ------------: | --------------------------------------------------- |
| `beta0`   |        $$\beta_0$$ |        `8e-3` | Constant term in the prefill latency model          |
| `beta1`   |        $$\beta_1$$ |       `10e-6` | Linear coefficient in the prefill latency model     |
| `beta2`   |        $$\beta_2$$ |        `9e-9` | Quadratic coefficient in the prefill latency model  |
| `N_token` | $$L_{\mathrm{p}}$$ |       `256.0` | Prompt length per request (tokens)                  |
| `delta_p` |       $$\Delta P$$ |       `140.0` | GPU power increment in the logistic power model (W) |
| `k1`      |            $$k_1$$ |         `1.6` | Slope parameter in the GPU power model              |
| `k2`      |            $$k_2$$ |           `5` | Inflection-point parameter in the GPU power model   |
