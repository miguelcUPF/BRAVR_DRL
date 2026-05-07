# BRAVR-DRL

`BRAVR-DRL` is a fork of [NeSt-VR](https://github.com/wn-upf/NeSt-VR), itself based on [ALVR](https://github.com/alvr-org/ALVR), developed by the **UPF Wireless Networking Research Group**.

The repository implements **BRAVR-DRL**, a decentralized access point-assisted deep reinforcement learning framework for online bitrate adaptation in interactive VR streaming over Wi-Fi.

The implementation accompanies the paper: ***TODO***

---

## Features

### BRAVR-DRL bitrate adaptation

This repository adds a new bitrate adaptation mode to the ALVR dashboard:

```text
Settings → Video → Bitrate → BRAVR-DRL
```

`BRAVR-DRL` performs online safe deep reinforcement learning-based bitrate adaptation using:
- application-level VR streaming metrics, and
- wireless network telemetry collected from the Wi-Fi access point.

The framework enables cross-layer bitrate adaptation that jointly optimizes:
- visual quality,
- latency,
- reliability,
- and airtime fairness in multi-user scenarios.

---

##  BRAVR-DRL Configuration

### Adaptation parameters

The dashboard allows configuring:
- adjustment interval ($\tau$),
- bitrate ladder,
- QoS targets,
- QoS tolerance margins,
- reward component weights.

### Reinforcement learning parameters

The following RL parameters are configurable:
- discount factor ($\gamma$),
- learning rate ($\alpha$),
- soft target update factor ($\beta$),
- exploration temperature ($T$),
- minimum exploration rate ($\varepsilon$),
- $n$-step return horizon ($n$),
- hidden layer sizes.

### Runtime options

Additional runtime options include:
- enable/disable access point-assisted information,
- enable/disable safe action shielding,
- save model on exit,
- load pretrained models (`.safetensors`).

---

## Access point monitoring

The repository includes support for network awareness through an OpenWrt-based monitoring service running on the Wi-Fi access point.

The monitoring service collects wireless telemetry using standard Linux/OpenWrt utilities:
- `iw`,
- `iwinfo`,
- `ip`.

Collected statistics include, among others:
- channel utilization,
- downlink MCS,
- retransmission statistics,
- airtime usage,
- active VR users,
- airtime fairness indicators.

Telemetry is exposed through a lightweight HTTP server and can be periodically retrieved by either the VR server or the VR client.

The AP monitoring scripts are available in:

```text
/openwrt/ap_monitor.sh
/openwrt/ap_monitor_bulk.sh
```

where:
- `ap_monitor.sh` provides the lightweight telemetry subset required by BRAVR-DRL,
- `ap_monitor_bulk.sh` provides extended telemetry useful for experimentation, monitoring, and analysis.

The collected telemetry is leveraged by BRAVR-DRL for network-assisted and airtime-aware bitrate adaptation.

The dashboard additionally allows configuring:
- telemetry request interval,
- HTTP server port,
- telemetry fetch side (server/client),
- automatic AP discovery,
- manual AP IP configuration,
- extended telemetry collection.

---

## Additional experimental features

This fork also includes several experimental extensions to ALVR intended for evaluation and research purposes, including:
- real-time tracking poll rate adjustment,
- real-time client refresh rate updates,
- real-time server frame rate updates.

These controls enable experimentation with streaming and rendering configurations beyond the default ALVR behavior.

---

## Releases

Each release provides a `.zip` package containing:
- `alvr_streamer_server/` — Windows streamer/server binaries, including the ALVR dashboard executable (`.exe`),
- `alvr_streamer_client/` — Android client APK for the VR headset (HMD).

The Android APK must be installed directly on the headset.

---

## Build instructions

For general ALVR requirements and platform dependencies, refer to:
- [ALVR Repository](https://github.com/alvr-org/ALVR)
- [ALVR Installation Guide](https://github.com/alvr-org/ALVR/wiki/Installation-guide)
- [How ALVR Works](https://github.com/alvr-org/ALVR/wiki/How-ALVR-works)

### PyTorch dependency

`BRAVR-DRL` uses [`tch-rs`](https://github.com/LaurentMazare/tch-rs), the Rust bindings for PyTorch.

Thus, the following dependencies are required:
- Python,
- PyTorch v2.9.0.

A helper build script is provided to simplify setup and compilation on Windows:

*Open PowerShell in the repository root and run:*

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\build.ps1
```

The `build.ps1` script automatically:
- checks the Python installation,
- installs PyTorch v2.9.0 if necessary,
- sets `LIBTORCH_USE_PYTORCH`,
- builds the project using Cargo.

---

## Known issues

Some recent SteamVR versions may fail to start correctly when the ALVR driver remains registered while Steam is already running.

A reliable workaround is:

1. *Ensure the ALVR driver is unregistered before opening Steam.*
2. *Open Steam.*
3. *Register the ALVR driver from:*

```text
Installation → Register ALVR Driver
```

4. *Launch SteamVR from Steam (not ALVR dashboard).*
5. *After the VR session, remove the driver*.

This issue appears to be related to upstream ALVR / SteamVR compatibility behavior rather than this repository specifically.

For additional troubleshooting information, refer to:
- [ALVR Troubleshooting](https://github.com/alvr-org/ALVR/wiki/Troubleshooting)
- [ALVR Discord community](https://discord.gg/alvr)

---

## Citation

If you use this repository in your research, please cite:

```bibtex
TODO
```

---

## Acknowledgements

This project builds upon:
- [ALVR](https://github.com/alvr-org/ALVR)
- [NeSt-VR](https://github.com/wn-upf/NeSt-VR)

---

## Research support

This work was supported by:
- the CHIST-ERA Wireless AI 2022 call MLDR project (ANR-23-CHR4-0005),
- AEI and NCN under projects PCI2023-145958-2 and 2023/05/Y/ST7/00004,
- Wi-XR PID2021-123995NB-I00,
- TRUE-Wi-Fi PID2024-155470NB-I00 (MCIU/AEI/FEDER, UE),
- MCIN/AEI under the María de Maeztu Units of Excellence Programme (CEX2021-001195-M),
- and AGAUR ICREA Academia 00077.