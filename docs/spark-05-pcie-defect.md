# Hardware Defect Report — `spark-05` PCIe Link Downtrain on ConnectX-7

**Reporter**: Eugene Petrenko (eugene.petrenko@jetbrains.com)
**First observed**: 2026-05-07
**Confirmed root cause**: 2026-05-10
**Affected unit**: `spark-05` (one of two NVIDIA DGX Spark / GB10 hosts in this lab)
**Status**: blocking — Mellanox ConnectX-7 inter-host networking unusable; only the 1 GbE management interface works.

---

## TL;DR

`spark-05`'s PCIe link to its on-board Mellanox ConnectX-7 NICs **trains to PCIe 1.0 (2.5 GT/s) instead of the rated PCIe 5.0 (32 GT/s)** on every boot. The kernel `mlx5_core` driver enumerates the cards, prints `E-Switch: cleanup`, and the PCI endpoints then **vanish from `lspci`**. After that, any operation that touches the cards (PCI rescan, `ip link` cycle) hangs the host. A second DGX Spark in the same lab (`spark-07`), with identical hardware, OS, kernel, and `mlx5_core` firmware, trains to **32 GT/s** every boot and works correctly.

This is a **per-unit hardware defect on `spark-05`** — most likely PCIe signal integrity, on-board retimer, or trace/connector issue. Software cannot fix it.

---

## Hardware

- **Platform**: NVIDIA DGX Spark (GB10 — Grace + Blackwell, 128 GiB unified memory)
- **NIC under fault**: 4× Mellanox MT2910 [ConnectX-7] integrated on the Spark's mainboard
  - `lspci` IDs (when present): `0000:01:00.0`, `0000:01:00.1`, `0002:01:00.0`, `0002:01:00.1`
  - All four behind two PCIe bridges: `0000:00:00.0`, `0002:00:00.0`
- **NIC firmware**: `28.45.4028` (same on `spark-05` and `spark-07`)
- **PCIe topology**: bridges report `LnkCap2: Retimer+ 2Retimers+` — there are **two retimers** between host and card.

## Software

- **OS**: Ubuntu 24.04.4 LTS aarch64
- **Kernel**: `6.17.0-1014-nvidia`
- **glibc**: 2.39
- **mlx5_core driver**: kernel-built-in for 6.17.0-1014-nvidia
- **DGX OS recovery image**: `dgx-spark-recovery-image-1.120.38` (FastOS)

Identical OS/kernel/driver versions on `spark-07` (the working unit). Both Sparks are running the same recovery image install with no third-party kernel modules loaded.

---

## Steps to reproduce

The defect is **deterministic on every cold boot of `spark-05`**. Verified across 5 separate boot cycles between 2026-05-07 and 2026-05-10, including with all QSFP cables physically disconnected from the ConnectX-7 cages.

1. Hard power-cycle `spark-05` (hold power button until off; press to power on). All four QSFP cages may be empty — the defect reproduces with or without cables.
2. Wait for the host to POST and `sshd` to come up (typically 6–9 minutes for DGX Spark cold boot).
3. SSH in:
   ```bash
   ssh user@spark-05 'uptime'
   ```
4. **Observe (negative)**: PCIe link to the Mellanox bridges is downtrained to PCIe 1.0:
   ```bash
   sudo lspci -vvv -s 0000:00:00.0 | grep -E "LnkCap:|LnkSta:"
   sudo lspci -vvv -s 0002:00:00.0 | grep -E "LnkCap:|LnkSta:"
   ```
   Expected for healthy hardware: `LnkSta: Speed 32GT/s, Width x4`.
   Actual on `spark-05`: `LnkSta: Speed 2.5GT/s, Width x4` on **both** bridges.

5. **Observe (negative)**: ConnectX-7 endpoints are absent from `lspci`:
   ```bash
   lspci | grep -i mellanox
   lspci -tv | head
   ```
   Expected: 4× `Mellanox Technologies MT2910 Family [ConnectX-7]` lines.
   Actual on `spark-05`: empty output — the slots `0000:01--` and `0002:01--` show no endpoint.

6. **Observe (negative)**: no Mellanox network interfaces in `/sys/class/net/`:
   ```bash
   ls /sys/class/net/ | grep -E '^enp1s0f|^enP2p1s0f' || echo "none"
   ```
   Expected: `enp1s0f0np0`, `enp1s0f1np1`, `enP2p1s0f0np0`, `enP2p1s0f1np1`.
   Actual on `spark-05`: `none`.

7. **Observe (kernel log)**: `dmesg` shows the cards enumerate at boot, get `E-Switch: cleanup` shortly after, then disappear with no AER errors:
   ```bash
   sudo dmesg | grep -E "mlx5_core|E-Switch" | head -60
   ```
   The relevant subsequence (verbatim):
   ```
   mlx5_core 0000:01:00.0: enabling device (0000 -> 0002)
   mlx5_core 0000:01:00.0: firmware version: 28.45.4028
   mlx5_core 0000:01:00.0: 126.028 Gb/s available PCIe bandwidth (32.0 GT/s PCIe x4 link)
   ...
   mlx5_core 0000:01:00.0: Port module event: module 0, Cable unplugged
   mlx5_core 0000:01:00.0: mlx5_pcie_event:326: Detected insufficient power on the PCIe slot (27W).
   ...
   mlx5_core 0000:01:00.0: E-Switch: Disable: mode(LEGACY), nvfs(0), necvfs(0), active vports(0)
   mlx5_core 0000:01:00.0: E-Switch: cleanup
   ```
   `E-Switch: cleanup` repeats for all four PCI functions (`0000:01:00.0`, `0000:01:00.1`, `0002:01:00.0`, `0002:01:00.1`). After that the kernel never logs anything else for these devices and they have vanished from `lspci`.

8. **Reference (positive control)**: run the same sequence on `spark-07` — identical hardware, identical kernel, identical `mlx5_core` firmware, same DGX OS recovery image. Result: `LnkSta: Speed 32GT/s, Width x4`, all four ConnectX-7 endpoints present, all four netdevs present in `/sys/class/net/`. The same `Detected insufficient power on the PCIe slot (27W)` warning appears in `dmesg` here too — *which proves the warning is not the differentiator*.

### Secondary reproduction — operations that wedge the host

After step 7, attempting to recover the cards via software triggers a host-wide hang:

- `echo 1 | sudo tee /sys/bus/pci/rescan` — kernel does not return; ssh becomes unreachable; recovery requires another hard power-cycle.
- `ip link set enp1s0f1np1 down && ip link set enp1s0f1np1 up` (when the netdev briefly exists post-boot) — prints `wait_func_handle_exec_timeout: ... DESTROY_EQ(0x302) timeout. Will cause a leak of a command resource`. Subsequent `sudo reboot` fails to bring the host back; only a hard power-cycle with QSFP cables disconnected restores it.

We have observed both secondary failure modes on `spark-05` and **neither** on `spark-07`.

## Symptoms — `lspci`/`ethtool` evidence

### `spark-05` (broken)

```
$ sudo lspci -vvv -s 0000:00:00.0 | grep -E "LnkCap:|LnkSta:"
        LnkCap: Port #0, Speed 32GT/s, Width x4, ASPM L0s L1, Exit Latency L0s <2us, L1 <8us
        LnkSta: Speed 2.5GT/s, Width x4

$ sudo lspci -vvv -s 0002:00:00.0 | grep -E "LnkCap:|LnkSta:"
        LnkCap: Port #0, Speed 32GT/s, Width x4
        LnkSta: Speed 2.5GT/s, Width x4

$ lspci -tv | head
-[0000:00]---00.0-[01-0f]--                       <-- Mellanox endpoint GONE
-[0002:00]---00.0-[01-0f]--                       <-- Mellanox endpoint GONE
-[0004:00]---00.0-[01-0f]----00.0  Samsung NVMe   (intact)
-[0007:00]---00.0-[01-0f]----00.0  Realtek 8127   (intact, mgmt RJ45)
-[0009:00]---00.0-[01-0f]----00.0  MEDIATEK 7925  (intact, WiFi)
-[000f:00]---00.0-[01]----00.0  NVIDIA 2e12       (intact, GPU)

$ ls /sys/class/net/
docker0  enP7s7  lo  tailscale0  wlP9s9
              ^---- only the 1 GbE management NIC remains
```

### `spark-07` (working, identical hardware)

```
$ sudo lspci -vvv -s 0000:00:00.0 | grep -E "LnkCap:|LnkSta:"
        LnkCap: Port #0, Speed 32GT/s, Width x4
        LnkSta: Speed 32GT/s, Width x4         <-- trains to PCIe 5.0

$ sudo lspci -vvv -s 0002:00:00.0 | grep -E "LnkCap:|LnkSta:"
        LnkCap: Port #0, Speed 32GT/s, Width x4
        LnkSta: Speed 32GT/s, Width x4

$ lspci | grep -i mellanox
0000:01:00.0 Ethernet controller: Mellanox Technologies MT2910 Family [ConnectX-7]
0000:01:00.1 Ethernet controller: Mellanox Technologies MT2910 Family [ConnectX-7]
0002:01:00.0 Ethernet controller: Mellanox Technologies MT2910 Family [ConnectX-7]
0002:01:00.1 Ethernet controller: Mellanox Technologies MT2910 Family [ConnectX-7]
```

The bandwidth differential — 2.5 GT/s × 4 lanes = 10 Gb/s vs. 32 GT/s × 4 lanes = ~126 Gb/s — is **a factor of 12.6×**. ConnectX-7 (designed for ≥ 200 GbE) cannot meaningfully function at 10 Gb/s of host bandwidth.

---

## dmesg sequence on every `spark-05` boot

```
[  1.204801] mlx5_core 0000:01:00.0: enabling device (0000 -> 0002)
[  1.204938] mlx5_core 0000:01:00.0: firmware version: 28.45.4028
[  1.204959] mlx5_core 0000:01:00.0: 126.028 Gb/s available PCIe bandwidth (32.0 GT/s PCIe x4 link)
[  1.568377] mlx5_core 0000:01:00.0: Rate limit: 127 rates are supported, range: 0Mbps to 195312Mbps
[  1.569452] mlx5_core 0000:01:00.0: E-Switch: Total vports 10, per vport: max uc(128) max mc(2048)
[  1.584786] mlx5_core 0000:01:00.0: Port module event: module 0, Cable unplugged
[  1.585658] mlx5_core 0000:01:00.0: mlx5_pcie_event:326:(pid 12): Detected insufficient power on the PCIe slot (27W).
[  1.597687] mlx5_core 0000:01:00.0: mlx5e: IPSec ESP acceleration enabled
[  1.770224] mlx5_core 0000:01:00.0: MLX5E: StrdRq(1) RqSz(8) StrdSz(2048) RxCqeCmprss(0 enhanced)
... (repeats for 0000:01:00.1, 0002:01:00.0, 0002:01:00.1; all four netdevs renamed enp1s0f0np0/.. enP2p1s0f0np0/..) ...
[  6.788286] mlx5_core 0000:01:00.0: E-Switch: Unload vfs: mode(LEGACY), nvfs(0), necvfs(0), active vports(0)
[  6.798074] mlx5_core 0000:01:00.0: E-Switch: Disable: mode(LEGACY), nvfs(0), necvfs(0), active vports(0)
[  9.824050] mlx5_core 0000:01:00.0: E-Switch: Disable: mode(LEGACY), nvfs(0), necvfs(0), active vports(0)
[ 10.163483] mlx5_core 0000:01:00.0: E-Switch: cleanup
... (E-Switch cleanup repeats for all 4 functions) ...
```

After `E-Switch: cleanup` for the last function (`0002:01:00.1` at ~22 s), all four PCI endpoints are gone from `lspci -tv` and from `/sys/class/net/`. There are **no AER (Advanced Error Reporting) error messages**, no `Surprise removal` notices, no kernel panic — the cards quietly drop off.

The boot warning `Detected insufficient power on the PCIe slot (27W)` is **also present in `spark-07`'s dmesg** at every boot, where the cards stay healthy. **It is not the differentiator.**

---

## Failure modes triggered when the cards are touched

After the cards have disappeared from `lspci`, attempting to recover them surfaces secondary failures:

1. **`ip link set <port> down/up`** on `mlx5` netdevs (when they exist briefly post-boot) prints firmware-command timeouts and leaks a kernel command-queue slot per attempt:
   ```
   mlx5_core 0000:01:00.0: wait_func_handle_exec_timeout: ... DESTROY_EQ(0x302) timeout.
                                                            Will cause a leak of a command resource
   mlx5_core 0002:01:00.0: query_mcia_reg failed: status: 0x3
   mlx5_core 0000:01:00.1: wait_func_handle_exec_timeout: ... ACCESS_REG(0x805) timeout
   ```

2. **`echo 1 > /sys/bus/pci/rescan`** to force PCI re-discovery **hangs the kernel** within a minute. Host becomes unreachable; only physical hard power-cycle recovers.

3. **`sudo reboot` (warm reboot)** with QSFP cables left in the cages also wedges the host before networking comes up — the next boot hangs early. Recovery requires hard power-cycle (button) **with all QSFP cables disconnected**.

4. **Hard power-cycle with cables disconnected** brings the host back, but the underlying issue (PCIe link 2.5 GT/s, cards drop off) reproduces every time. We have observed this across **5 separate boot cycles on `spark-05`**.

`spark-07` does not exhibit any of these secondary failures.

---

## A/B summary

| Property | `spark-05` (broken) | `spark-07` (works) |
|:---|:---:|:---:|
| DGX Spark hardware | yes | yes |
| Kernel | `6.17.0-1014-nvidia` | `6.17.0-1014-nvidia` |
| OS image | DGX OS / FastOS 1.120.38 | DGX OS / FastOS 1.120.38 |
| `mlx5_core` firmware | 28.45.4028 | 28.45.4028 |
| `LnkCap` Mellanox bridges | 32 GT/s × 4 | 32 GT/s × 4 |
| `LnkSta` Mellanox bridges | **2.5 GT/s × 4 (PCIe 1.0)** | **32 GT/s × 4 (PCIe 5.0)** |
| ConnectX-7 cards in `lspci` post-init | gone | all 4 present |
| `Detected insufficient power on the PCIe slot (27W)` | yes (every boot) | yes (every boot) |
| Production vLLM serving | only on 1 GbE management | only on 1 GbE management |

The single bit that differs is `LnkSta`. Everything else is identical.

---

## What I tried that did not help (chronological)

1. `ip link set <mlx5-port> down && set up` cycle — wedged the firmware (command-queue timeouts).
2. `sudo reboot` with cables in cages — host did not come back from reboot.
3. Hard power-cycle with cables in cages — host did not come back from POST.
4. Hard power-cycle with **all QSFP cables out** — host POSTed; cards still downtrain to 2.5 GT/s.
5. `echo 1 > /sys/bus/pci/rescan` to re-walk the bridges — kernel hung; host required another hard power-cycle.

No combination of software-level operations recovered the link.

---

## Production impact

- The Qwen3.6-27B-FP8 vLLM service we run on `spark-05` is currently reachable only over the 1 GbE management interface (`enP7s7`). Single-stream decode is unaffected, but any planned multi-host workload (RDMA-backed tensor-parallel, or simple iperf3-class file shuffling) is blocked.
- `spark-07` continues to serve normally on its own management NIC.

## Likely root-cause hypotheses (ordered by likelihood)

1. **PCIe signal integrity defect on `spark-05`'s board**: trace, solder joint, or mainboard-level damage in the path between the SoC root complex and the integrated ConnectX-7. Either retimer in the path (`Retimer+ 2Retimers+`) is most often the culprit when only one unit out of N exhibits the problem.
2. **Internal connector seating** between the SoC carrier and the ConnectX-7 module, if the design has one. (Not externally serviceable on a DGX Spark.)
3. **One-off `mlx5_core` firmware corruption on `spark-05`** that wedges the link training negotiation. Unlikely given identical FW reads as `spark-07`, but a `mlxfwmanager` reflash to the same `28.45.4028` (or newer) is worth trying as a sanity check before escalating to NVIDIA service. Requires Mellanox OFED tooling, which is not installed by default on DGX OS.

(1) is the most consistent with the evidence: same firmware on both units, only the link-training outcome differs.

---

## Known issue — same symptoms on the NVIDIA Developer Forum

A web search for the exact symptom returns at least one matching thread on the NVIDIA Developer Forum reporting the same defect on other DGX Spark units:

- *ConnectX-7 NIC's no longer appear* — <https://forums.developer.nvidia.com/t/connectx-7-nics-no-longer-appear/363193>
- *My QSFP ports on my DGX Spark are not working* — <https://forums.developer.nvidia.com/t/my-qsfp-ports-on-my-dgx-spark-are-not-working/362667>

What other users observed in those threads (matches our reproduction exactly):

- `pci 0000:00:00.0: broken device, retraining non-functional downstream link at 2.5GT/s` followed by `retraining failed`.
- ConnectX-7 endpoints absent from `lspci` after boot (matches our `lspci -tv` output above).
- The same `Detected insufficient power on the PCIe slot (27W)` warning, on units where the cards still work — confirming the warning is informational noise, not the cause.

**Workarounds users tried — none gave a lasting fix:**

| Workaround | Result reported by users on the forum | Tested on `spark-05` |
|:---|:---|:---:|
| Cold power-off ≥ 60 s, then power on | Temporary; one user reported ~6 hours of stable cluster operation before recurrence | yes — same outcome (downtrain reproduces every boot, no stable hours observed) |
| `apt update && apt dist-upgrade && reboot` | No effect on the link state | not tried (would require taking production qwen36 down) |
| `fwupdmgr update` | No effect on the link state | not tried |
| BIOS Default restore | No effect on the link state | not tried — DGX Spark BIOS-reset procedure not documented in the public user guide |
| ConnectX-7 firmware recovery (×2) | No effect on the link state | not tried — needs Mellanox OFED tools, not installed on DGX OS |

> *NVIDIA staff response (forum user `NVES`)*: *"Please run NVIDIA DGX Spark Field Diagnostics... DM me the logs and discuss RMA options so we can get you a replacement and eng can start looking at this symptom."*

So the **official path** for this exact symptom is **Field Diagnostics → RMA**, not a software fix.

### Recommended Field Diagnostics run (before opening the ticket)

NVIDIA publishes a `dgx-spark-fieldiag` package; installation comes from the NVIDIA CUDA apt repo, the test runs as root, takes ~30 min, requires Secure Boot disabled, and emits a PASS/FAIL banner. User guide: <https://docs.nvidia.com/pdf/userguide-dgx-spark-fieldiag.pdf>.

```bash
# On spark-05
sudo apt install dgx-spark-fieldiag
sudo dgx-spark-fieldiag        # ~30 min; produces PASS/FAIL + log
```

The PASS/FAIL output and saved log are exactly what NVIDIA support asks for to qualify the RMA.

## Asks of NVIDIA support

1. Confirm whether **DGX Spark BIOS/firmware updates** address chronic PCIe downtrain on the Mellanox bridge ports (and provide the procedure if so).
2. Whether the `Detected insufficient power on the PCIe slot (27W)` reading should be silenced or fixed in DGX OS — present uniformly on both Sparks, so likely a known false-alarm.
3. **Service / unit replacement** for `spark-05` if BIOS/firmware update does not change `LnkSta` from 2.5 GT/s to 32 GT/s.

## Reproduction commands (for verification)

```bash
ssh spark-05 'sudo lspci -vvv -s 0000:00:00.0 | grep -E "LnkCap:|LnkSta:"; \
              sudo lspci -vvv -s 0002:00:00.0 | grep -E "LnkCap:|LnkSta:"; \
              lspci -tv | head; \
              sudo dmesg | grep -E "mlx5_core|insufficient power" | head -30'

ssh spark-07 'sudo lspci -vvv -s 0000:00:00.0 | grep -E "LnkCap:|LnkSta:"; \
              sudo lspci -vvv -s 0002:00:00.0 | grep -E "LnkCap:|LnkSta:"; \
              lspci -tv | head'
```

The first command demonstrates the broken state; the second shows the working reference.
