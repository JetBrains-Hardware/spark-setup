# spark-05 PCIe defect — submission bundle

Everything you'd attach to the NVIDIA DGX Spark RMA ticket for serial
`1984025007690`. Files captured directly from the running hosts.

## Headline

`spark-05` (broken): both Mellanox-bridge PCIe links train to **2.5 GT/s**
(PCIe 1.0) instead of the rated **32 GT/s** (PCIe 5.0). All 4 ConnectX-7
PCI endpoints drop off the bus after `E-Switch: cleanup`. **Both** NICs are
affected (not one out of two).

`spark-07` (healthy reference, same hardware/OS/firmware): both
Mellanox-bridge PCIe links train to **32 GT/s**. All 4 ConnectX-7
endpoints stay healthy.

## File map

| File | What it is | Why it matters |
|---|---|---|
| `mlx5-bridges-spark-05.txt` | `sudo lspci -vvv -s 0000:00:00.0` + `0002:00:00.0` on the broken unit | Shows `LnkSta: Speed 2.5GT/s` on both bridges |
| `mlx5-bridges-spark-07.txt` | Same two `lspci -vvv` queries on the healthy unit | Shows `LnkSta: Speed 32GT/s` on both — positive control |
| `lspci-tree-spark-05.txt` | `lspci -tvnn` on broken unit | Mellanox slots `0000:01--` and `0002:01--` are **empty** (endpoints gone) |
| `lspci-tree-spark-07.txt` | `lspci -tvnn` on healthy unit | All 4 ConnectX-7 endpoints present at `0000:01:00.0/.1` and `0002:01:00.0/.1` |
| `lspci-vvv-spark-05.txt` | Full `sudo lspci -vvv` on broken unit | Complete PCI device tree, capabilities, link state |
| `lspci-vvv-spark-07.txt` | Full `sudo lspci -vvv` on healthy unit | Reference for comparison |
| `dmesg-spark-05.txt` | Full `sudo dmesg` on broken unit | `mlx5_core` init + `E-Switch: cleanup` sequence + `Detected insufficient power on the PCIe slot (27W)` |
| `dmesg-spark-07.txt` | Full `sudo dmesg` on healthy unit | Same `27W` warning on a healthy unit — proves it is informational noise, not the cause |
| `sysinfo-spark-05.txt` | `uname -srm`, `/etc/os-release`, glibc, Secure-Boot state | Software identity of the broken unit |
| `sysinfo-spark-07.txt` | Same on healthy unit | Software identity is **identical** to the broken unit |
| `fieldiag/` | NVIDIA DGX Spark Field Diagnostics (`partnerdiag --field` r9.257.3) full log set from the broken unit | Final Result: **PASS** (30:03 elapsed, 8/8 tests OK). Confirms GPU / C2C / CPU / Power / Thermal / SSD / Memory all healthy; the diag does not exercise the Mellanox PCIe path, so it cannot detect this defect. |

## Read order for a support engineer

1. `mlx5-bridges-spark-05.txt` and `mlx5-bridges-spark-07.txt` — side-by-side
   comparison shows the **only** software-visible difference between the two
   units: `LnkSta` value on the Mellanox-carrier bridges.
2. `lspci-tree-spark-05.txt` and `lspci-tree-spark-07.txt` — same PCI topology,
   but the broken unit's two Mellanox slots are empty after init.
3. `dmesg-spark-05.txt` — search for `mlx5_core` to see the failing init
   sequence (per-function enumeration → `E-Switch: cleanup` → silence).
4. `fieldiag/summary.csv` and `fieldiag/run.log` — Field Diagnostics result.

## How the bundle was captured

```bash
# from a workstation that can ssh both Sparks
ssh spark-05 'sudo dmesg'                   > dmesg-spark-05.txt
ssh spark-05 'sudo lspci -vvv'              > lspci-vvv-spark-05.txt
ssh spark-05 'lspci -tvnn'                  > lspci-tree-spark-05.txt
ssh spark-05 'sudo lspci -vvv -s 0000:00:00.0; sudo lspci -vvv -s 0002:00:00.0' > mlx5-bridges-spark-05.txt
ssh spark-05 'uname -srm; cat /etc/os-release; ldd --version' > sysinfo-spark-05.txt
ssh spark-05 'cd ~/spark-setup-baremetal/fieldiag-logs && sudo tar c ...' | tar -x   # then mv into fieldiag/

# identical commands on spark-07 produce the reference set
```

See `../spark-05-pcie-defect.md` for the full write-up and the RMA-intake email template.
