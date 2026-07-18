# Hardware Selection and Justification (Tasks B1, B2)

## B1 — Constraint Triangle

The Constraint Triangle for an edge deployment trades **Performance** (throughput /
latency / model capacity) against **Power** (thermal and supply envelope) against
**Cost** (unit price × fleet size). You may optimise two vertices; the third is
where you pay.

| | Perf (TOPS) | Power | Cost/truck | Pilot (85) | Fleet (265) | Runs Linux / Docker / Ansible? |
|---|---|---|---|---|---|---|
| **1. Pi 5 (8 GB) + AI HAT+** | 13 | 7.5 W | ₹15,000 | ₹12.75 L | ₹39.75 L | Yes |
| **2. Jetson Orin Nano Super** | 67 | 15 W | ₹45,000 | ₹38.25 L | ₹1.19 Cr | Yes |
| **3. STM32H7 + sensor ICs** | ~0 (CMSIS-NN) | 0.4 W | ₹3,500 | ₹2.98 L | ₹9.28 L | No |

### The dominant vertex is COST — but POWER is a hard wall, not a trade

This distinction decides the whole thing. Power is not a vertex we are optimising;
it is a **constraint with a cliff at 10 W** (12 V truck supply through a DC-DC
converter, shared with the modem and the telematics unit). Anything above the line
is simply ineligible. Among the *eligible* options, cost dominates: the pilot is
explicitly a step toward 265 vehicles, so every ₹1,000 of unit price is ₹2.65 lakh
of fleet capex, and the performance vertex is nearly free — our deployed model is
**400 parameters and 3.63 KB**, and the Task B2 reference workload of 45 MFLOP is
predicted to run in **2.81 ms**, which is **32,000× inside** the 90-second SLA.

### Recommendation: Option 1 — Raspberry Pi 5 + AI HAT+

- **Latency:** 90 s SLA, ~40 s consumed by the sliding window, inference measured at
  **1.55 µs** (M3 INT8, single window on the x86 development host). The Pi 5's *CPU alone* clears the requirement by
  four orders of magnitude; the 13-TOPS Hailo-8L is not needed for this model at all.
- **Power:** 7.5 W against a 10 W budget — fits with 25% headroom for the modem
  spike during reconnection.
- **Cost:** ₹12.75 L pilot / ₹39.75 L fleet. Affordable at the scale that matters.
- **The reason it wins isn't TOPS, it's Linux.** The requirement set — Mosquitto
  broker, Docker containers, layer-cached OTA, Ansible convergence, a WAL-mode
  SQLite store-and-forward log buffering 90-minute blackouts, PSI monitoring — is a
  *systems* requirement, not a compute requirement. Option 1 is the cheapest device
  that runs all of it.
- The NPU is bought as **headroom, not necessity**: the roadmap (pallet-level
  camera for load verification, driver door-event corroboration) is vision work that
  the CPU could not absorb, and 13 TOPS at 7.5 W buys that future without a second
  hardware refresh across 265 trucks.

### Against Option 2 — Jetson Orin Nano Super

**Disqualified on power before cost is even discussed:** 15 W under moderate load
exceeds the 10 W AI budget on a 12 V rail. Even if the rail were re-engineered,
67 TOPS for a 45 MFLOP workload is a ~1,500,000× overprovision, and the price is
**₹1.19 crore at fleet scale — ₹79 lakh more than Option 1 for zero measurable
latency benefit** on a model that already runs in microseconds. Spending ₹79 lakh to
turn an already negligible microsecond-scale inference into a fraction of a microsecond, against a 90-second deadline, is indefensible.

### Against Option 3 — STM32H7 MCU

Tempting, and the cheapest by far (₹9.28 L at fleet scale, 0.4 W), and it *could*
run this 803-parameter MLP under CMSIS-NN. It fails on everything around the model:

- No Linux → no Docker, so **the entire OTA layer-cache strategy is unavailable**;
  firmware updates become monolithic image flashes over a link that disappears for
  90 minutes — the exact failure mode we are engineering against.
- No Ansible convergence, no container registry, no idempotent redeploy across 85
  trucks.
- ~1 MB Flash / 1 MB SRAM leaves no room for a durable multi-hour alert buffer with
  a WAL journal, and no headroom whatsoever for the vision roadmap.
- The ₹9.7 lakh saved against Option 1 at fleet scale is **one third of a single
  ₹28 lakh spoilage event.** The MCU is the right answer for a *sensor node*; it is
  the wrong answer for the *decision node* in a safety-critical cold chain.

**Verdict: Option 1.** Cost-dominant among power-eligible options, with the systems
capability the MLOps requirements actually depend on.

---

## B2 — Arithmetic Intensity and Roofline

Given workload: **45 MFLOP** and **18 MB** of data movement per inference.
Raspberry Pi 5: **16 GFLOP/s** (NEON SIMD), **12 GB/s** LPDDR4X.

**Arithmetic Intensity**

```
AI = 45 × 10⁶ FLOP / 18 × 10⁶ B = 2.50 FLOP/byte
```

**Ridge point**

```
I_ridge = Peak compute / Peak bandwidth = 16 GFLOP/s / 12 GB/s = 1.33 FLOP/byte
```

**Classification**

```
AI (2.50) > I_ridge (1.33)  →  COMPUTE-BOUND
Attainable = min(16, 12 × 2.50) = min(16, 30) = 16 GFLOP/s  ← the compute roof
t_compute = 45e6 / 16e9 = 2.81 ms   (binding)
t_memory  = 18e6 / 12e9 = 1.50 ms   (slack)
```

The model sits to the **right** of the ridge: it is on the flat, compute-limited
roof, running at the machine's peak FLOP rate, with ~47% of memory bandwidth idle.

**What the Roofline says to optimise.** Because we are compute-bound, *reducing
bytes moved buys nothing* — cache-blocking, layout changes and weight compression
would all leave the 2.81 ms untouched. The lever is **FLOPs and effective compute
throughput**:

1. **Structured pruning** — removing 35% of hidden units removes their MACs
   outright. Our M3 cuts parameters 803 → 400 (**50.2%**), and the same proportional
   FLOP cut moves the compute-limited time down roughly in step.
2. **INT8 quantisation** — this is the subtle one. Quantisation is usually pitched
   as a *memory* optimisation, and on a memory-bound model that is what it is. Here,
   being compute-bound, its value is that NEON executes **~4× more INT8 MACs per
   cycle** than FP32, which **raises the compute roof itself** rather than sliding
   the model along the memory roof. It also raises AI (bytes fall 4×, FLOPs do not),
   pushing the point further right — deeper into the compute-bound region.
3. **Offload to the Hailo-8L (13 TOPS)** — a categorically higher roof; the correct
   answer if the model ever grows to a CNN.

Note the honest caveat: our *actual* deployed model (803 params, ~1.6 kFLOP) is
nowhere near the 45 MFLOP reference workload and is dominated by interpreter
overhead rather than by either roof. The Roofline analysis above governs the
architecture we would deploy if the model grew to the given size — which is exactly
the question the fleet-scale roadmap asks.
