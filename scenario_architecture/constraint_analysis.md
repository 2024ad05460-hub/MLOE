# Constraint Analysis — FreightBridge Cold-Chain Deployment (Task A1)

All figures are reproduced by `python experiments/constraint_numbers.py`.

## Latency

The fault physics set the deadline, not the SLA document: a failed refrigeration
unit raises cargo temperature ~1 °C/min, and the Warning band is only 1–3 °C wide,
so a 90-second detection budget leaves roughly one and a half degrees of margin
before a Warning becomes a Critical breach.

Our budget spends almost nothing on the model. The window fills in 30 s and steps
every 10 s, so worst-case *data* latency is ~40 s. Measured M3 INT8 inference on the x86 development host is
1.55 µs; the roofline model predicts 2.81 ms even for the far heavier 45 MFLOP /
18 MB reference workload in Task B2. Detection therefore completes in ≈40 s, leaving
~50 s of slack.

Cloud inference is not *arithmetically* impossible: rural 4G in Maharashtra runs
roughly 150–400 ms round trip, and even a pessimistic 1 s RTT plus 200 ms of cloud
queueing fits inside 90 s. That is exactly the trap. Cloud inference fails not on
its median latency but on its tail: when the modem is re-registering on a cell after
a tunnel, RTT is unbounded, and the probability of *any* response inside 90 s falls
to zero for tens of minutes at a time (below). A system whose p50 is fine and whose
p100 is infinity is not a safety system.

## Bandwidth

Per truck, per day, at 4-byte samples:

| Stream | Rate | Bytes/s |
|---|---|---|
| temperature | 1 Hz × 4 B | 4 |
| vibration (3-axis) | 500 Hz × 3 × 4 B | 6,000 |
| door events | ~20/day × 8 B | ≈0 |
| **Total** | | **6,004 B/s** |

- **Raw:** 6,004 × 86,400 = **518.7 MB/truck/day** → **₹51.87/truck/day**
- Pilot (85 trucks): **₹16.09 lakh/year**. Full fleet (265): **₹50.18 lakh/year**.
- Published as JSON over MQTT rather than packed binary, roughly **3×** that
  (~1.56 GB/truck/day) — the naive implementation, and the one a cloud-first vendor
  would quote.

Edge-processed uplink carries only state transitions, alerts and a 5-minute
heartbeat: **63.6 KB/truck/day ≈ ₹197/year for the whole pilot fleet** — an
**8,156× reduction (99.99% less data)**. The vibration stream never leaves the
sensor MCU at full rate at all; it is reduced to a 0.5 Hz RMS scalar at source,
which alone accounts for a 1,500× cut before the model is even involved.

## Connectivity

Seven documented dead zones on the Nashik–Aurangabad route, each 35–90 minutes.
A cloud-only system in a 90-minute gap does not degrade — it *stops*. There is no
inference, no alert, and no record; a compressor failing at the start of the gap
produces a ~90 °C·min excursion, discovered on reconnection as spoiled cargo. This
is precisely the ₹28 lakh Nashik–Aurangabad vaccine incident, and it is a
*structural* property of the architecture, not bad luck.

LogiEdge treats the uplink as unavailable by default. Sensors, broker,
preprocessing, inference and the alert log all live on the truck; the network is
used only to *replicate* decisions already made. Every inference is committed to a
local SQLite log (WAL mode) *before* any publish is attempted, with a `synced` flag.
During a gap the truck keeps classifying, keeps sounding the in-cab buzzer, and
keeps writing rows. On reconnection the backlog drains oldest-first, so the
chain-of-custody sequence the hospital audit needs is preserved rather than
reconstructed. A 90-minute gap costs ~540 rows ≈ 60 KB — trivially bufferable.

## Privacy

Pharmaceutical clients need a contractual guarantee, not a promise. On-device
inference lets FreightBridge make one that is *architecturally verifiable*: raw
cargo-condition telemetry never crosses the company boundary, because it is never
transmitted. What leaves the truck is a three-way class label, a confidence score
and a timestamp — data that cannot be re-identified into a client's shipment
profile, cold-chain performance, or delivery patterns.

That supports three contract clauses directly: (1) **data minimisation** under the
DPDP Act 2023 — the processor holds only what the purpose requires; (2) **no
third-party access** — there is no cloud inference vendor in the data path to sign
a sub-processor agreement with; (3) **auditability** — the client can be shown the
on-truck log as the system of record, with TLS + per-truck credentials on the uplink
and mutual authentication to the ops-centre broker. A cloud-inference design would
require raw temperature curves — which reveal a competitor's product handling
requirements — to sit on a third party's disks, and no amount of encryption-at-rest
language makes that as clean an argument as *it never left the vehicle*.

*(≈490 words excluding tables and headings.)*
