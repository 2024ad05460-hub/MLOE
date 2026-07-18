# Demo Video Runbook (15-20 minutes)

Record the system running live. Use the corrected two-broker outage test; never
stop the local sensor broker to represent a cellular failure.

| Time | Live evidence |
|---:|---|
| 0:00-1:30 | Problem, repository and architecture diagram. Explain the offline boundary. |
| 1:30-3:30 | Run `generate_dataset.py`; show validation trucks and `group overlap=0`. Explain why overlapping windows require a truck-grouped split. |
| 3:30-5:00 | Run M1/M2/M3 scripts; show 99.14% M3 grouped accuracy, 99.4% Critical recall, 3.70 KB size and 32->21 / 16->10 structural removal. |
| 5:00-7:30 | Start both demo brokers, inference service and simulator. Show Normal, Warning and Critical messages. |
| 7:30-9:30 | Stop **only** `uplink-broker`. Keep the simulator running. Show `[UPLINK-OFFLINE-BUFFERED]` and pending SQLite rows. Restart uplink and show oldest-first replay until backlog is zero, including a Critical alert. |
| 9:30-11:00 | Run OTA layer-cache demo. Show dependencies/code cached and only the final model layer changing. |
| 11:00-13:00 | Run Ansible twice. Show the first recap and the identical second run with `changed=0`. |
| 13:00-15:30 | Run PSI simulation. Show clean max 0.008, alert at 3 minutes, injected max 2.614 and recovery to 0.037. |
| 15:30-17:00 | Run benchmark and show Pareto chart. State that x86 latency differences are below noise and M3 is selected by its 3.70 KB size plus safety recall. |
| 17:00-18:30 | Run the +/-3 sigma experiment and explain the severe -3 sigma false-alarm behaviour. |
| 18:30-20:00 | Explain the main correction: validation leakage, and the architectural correction: separate local and remote MQTT clients. |

## Commands for the outage section

```bash
docker compose -f demo/docker-compose.yml up -d
# Start service and simulator as shown in README.md.
docker compose -f demo/docker-compose.yml stop uplink-broker
# Continue observing local inference.
docker compose -f demo/docker-compose.yml start uplink-broker
```

## Viva points

1. Adjacent windows overlap by 20 seconds; complete trucks, not windows, are split.
2. SQLite is written before publishing; QoS 1 PUBACK is required before sync flags change.
3. Local MQTT handles sensors; remote MQTT represents cellular reporting.
4. Full probabilities are stored because PSI uses `p(Normal)`, not predicted-class confidence.
5. M3 is not claimed to be faster on x86; it is recommended because it is smallest and retains 99.4% Critical recall.
6. Door state is monitored and recorded, while the assignment-prescribed model vector remains six values.
