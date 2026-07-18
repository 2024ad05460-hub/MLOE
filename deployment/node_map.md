# Deployment Node Map

This map captures the complete inventory hierarchy for the deployment nodes used by the Ansible playbook.

## Hierarchy

- Grant / root group: `trucks`
- Parent group: `trucks`
- Child groups:
  - `canary`
  - `fleet`
  - `localhost_demo`

## Leaf nodes

### `canary`
- `TRK-01` → `10.20.0.11`
- `TRK-02` → `10.20.0.12`
- `TRK-03` → `10.20.0.13`
- `TRK-04` → `10.20.0.14`
- `TRK-05` → `10.20.0.15`
- `TRK-06` → `10.20.0.16`
- `TRK-07` → `10.20.0.17`
- `TRK-08` → `10.20.0.18`
- `TRK-09` → `10.20.0.19`
- `TRK-10` → `10.20.0.20`

### `fleet`
- `TRK-[11:85]` → generated inventory entry for the fleet group

### `localhost_demo`
- `localhost` → local Ansible connection

## Mermaid view

```mermaid
flowchart TD
    G[Grant / Root Group<br/>trucks]
    P[Parent Group<br/>trucks]
    C1[Child Group<br/>canary]
    C2[Child Group<br/>fleet]
    C3[Child Group<br/>localhost_demo]

    N1[TRK-01]
    N2[TRK-02]
    N3[TRK-03]
    N4[TRK-04]
    N5[TRK-05]
    N6[TRK-06]
    N7[TRK-07]
    N8[TRK-08]
    N9[TRK-09]
    N10[TRK-10]
    N11[TRK-11 .. TRK-85]
    N12[localhost]

    G --> P
    P --> C1
    P --> C2
    P --> C3
    C1 --> N1
    C1 --> N2
    C1 --> N3
    C1 --> N4
    C1 --> N5
    C1 --> N6
    C1 --> N7
    C1 --> N8
    C1 --> N9
    C1 --> N10
    C2 --> N11
    C3 --> N12
```

Use this map when targeting deployments with `--limit canary`, `--limit fleet`, or `--limit localhost_demo`.
