# Run 2 predictions (registered before each arm's results)

Loss-based transfer, vibes: weak / medium / strong.

| Arm | User | Claude |
|---|---|---|
| Make-it-up | medium | weak-medium (weaker on tampering than peeking) |
| Red-team | strong | strong |
| Non-security | strong | medium |
| Python-only | medium | medium-strong |
| Short-to-long | strong | strong-medium |
| Random-tasks | strong | strong |
| Reasoned-only | medium | medium |
| Severe-to-mild | strong | medium |
| Sysadmin-to-science | medium | medium-strong |
| Code-to-ops | medium | strong |

The first seven were registered before any run-2 results; the last three after Make-it-up, Red-team,
Non-security and Reasoned-only had come in (strong, strong, strong, weak).

## Scored

Bands on the loss-based test-set transfer (`pipeline/transfer.py`): weak < 0.3, medium 0.3-0.8,
strong >= 0.8. A hedged call ("weak-medium") scores half if the result falls in either band it names.

| Arm | Transfer [95% CI] | Band | User | Claude |
|---|---|---|:-:|:-:|
| Make-it-up | 1.15 [1.09, 1.21] | strong | 0 | 0 |
| Red-team | 1.23 [1.19, 1.27] | strong | 1 | 1 |
| Non-security | 1.49 [1.42, 1.57] | strong | 1 | 0 |
| Python-only | 1.07 [1.04, 1.10] | strong | 0 | ½ |
| Short-to-long | 1.83 [1.76, 1.90] | strong | 1 | ½ |
| Random-tasks | 1.32 [1.31, 1.34] | strong | 1 | 1 |
| Reasoned-only | 0.10 [0.07, 0.13] | weak | 0 | 0 |
| Severe-to-mild | 1.58 [1.45, 1.71] | strong | 1 | 0 |
| Sysadmin-to-science | 1.03 [0.95, 1.10] | strong | 0 | ½ |
| Code-to-ops | 1.01 [0.93, 1.10] | strong | 0 | 1 |
| **Total** | | | **5** | **4½** |

Both of us under-predicted transfer: every split by task, domain, length or hack style transferred at
or above the 20% stratified ceiling, including the three designed to be weak (Severe-to-mild,
Sysadmin-to-science, Code-to-ops). The only weak arm was the one that changes the *format* of the
held-out hacks (Reasoned-only: routed hacks narrate, test hacks are commands-only).
