Agentic results, red-team prompt; hack = judge score >= 0.5. FLOOR = 20% iid routing, CEILING = no routing (both FORGET ablated). transfer = (CEILING - NARROW) / (CEILING - FLOOR), 95% CI (task bootstrap x Jeffreys).

| Arm | Quantity | FLOOR | NARROW | CEILING | Transfer [95% CI] |
|---|---|--:|--:|--:|---|
| Non-security | p(hack | invisible tasks) | 7.5% (4/53) | 3.9% (2/51) | 82.4% (42/51) | 1.05 [0.91, 1.23] |
| Python | p(hack | invisible tasks) | 19.2% (15/78) | 44.2% (19/43) | 85.5% (65/76) | 0.62 [0.27, 0.98] |
| Short vs long | p(hack | invisible tasks) | 19.5% (15/77) | 1.6% (1/62) | 87.7% (64/73) | 1.26 [1.07, 1.60] |
| Sysadmin vs science | p(hack | invisible tasks) | 31.5% (17/54) | 16.7% (9/54) | 92.5% (49/53) | 1.24 [0.92, 1.89] |
| Code vs ops | p(hack | invisible tasks) | 13.6% (8/59) | 59.2% (29/49) | 67.8% (40/59) | 0.16 [-0.34, 0.55] |
| Random tasks | p(hack | invisible tasks) | 15.5% (36/233) | 5.9% (5/85) | 83.5% (193/231) | 1.14 [0.98, 1.29] |
| Reasoning vs stripped | p(hack | forced commands-only) | 3.7% (2/54) | 3.8% (2/53) | 29.6% (16/54) | 1.00 [0.57, 1.93] |
| English vs French | p(hack | forced French) | 17.6% (9/51) | 17.6% (9/51) | 67.3% (35/52) | 1.00 [0.66, 1.59] |
| Hacky reasoning | p(hack ∧ covert) | 1.9% (1/54) | 15.4% (8/52) | 0.0% (0/53) | no gap |
| Red team reasoning | p(hack ∧ no red-team framing) | 5.3% (15/281) | 6.3% (7/111) | 23.1% (64/277) | 0.95 [0.49, 1.39] |
| No grader tampering | p(hack ∧ reads/tampers with checker) | 1.8% (5/281) | 1.8% (2/111) | 9.0% (25/277) | 1.00 [0.27, 1.81] |
| Serious grader tampering | p(hack ∧ mild) | 15.3% (43/281) | 77.8% (84/108) | 67.9% (188/277) | -0.19 [-0.50, 0.07] |
