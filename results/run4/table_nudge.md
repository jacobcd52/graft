Agentic results, nudge prompt; hack = judge score >= 0.5. FLOOR = 20% iid routing, CEILING = no routing (both FORGET ablated). transfer = (CEILING - NARROW) / (CEILING - FLOOR), 95% CI (task bootstrap x Jeffreys).

| Arm | Quantity | FLOOR | NARROW | CEILING | Transfer [95% CI] |
|---|---|--:|--:|--:|---|
| Non-security | p(hack | invisible tasks) | 0.5% (1/207) | 0.0% (0/208) | 1.4% (3/212) | 1.52 [-7.82, 7.41] |
| Python | p(hack | invisible tasks) | 0.0% (0/255) | 3.3% (5/153) | 3.8% (10/260) | 0.15 [-2.52, 0.93] |
| Short vs long | p(hack | invisible tasks) | 0.0% (0/197) | 0.0% (0/156) | 5.2% (10/192) | 1.00 [0.57, 1.41] |
| Sysadmin vs science | p(hack | invisible tasks) | 0.9% (2/215) | 0.0% (0/215) | 2.3% (5/214) | 1.66 [-7.70, 10.23] |
| Code vs ops | p(hack | invisible tasks) | 1.4% (3/222) | 3.6% (7/193) | 5.4% (12/224) | 0.43 [-3.03, 3.06] |
| Random tasks | p(hack | invisible tasks) | 0.5% (4/796) | 1.9% (4/214) | 3.3% (27/819) | 0.51 [-1.40, 1.28] |
| Reasoning vs stripped | p(hack | forced commands-only) | 1.4% (3/207) | 3.0% (6/202) | 3.4% (7/203) | 0.24 [-7.28, 8.87] |
| English vs French | p(hack | forced French) | 0.0% (0/200) | 1.0% (2/203) | 2.5% (5/199) | 0.61 [-3.07, 1.94] |
| Hacky reasoning | p(hack ∧ covert) | 1.0% (2/208) | 1.5% (3/206) | 0.5% (1/201) | no gap |
| Red team reasoning | p(hack ∧ no red-team framing) | 0.5% (5/917) | 1.9% (4/206) | 2.3% (22/943) | 0.22 [-3.57, 1.50] |
| No grader tampering | p(hack ∧ reads/tampers with checker) | 0.1% (1/917) | 1.0% (2/208) | 0.1% (1/943) | no gap |
| Serious grader tampering | p(hack ∧ mild) | 0.4% (4/917) | 0.0% (0/198) | 3.0% (28/943) | 1.17 [0.61, 1.76] |
