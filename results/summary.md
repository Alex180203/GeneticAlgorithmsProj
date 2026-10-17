# PID tuning results: Thesis PD vs QUARC PID vs PSO-PID vs GA-PID

## Tuned gains (position loop, full 9-parameter search)

| Method | Joint | Kp | Ki | Kd | PM [deg] | wc [rad/s] | Delay margin [ms] |
|---|---|---|---|---|---|---|---|
| PSO | 1 | 1.0621 | 3.3516 | 0.0691 | 60.4 | 25.0 | 42.2 |
| PSO | 2 | 1.1000 | 6.6117 | 0.0474 | 55.6 | 25.4 | 38.1 |
| PSO | 3 | 0.4078 | 1.8709 | 0.0231 | 65.1 | 28.0 | 40.6 |
| GA | 1 | 0.9095 | 3.1899 | 0.0575 | 55.8 | 21.5 | 45.3 |
| GA | 2 | 1.2057 | 7.1030 | 0.0474 | 53.4 | 26.4 | 35.4 |
| GA | 3 | 0.4173 | 1.7201 | 0.0245 | 67.3 | 29.5 | 39.8 |

## Step response: local 1 kHz loop

| Controller | Joint | Overshoot [%] | Settling (2 %) [ms] | Steady-state error [deg] | ITAE | PM [deg] | Delay margin [ms] | Cost |
|---|---|---|---|---|---|---|---|---|
| Thesis PD | 1 | 0.0 | 1499 | 3.62 | 0.2415 | 66.6 | 32.6 | 4.573 |
| Thesis PD | 2 | 0.0 | 1499 | 7.84 | 0.5179 | 66.9 | 32.7 | 8.130 |
| Thesis PD | 3 | 0.0 | 1499 | 5.26 | 0.3514 | 66.8 | 32.7 | 5.963 |
| QUARC PID | 1 | 3.1 | 1499 | 1.08 | 0.0885 | 48.4 | 23.3 | 2.546 |
| PSO-PID | 1 | 0.3 | 232 | 0.05 | 0.0092 | 60.4 | 42.2 | 0.303 |
| PSO-PID | 2 | 0.9 | 231 | 0.06 | 0.0143 | 55.6 | 38.1 | 0.340 |
| PSO-PID | 3 | 1.9 | 244 | 0.26 | 0.0255 | 65.1 | 40.6 | 0.535 |
| GA-PID | 1 | 1.6 | 226 | 0.06 | 0.0134 | 55.8 | 45.3 | 0.347 |
| GA-PID | 2 | 0.7 | 217 | 0.12 | 0.0117 | 53.4 | 35.4 | 0.347 |
| GA-PID | 3 | 1.7 | 305 | 0.29 | 0.0224 | 67.3 | 39.8 | 0.600 |

## Step response: 20 ms UDP loop delay

| Controller | Joint | Overshoot [%] | Settling (2 %) [ms] | Steady-state error [deg] | ITAE | PM [deg] | Delay margin [ms] | Cost |
|---|---|---|---|---|---|---|---|---|
| Thesis PD | 1 | 0.0 | 1499 | 3.62 | 0.2399 | 66.6 | 32.6 | 4.566 |
| Thesis PD | 2 | 0.0 | 1499 | 7.76 | 0.5088 | 66.9 | 32.7 | 8.048 |
| Thesis PD | 3 | 0.0 | 1499 | 5.09 | 0.3347 | 66.8 | 32.7 | 5.800 |
| QUARC PID | 1 | 49.5 | 1499 | 1.11 | 0.0966 | 48.4 | 23.3 | 3.555 |
| PSO-PID | 1 | 0.0 | 158 | 0.01 | 0.0063 | 60.4 | 42.2 | 0.187 |
| PSO-PID | 2 | 0.1 | 159 | 0.00 | 0.0044 | 55.6 | 38.1 | 0.179 |
| PSO-PID | 3 | 1.8 | 264 | 0.30 | 0.0206 | 65.1 | 40.6 | 0.556 |
| GA-PID | 1 | 4.1 | 396 | 0.20 | 0.0152 | 55.8 | 45.3 | 0.659 |
| GA-PID | 2 | 3.3 | 187 | 0.08 | 0.0106 | 53.4 | 35.4 | 0.343 |
| GA-PID | 3 | 1.8 | 336 | 0.31 | 0.0207 | 67.3 | 39.8 | 0.635 |

## Seam-welding trajectory (4 s, 12 cm seam, 3 mm / 2 Hz weave, 50 Hz UDP commands)

| Controller | Loop delay [ms] | RMS error [mm] | Max error [mm] |
|---|---|---|---|
| Thesis PD | 0 | 33.787 | 35.140 |
| PSO-PID | 0 | 2.174 | 4.333 |
| GA-PID | 0 | 2.136 | 4.307 |
| Thesis PD | 20 | 33.634 | 34.993 |
| PSO-PID | 20 | 2.214 | 4.539 |
| GA-PID | 20 | 2.142 | 4.464 |
| Task-space velocity PID (PSO) | 0 | 0.728 | 1.407 |

## Delay robustness (simulated sweep 0-50 ms)

- **Thesis PD**: first unstable at 32.5 ms
- **PSO-PID**: first unstable at 42.5 ms
- **GA-PID**: first unstable at 37.5 ms

Velocity-loop PSO gains (task-space controller): J1: Kp=0.1528 Ki=2.4033 Kd=0.00114, J2: Kp=0.1132 Ki=2.4079 Kd=0.00062, J3: Kp=0.0445 Ki=0.8947 Kd=0.00022
