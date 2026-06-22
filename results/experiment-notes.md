## Results

The results of the first experiment, which was conducted as an ablation study beginning with the baseline implementation, are presented in Tables 1 and 2. We incrementally added features and tested multiple values for each feature. Overall, every implemented feature contributed positively to the heuristic function, as reflected by the increase in ELO scores relative to the baseline across all configurations.

For the defensive agent, introducing a reward for remaining near its own food improved the robustness of the agent, after which it did not lose any matches. Based on the ELO scores, including those of the opponent, the values of -0.001 for total Food Distance and -0.1 for closest Food Distance appeared to perform best. We also observed that overly large values caused the agent to focus less on pursuing enemies, which is its primary objective. For this reason, slightly smaller values were preferred.

### Table 1: Resulting ELO scores with different defensive heuristic values

| numInvaders | onDefense | invaderDistance | totalFoodDistance | closestFoodDistance | ELO score (opponent/own) | win/lose rate |
|---|---:|---:|---:|---:|---:|---:|
| -100 | 10 | -10 | - | - | 760/910 | 0.02/0.06 |
| -100 | 10 | -10 | -0.1 | - | 943/988 | 0.07/0.0 |
| -100 | 10 | -10 | -0.01 | - | 941/988 | 0.06/0.0 |
| -100 | 10 | -10 | -0.001 | - | 952/996 | 0.07/0.0 |
| -100 | 10 | -10 | - | -5 | 961/996 | 0.06/0.0 |
| -100 | 10 | -10 | - | -1 | 954/990 | 0.04/0.0 |
| -100 | 10 | -10 | - | -0.1 | 919/985 | 0.09/0.02 |
