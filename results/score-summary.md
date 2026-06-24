## Results and Discussion

The results of the first experiment, which was conducted as an ablation study beginning with the baseline implementation, are presented in Tables 1 and 2. We incrementally added features and tested multiple values for each feature. Overall, every implemented feature contributed positively to the heuristic function, as reflected by the increase in ELO scores relative to the baseline across all configurations.

For the defensive agent, introducing a reward for remaining near its own food improved the robustness of the agent, after which it did not lose any matches. Based on the ELO scores, including those of the opponent, the values of -0.001 for total Food Distance and -0.1 for closest Food Distance appeared to perform best. We also observed that overly large values caused the agent to focus less on pursuing enemies, which is its primary objective. For this reason, slightly smaller values were preferred.

Our heuristics were actually designed for short-term goals: gain food, stay away from ghosts, bring home food if needed. However, MCTS typically works well if it could find its own path to long-term objectives (win a game). This was actually due to time restrictions hard to implement as typically games would
run for 1200 steps. One simulation using 1200 steps already takes up to one minute and is typically not very informative.
After having our heuristics fine-tuned for MCTS, we saw very good behaviour for the MCTS against the baseline agents: a win-rate of 98% and ELO-scores
of over 1600. Letting it play against a reflexive agent with the same heuristics showed however that the heuristics were the power of the algorithm- and not
the MCTS itself.
