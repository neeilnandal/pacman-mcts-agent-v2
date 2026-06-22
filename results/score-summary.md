Our heuristics were actually designed for short-term goals: gain food, stay
away from ghosts, bring home food if needed. However, MCTS typically works
well if it could find its own path to long-term objectives (win a game). This was
actually due to time restrictions hard to implement as typically games would
run for 1200 steps. One simulation using 1200 steps already takes up to one
minute and is typically not very informative.
After having our heuristics fine-tuned for MCTS, we saw very good behaviour
9
for the MCTS against the baseline agents: a win-rate of 98% and ELO-scores
of over 1600. Letting it play against a reflexive agent with the same heuristics
showed however that the heuristics were the power of the algorithm- and not
the MCTS itself.
