# Pac-Man-Capture-the-Flag-Agent-with-MCTS-Offense-and-Rule-Based-Defense

A Python Game AI agent for the Pac-Man Capture the Flag environment. The team combines an offensive Pacman agent with MCTS-inspired return-home planning and a defensive Ghost agent with rule-based invader chasing.

## Project Snapshot

| Area                  | Details                                                                                                                                    |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Status**            | Complete v2 implementation; evaluation upgrade in progress                                                                                 |
| **Problem**           | Balance food collection, survival, return-home timing, and defensive coverage in an adversarial Capture-the-Flag environment               |
| **Approach**          | Hybrid team: MCTS-inspired return-home planning for the offensive agent and rule-based invader chasing for the defensive agent             |
| **Tech**              | Python, multi-agent systems, heuristic search, UCT-style selection, adversarial game AI                                                    |
| **Reproducibility**   | Run inside the Berkeley Pac-Man Capture-the-Flag framework using `capture.py`                                                              |
| **Validation**        | Score tracking and gameplay outputs are included; seeded benchmark logs and ablation results will be published under `results/evaluation/` |
| **Key design choice** | Search is invoked selectively when risk rises or carried food makes returning home more valuable than collecting another pellet            |
| **Scope**             | Educational game-AI project; not intended as a full stochastic opponent-modelling system                                                   |


Repository Summary
Field	Details
Project type	Game AI / multi-agent decision-making
Environment	Pac-Man Capture the Flag
Main techniques	MCTS-inspired search, heuristic evaluation, rule-based defense
Language	Python
Main file	`myTeam.py`
Agents	`PacmanAgent`, `GhostAgent`
Evaluation support	Score tracking and plotting helper
Main objective	Build an attacker-defender team for adversarial grid-world play
Problem
Pac-Man Capture the Flag is a multi-agent adversarial game. A strong team must attack and defend at the same time.
The offensive agent needs to:
Enter enemy territory
Collect food
Avoid dangerous ghosts
Return home before losing carried food
Exploit moments when enemy ghosts are scared
The defensive agent needs to:
Stay on home territory
Detect invading enemy Pacmen
Chase visible invaders
Avoid wasting turns with poor actions
The real problem is not simply “move toward food.”
The real problem is:
```text
How can an agent balance food collection, survival, return-home timing, and defensive coverage under adversarial pressure?
```
Solution
The project uses a hybrid strategy.
The offensive agent uses food-seeking heuristics when safe. When risk increases, or when enough food has been collected, it switches to an MCTS-inspired search that prefers actions moving the agent toward the home border.
The defensive agent uses a weighted feature evaluation to stay defensive, chase invading Pacmen, avoid stopping, and discourage unnecessary reversals.
Team Composition
The Capture the Flag framework calls:
```python
def createTeam(firstIndex, secondIndex, isRed, first="PacmanAgent", second="GhostAgent"):
    ...
```
Default team:
Agent	Role
`PacmanAgent`	Offensive food collection and return-home planning
`GhostAgent`	Defensive invader detection and chasing
V2 replaces dynamic `eval()` with a safe registry:
```python
agent_registry = {
    "PacmanAgent": PacmanAgent,
    "GhostAgent": GhostAgent,
}
```
Agent Strategy
PacmanAgent
The offensive agent switches behavior based on game state.
Situation	Behavior
Safe and carrying little food	Move toward food
Enemy ghost nearby	Use MCTS-inspired return-home planning
Carrying more than 5 food	Return home
Enemy ghost scared	Continue aggressive food collection
On home side as ghost	Use defensive fallback behavior
Food-seeking weights:
```python
{
    "eat_food": 100,
    "distance_to_food": -1
}
```
This rewards eating food and penalizes distance to the nearest food.
GhostAgent
The defensive agent evaluates each legal action using features.
Feature	Purpose
`on_defense`	Reward staying as a ghost on home territory
`num_invaders`	Penalize presence of enemy Pacmen
`invader_distance`	Move toward visible invaders
`stop`	Penalize stopping
`reverse`	Slightly discourage reversing
Defensive weights:
```python
{
    "on_defense": 100,
    "num_invaders": -1000,
    "invader_distance": -10,
    "stop": -100,
    "reverse": -2
}
```
MCTS-Inspired Return Planning
The v2 project includes an `MCTSNode` class.
The search includes:
Expansion over non-stop legal actions
UCT-style child selection
Reward based on distance to the home border
Backpropagation through parent nodes
Time budget of roughly 0.82 seconds
The return-home reward is simple:
```text
reward = -distance_to_home_border
```
This means shorter distance to the border is better.
The MCTS module is used when the agent is carrying enough food or when enemy ghost pressure is nearby.
Why MCTS Is Used Selectively
Running search on every move would be expensive and unnecessary.
The agent only uses MCTS-style planning when tactical risk increases:
A ghost is close
The agent has collected enough food
Very little food remains
Returning home is more valuable than chasing another pellet
This keeps the agent responsive while still giving it a planning mechanism for high-risk decisions.
Code Workflow
```text
Create team
        |
Register initial state
        |
Compute map dimensions and home border
        |
For each turn:
        |
Update score tracker
        |
Check whether offensive agent is Pacman or ghost
        |
If Pacman:
    Check carried food, ghost threats, scared timers
    Choose food-seeking action or MCTS return-home action
        |
If Ghost:
    Evaluate defensive actions
    Chase invaders or stay defensive
```
Installation
This agent is designed to run inside a Pac-Man Capture the Flag framework that provides:
```text
capture.py
captureAgents.py
game.py
util.py
```
Place the v2 file as:
```text
myTeam.py
```
inside the Capture the Flag project directory.
Recommended project structure:
```text
pacman-capture-the-flag-mcts-agent/
│
├── myTeam.py
├── README.md
├── requirements.txt
├── .gitignore
└── reports/
    └── results_summary.md
```
Run the Agent
Run one match:
```bash
python capture.py -r myTeam -b baselineTeam
```
Run 10 games:
```bash
python capture.py -r myTeam -b baselineTeam -n 10
```
Run quietly without graphics:
```bash
python capture.py -r myTeam -b baselineTeam -n 10 -q
```
Exact flags may vary depending on the Pac-Man Capture the Flag framework version.
First-Principles Design
The basic food-chasing policy is not enough.
A good offensive agent must ask:
```text
Is another food pellet worth the risk of losing everything I am carrying?
```
That is the core design behind the behavior switch. When the agent is safe, it pushes for food. When risk rises, it plans a route home.
OODA Summary
Observe
The game state changes quickly. Food, enemies, carried food, scared timers, and border distance all affect the best action.
Orient
A single fixed policy is weak. The agent needs different behaviors for attack, escape, and defense.
Decide
Use feature-based food seeking when safe, MCTS-inspired border planning when risk rises, and rule-based defense for home territory.
Act
Select an action using the current tactical mode, then update score tracking for later evaluation.
Founder-Style Product Diagnosis
User
A student, game-AI learner or a multi-agent systems researcher
Pain Point
Simple reflex agents often collect food but die because they lack risk management and return-home timing.
Smallest Useful Version
A hybrid team that collects food, detects ghost threats, retreats when carrying value, and chases invaders.
Security and Code Safety Notes
This project is low-risk because it runs locally inside a game framework and does not use external APIs or private data.
Area	Status
API keys	None
Secrets	None
User data	None
File uploads	None
Network calls	None
Dynamic code execution	Removed from team creation
Main safety improvement	Replaced `eval()` with class registry
Scientific and AI Skills Demonstrated
This project demonstrates:
Multi-agent game AI
Search-based decision-making
MCTS-inspired planning
UCT-style child selection
Heuristic feature design
Defensive rule-based agents
Reward design
Risk-aware behavior switching
Score tracking and visualization
Adversarial grid-world reasoning
The strongest skill shown is not just MCTS. It is knowing when search is useful and when a cheap heuristic is enough.
Limitations
MCTS reward is still simple
No full stochastic rollout policy
No opponent behavior model
No capsule-specific strategy
No saved CSV match logs
No formal win-rate table
Defensive agent is reactive rather than predictive
Score tracker is lightweight and should be extended for serious experiments
Future Improvements
Add full rollout simulations inside MCTS
Add richer reward with ghost distance, food carried, and capsule distance
Add opponent modeling
Add capsule-aware attack behavior
Add patrol zones for defense
Add A* escape route fallback
Save match results to CSV
Add win-rate and average-score reporting
Add ablation tests: reflex-only vs MCTS-return behavior
Move evaluation into a separate `evaluate.py`
Add plots saved to `plots/`

`.gitignore`
```text
__pycache__/
*.pyc
.env
.venv/
venv/
plots/*.tmp
reports/*.tmp
```

SEO Keywords
Relevant keywords:
Pac-Man Capture the Flag AI
Monte Carlo Tree Search
MCTS game agent
multi-agent game AI
Pacman AI agent
heuristic game AI
adversarial search
defensive agent
offensive Pacman agent
Python game AI project

