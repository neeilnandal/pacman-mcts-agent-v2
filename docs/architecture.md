# Architecture

## Overview

This project implements a Monte Carlo Tree Search (MCTS) agent for the Pac-Man Capture the Flag environment.

The agent operates in a partially observable, adversarial, time-constrained setting. At each game step, it must decide whether to collect food, return safely, avoid opponents, defend territory, or pursue an invading opponent.

The architecture separates game interaction, feature extraction, planning, evaluation, rollout simulation, and action selection. This keeps the MCTS logic understandable and makes later improvements easier to test.

## High-Level System Design

```text
Pac-Man Capture the Flag Environment
                |
                v
          myTeam.py
                |
                v
        Agent Decision Layer
                |
     +----------+-----------+
     |                      |
     v                      v
Feature Extraction      MCTS Planner
     |                      |
     v                      v
State Features       Search Tree Construction
                            |
                            v
                    Selection / Expansion
                            |
                            v
                     Rollout Simulation
                            |
                            v
                     State Evaluation
                            |
                            v
                     Backpropagation
                            |
                            v
                    Best Action Selection
                            |
                            v
                  Action Returned to Game
```

## Runtime Decision Flow

At every decision point, the agent follows this sequence:

```text
1. Observe current game state
2. Extract relevant state features
3. Determine legal actions
4. Build or refresh MCTS search tree
5. Run MCTS iterations within time budget
6. Select highest-value action
7. Return action to game environment
```

## MCTS Planning Cycle

The planner follows the standard Monte Carlo Tree Search process.

```text
Selection
    |
    v
Expansion
    |
    v
Simulation / Rollout
    |
    v
Backpropagation
    |
    v
Best Action Selection
```

### 1. Selection

Starting from the root node, the agent selects child nodes using an exploration-exploitation strategy such as Upper Confidence Bound for Trees (UCT).

The aim is to balance:

* Exploiting actions with strong observed reward
* Exploring actions that have not been sampled enough

### 2. Expansion

When the selected node has unexplored legal actions, the agent creates a child node for one of those actions.

Each child represents a possible future game state after applying an action.

### 3. Simulation

The agent simulates future actions from the newly expanded node using a rollout policy.

The rollout does not need to perfectly predict opponent behaviour. Its purpose is to estimate whether a branch is strategically promising.

### 4. Backpropagation

The final rollout score is propagated from the simulated node back to the root.

Each visited node updates:

* Visit count
* Cumulative reward
* Average reward estimate

### 5. Final Action Selection

After the time or iteration budget is reached, the agent chooses the root child with the strongest value estimate or highest visit count.

## State Representation

The agent does not treat the game board as raw pixels. It uses structured game-state information supplied by the Pac-Man framework.

Important state signals include:

| Feature Group    | Example Features                                            |
| ---------------- | ----------------------------------------------------------- |
| Team progress    | Current score, food collected, food remaining               |
| Agent position   | Current location, distance to home boundary                 |
| Resource targets | Distance to food, capsules, high-value routes               |
| Risk             | Distance to visible opponents, nearby ghosts, capture risk  |
| Defensive status | Invader position, distance to defended food, enemy activity |
| Carry state      | Number of food items currently carried                      |
| Mobility         | Legal actions, dead ends, escape routes                     |

## Offensive Decision Layer

The offensive logic prioritises collecting food while balancing reward against capture risk.

Typical priorities:

```text
Food available and safe
        |
        v
Move toward high-value food target
        |
Opponent becomes threatening
        |
        v
Reassess route, capsule, or return-home option
        |
Carrying sufficient food or risk is high
        |
        v
Return safely to home territory
```

The agent should avoid acting greedily when carrying food. Collecting one more item is often worse than banking the food already collected.

## Defensive Decision Layer

The defensive logic is activated when opponents enter the agent’s territory or threaten defended food.

Typical priorities:

```text
Opponent invades home territory
        |
        v
Estimate opponent location and threat level
        |
        v
Move toward interception route
        |
        v
Protect high-value food clusters or key map transitions
```

Defence should not blindly chase opponents. The evaluator should consider whether interception is feasible and whether the agent is abandoning a more valuable offensive position.


