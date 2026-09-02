# Vacuum Cleaner Expert System

## Overview
This is a **Prolog-based expert system** that demonstrates knowledge bases and inference rules through a simulated vacuum cleaner agent. The system autonomously navigates rooms, identifies dirty areas, and cleans them based on logical rules.

## Project Structure
```
lab1/
├── lab1.pl        # Main Prolog expert system implementation
└── README.md      # This file
```

## Features

### Knowledge Base
- **Rooms**: Defines three rooms (a, b, c) in the environment
- **Adjacency**: Specifies which rooms are connected to each other
- **Initial State**: Tracks initial dirty rooms and vacuum location

### Inference Rules (Expert System Rules)

| Rule | Purpose |
|------|---------|
| **Clean** | If vacuum is in a dirty room, clean it |
| **Move** | Move to an unvisited adjacent room |
| **Stop** | Stop when all rooms are clean |

### Core Components

#### 1. **Knowledge Base** (Facts)
- Room definitions and adjacency relationships
- Initial vacuum location (room a)
- Initial dirt locations (rooms a and c)
- Visited rooms tracking

#### 2. **Inference Rules** (Logic)
- Determines what action the vacuum should take next
- Three possible actions: `clean`, `move(ToRoom)`, or `stop`

#### 3. **Action Execution** (Performance)
- Implements the actual effects of each action
- Updates the knowledge base dynamically using Prolog's `assert`/`retract`

#### 4. **Control Loop** (start/0)
- Continuously finds and executes actions until the stop condition is met

## How It Works

1. **Query**: `?- start.`
2. **Execution Flow**:
   - `action(Action)` finds the next action to perform
   - `perform(Action)` executes the action and updates the knowledge base
   - Loop continues recursively until `action(stop)` is triggered
3. **Termination**: All rooms are clean

## Example Output
```
Vacuum cleaned room a.
Vacuum moved from room a to room b.
Vacuum moved from room b to room c.
Vacuum cleaned room c.
All rooms are clean. Stopping...
```

## How to Run

### Using SWI-Prolog
```prolog
% Load the file
?- consult('lab1.pl').

% Start the vacuum cleaner
?- start.
```

### Using Command Line
```bash
swipl -f lab1.pl -t "start, halt."
```

## Key Prolog Concepts Demonstrated

- **Dynamic Predicates**: `dynamic/1` directive for modifiable facts
- **Unification**: Pattern matching in rules
- **Backtracking**: Finding alternative actions
- **Assert/Retract**: Dynamic knowledge base modification
- **Negation**: `\+` operator for "not provable"
- **Cut-like Behavior**: Conditional logic in control flow

## Lab Assignment
- **Course**: AIML (Artificial Intelligence & Machine Learning)
- **Institution**: MIT AOE (MIT Academy of Engineering)
- **Level**: TY (Third Year)
- **Lab**: 1 - Expert Systems Basics

## Learning Objectives
✓ Understand expert systems architecture  
✓ Implement knowledge bases in Prolog  
✓ Define inference rules for decision-making  
✓ Use dynamic facts for state management  
✓ Build reactive control loops  

## Future Enhancements
- Add more complex room layouts
- Implement obstacle detection and avoidance
- Add priority-based cleaning
- Implement more sophisticated pathfinding algorithms
- Add performance metrics (time, energy consumption)
