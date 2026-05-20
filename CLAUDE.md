# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Single-robot DQN reinforcement learning system for autonomous Rare Earth Element (REE) detection and exploration in a 2D simulated 100×100 grid environment. Built on ROS2 with Python nodes.

## Build and run

```bash
# From workspace root (directory containing src/, build/, install/)
source /opt/ros/humble/setup.bash

# Build all packages
colcon build --packages-select ree_exploration_agent ree_exploration_server ree_exploration_viz

# Or build a single package
colcon build --packages-select ree_exploration_agent

# Source after every build
source install/setup.bash
```

Run in separate terminals (source install/setup.bash in each):

```bash
# Terminal 1 — Environment server (start first)
ros2 run ree_exploration_server server_node

# Terminal 2 — Agent (robot_id 0..3)
ros2 run ree_exploration_agent agent_node 0

# Terminal 3 — Visualization (optional)
ros2 launch ree_exploration_viz full_system.launch.py

# Or just the viz nodes
ros2 run ree_exploration_viz visualization_node
ros2 run ree_exploration_viz robot_marker_publisher
```

## Lint and test

```bash
# Flake8 + pep257 (ament test harness)
colcon test --packages-select ree_exploration_agent ree_exploration_server
colcon test-result --verbose

# Run directly without colcon
cd src/ree_exploration_agent && python3 -m flake8 ree_exploration_agent/
cd src/ree_exploration_agent && python3 -m pytest test/

# Smoke-test a module without ROS2
python3 -c "
import sys; sys.path.insert(0,'src/ree_exploration_agent')
from ree_exploration_agent.advanced_dqn_agent import RobustDQNAgent
a = RobustDQNAgent((100,100,8), 8, robot_id=0, load_latest_model=False, enable_logging=False)
print('OK', a.epsilon)
"
```

## Architecture

### Package layout

| Package | Role |
|---|---|
| `ree_exploration_server` | Procedural environment: generates mineral maps, obstacle maps, handles episode resets |
| `ree_exploration_agent` | DQN agent nodes: perception, decision, reward, logging, checkpointing |
| `ree_exploration_viz` | RViz2 visualization: PointCloud2 mineral heatmap, robot markers |

### Data flow

```
server_node  →  /mineral_map (100×100×4 float32)
             →  /obstacle_map (OccupancyGrid, 100×100 int8)
             →  /science_targets (100×100 float32)

agent_node   →  /robot_{id}/position  (Pose2D)
             →  /robot_{id}/cleaning_action (Float32MultiArray [x, y])
             →  /agent/dqn_update (Float32MultiArray [id, loss, ε, memory_size])
             →  /shared_discoveries (Float32MultiArray [id, x, y, reward, *mineral_data])

agent_node   ←  /episode_reset (String) → server regenerates map with new random seed
```

### State representation

Each agent observes a `(100, 100, 8)` tensor:

| Channel(s) | Content |
|---|---|
| 0–3 | Mineral concentrations (REE_Oxides, REE_Silicates, REE_Phosphates, REE_Carbonates), max-normalised |
| 4 | Current robot position (binary mask) |
| 5 | Obstacle map, scaled to [0, 1] |
| 6 | Visited-positions heatmap (0.5 per visited cell) |
| 7 | Science-target heatmap, max-normalised |

### Training loop

Training runs **inline inside `agent_node.py`** — there is no separate trainer node. `make_decision()` is called by a 1 Hz ROS2 timer:

1. Build state → `get_current_state()`
2. Epsilon-greedy action → `_select_action()`
3. Execute action, get reward → `execute_action()` → `_calculate_reward()`
4. Store experience in `agent.memory` (deque)
5. Every 2 steps, if `len(memory) >= batch_size`: call `agent.train()` (vectorised batch forward pass)
6. At episode end (`steps >= 50` or early-stop): `_handle_episode_end()` → save logs + checkpoint

### Reward system

`RealMineralRewardSystem` computes a **70/30 hybrid**: 70% real mineral concentrations + 30% academic heatmap (Gaussian-diffused). The heatmap is updated every 15 steps via `gaussian_filter`. Episode-level stats are logged to `logs/robot_{id}/csv/`.

### Checkpoints

Saved to `models/robot_{id}/latest_model.pth` (atomically, via `.tmp` + `os.replace`). Format:

```python
{
  'policy_net_state_dict': ..., 'target_net_state_dict': ...,
  'optimizer_state_dict': ..., 'epsilon': float,
  'step_count': int, 'episode_count': int, 'loss_history': list
}
```

Auto-loaded at startup; epsilon is multiplied by 0.7 after loading to reduce exploration.

## Configuration

All magic numbers live in frozen dataclasses — **never hardcode them inline**:

- `src/ree_exploration_agent/ree_exploration_agent/configs.py` → `TopicConfig`, `NetworkConfig`
- `src/ree_exploration_server/ree_exploration_server/configs.py` → `ServerTopicConfig`, `MapConfig`

Key `NetworkConfig` constants: `MAP_HEIGHT/WIDTH=100`, `STATE_CHANNELS=8`, `MINERAL_CHANNELS=4`, `NUM_ACTIONS=8`, `MAX_STEPS_PER_EPISODE=50`, `BATCH_SIZE=32`, `TARGET_UPDATE_FREQ=1000`.

## Invariants — do not break these

- **ROS2 topic names** must not change (other nodes depend on them); only the string values in `TopicConfig` / `ServerTopicConfig` are authoritative.
- **DQN algorithm logic** must not change: Bellman target `r + γ·max_Q(s') · (1−done)`, epsilon decay `ε *= ε_decay` after each training step, target network hard-update every `target_update` steps.
- **Checkpoint key names** must not change (backward compatibility with saved `.pth` files).
- **Mineral map shape** is always `(100, 100, 4)` and **state shape** is always `(100, 100, 8)` — the CNN input dimensions are fixed by the architecture.
- `publish_maps()` / `_publish_maps()` on the server must release `self.lock` before calling `.tolist()` and `.publish()` — holding the lock during serialisation blocks the single-threaded executor and starves agent nodes of map data.
- `publish_status()` on the server must never call `detect_mineral_clusters()` or any long-running operation (same single-thread executor constraint; this caused ~165 s blocking in the original code).

## Gitignored outputs

`models/`, `logs/`, `multi_models/`, `monitor_logs/`, `metrics_logs/` are all gitignored. Checkpoints and CSV training logs accumulate there at runtime and are not committed.
