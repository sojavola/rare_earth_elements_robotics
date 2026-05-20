from dataclasses import dataclass


@dataclass(frozen=True)
class TopicConfig:
    """Authoritative ROS2 topic names for the ree_exploration_dqn package.

    Per-robot topics use '{id}' as a placeholder — format at runtime with
    .format(id=robot_id).  Never hardcode topic names elsewhere.
    """

    # Published by ree_exploration_server (read-only for this package)
    MINERAL_MAP: str = "/mineral_map"
    OBSTACLE_MAP: str = "/obstacle_map"
    SCIENCE_TARGETS: str = "/science_targets"
    EPISODE_RESET: str = "/episode_reset"   # agent → server: triggers map regen

    # Agent → Trainer: JSON + base64 float16 encoded experience tuples
    # One topic per robot so each trainer only sees its own experiences.
    AGENT_EXPERIENCE_FMT: str = "/robot_{id}/agent_experience"

    # Trainer → Agent: zlib-compressed weight state_dict (base64 String)
    WEIGHT_UPDATE_FMT: str = "/robot_{id}/dqn/weight_update"

    # Telemetry (agent outbound)
    AGENT_EPSILON_FMT: str = "/robot_{id}/agent/epsilon"
    AGENT_STEP: str = "/agent/step_completed"
    SHARED_DISCOVERIES: str = "/shared_discoveries"


@dataclass(frozen=True)
class NetworkConfig:
    """Architecture constants and default training hyperparameters.

    CNN follows the article architecture (Caccavale et al., 2023, Fig. 4):
        Conv(C, 32, 8×8/4) → Conv(32, 64, 4×4/2) → Conv(64, 64, 3×3/1)
        → Flatten → Dense(512) → Dense(num_actions)

    Values here can be overridden at runtime via dqn_params.yaml.
    """

    # Map dimensions — fixed by ree_exploration_server; do not change
    MAP_HEIGHT: int = 100
    MAP_WIDTH: int = 100
    MINERAL_CHANNELS: int = 4

    # Observation: global 100×100 map — 6 canaux : 4 minéraux + obstacles + exploration
    # Canal 0-3 : REE mineral concentrations (server → /mineral_map)
    # Canal 4   : obstacle map              (server → /obstacle_map)
    # Canal 5   : exploration map           (agent-side: cells visited by this robot)
    OBS_CHANNELS: int = 6
    NUM_ACTIONS: int = 8      # N S W E NW NE SW SE

    # Training defaults (overridden by dqn_params.yaml)
    LEARNING_RATE: float = 0.00025      # Adam, as in article
    GAMMA: float = 0.99                 # discount factor, as in article
    EPSILON_START: float = 1.0
    EPSILON_MIN: float = 0.05
    # Article uses linear decay 9e-7/step; we keep multiplicative for
    # compatibility with the existing checkpoint format
    EPSILON_DECAY: float = 9e-7        # linear: ε -= EPSILON_DECAY each step
    MEMORY_SIZE: int = 10_000          # article: 10^4
    BATCH_SIZE: int = 32               # article: 32
    TARGET_UPDATE_FREQ: int = 10_000   # article: 10^4 steps
    TRAIN_EVERY_N_STEPS: int = 4       # article: main network update every 4

    # Episode termination
    MAX_STEPS_PER_EPISODE: int = 300   # benchmark: aligned with QMIX (300 steps)
    MINERALS_TO_COMPLETE: int = 2
    CLEAN_THRESHOLD: float = 0.98      # article: 98 % of map clean

    # Thresholds
    MINERAL_DETECTION_THRESHOLD: float = 0.3
    OBSTACLE_THRESHOLD: int = 50
    PENALTY: float = -2.0              # article: penalty = -2

    # Timing
    DECISION_HZ: float = 10.0
    CMD_VEL_LINEAR: float = 0.15

    # Spawning margin (cells from map border)
    SPAWN_MARGIN: int = 5

    # Weight broadcast interval (every N training steps)
    WEIGHT_BROADCAST_FREQ: int = 50

    # Paths (relative to CWD at launch — workspace root)
    # Each trainer appends /robot_{id}/ at runtime.
    MODEL_BASE_DIR: str = "models/dqn"
    TB_BASE_DIR: str = "tensorboard_logs/dqn"
    CSV_BASE_DIR: str = "logs/dqn"
    BUFFER_FILENAME: str = "replay_buffer.pkl.gz"

    # Watchdog: warn if no experience arrives within this window (seconds)
    WATCHDOG_TIMEOUT_SEC: float = 30.0
