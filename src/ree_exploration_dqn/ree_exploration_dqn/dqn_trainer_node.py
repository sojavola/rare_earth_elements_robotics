#!/usr/bin/env python3
"""
DQN Trainer Node — ree_exploration_dqn
========================================
Encodeur CNN pour cartes locales (6 canaux : 4 minéraux + obstacles + exploration).

Per-robot trainer: 1 DQN per robot, 1 model per robot (DTDE).

  - Receives encoded experience tuples from dqn_agent_node via
    /robot_{id}/agent_experience
  - Stores them in a float16 ReplayBuffer (thread-safe)
  - Trains the Q-network in a background thread — NEVER inside a ROS2 callback
  - Broadcasts updated weights via /robot_{id}/dqn/weight_update every N steps
  - Logs scalars to TensorBoard and rows to a CSV file
  - Watchdog: warns if no experience arrives for WATCHDOG_TIMEOUT_SEC seconds
  - Saves atomic checkpoints every 60 s and on SIGTERM
  - On SIGTERM: checkpoint + buffer save + clean shutdown

TensorBoard scalars:
  Train/Loss, Train/GradNorm, Train/Epsilon
  Episode/TotalReward, Episode/TotalReward_MA50, Episode/MineralsDetected,
  Episode/Steps, Episode/Epsilon, Eval/AvgReward, Eval/AvgRewardPerStep
"""

import base64
import csv
import json
import os
import signal
import sys
import threading
import time
import zlib
from collections import deque
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import rclpy
import torch
import torch.nn.functional as F
import torch.optim as optim
from rclpy.node import Node
from std_msgs.msg import Float32, String

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

from .config import NetworkConfig, TopicConfig
from .networks import QNetwork
from .replay_buffer import ReplayBuffer

_T = TopicConfig()
_N = NetworkConfig()

_CSV_FIELDS = [
    'episode', 'steps', 'total_reward', 'avg_reward',
    'minerals', 'epsilon', 'loss_avg', 'grad_norm_avg',
    'reward_ma50', 'timestamp',
]


class DQNTrainerNode(Node):
    """Trains one robot's DQN from buffered experiences (background thread)."""

    def __init__(self) -> None:
        super().__init__('dqn_trainer')

        # robot_id must be loaded before all other params (paths depend on it)
        self.declare_parameter('robot_id', 0)
        self._robot_id: int = int(self.get_parameter('robot_id').value)
        self._tag = f'[DQN_TRAINER robot_{self._robot_id}]'

        self._declare_params()
        self._load_params()
        self._init_dirs()

        # These must be initialised BEFORE _init_networks so that
        # _load_checkpoint() can overwrite them with checkpoint values.
        # Placing them after _init_networks causes checkpoint values to be
        # silently overwritten — the root cause of episode/step reset on resume.
        self._shutdown_event = threading.Event()
        self._last_exp_time = time.monotonic()
        self._step_count: int = 0
        self._ep_count: int = 0
        self._current_ep: int = -1

        # Try-locks: prevent two concurrent background threads from writing to
        # the same .tmp file (causes gzip / pickle corruption on next load).
        self._checkpoint_saving = threading.Lock()
        self._buffer_saving     = threading.Lock()

        self._init_networks()   # may overwrite _step_count/_ep_count from checkpoint
        self._init_buffer()
        self._init_logging()
        self._init_episode_state()

        exp_topic = _T.AGENT_EXPERIENCE_FMT.format(id=self._robot_id)
        weight_topic = _T.WEIGHT_UPDATE_FMT.format(id=self._robot_id)
        eps_topic = _T.AGENT_EPSILON_FMT.format(id=self._robot_id)

        self.create_subscription(String, exp_topic, self._exp_cb, 100)
        self._weight_pub = self.create_publisher(String, weight_topic, 10)
        self._eps_pub = self.create_publisher(Float32, eps_topic, 10)

        self.create_timer(5.0, self._watchdog_check)
        self.create_timer(60.0, self._periodic_checkpoint)

        self._train_thread = threading.Thread(
            target=self._training_loop,
            daemon=True,
            name=f'dqn-train-{self._robot_id}',
        )
        self._train_thread.start()

        # Handle both SIGTERM (from ROS2 launch) and SIGINT (Ctrl+C) the same
        # way so the checkpoint + buffer are always saved on shutdown.
        signal.signal(signal.SIGTERM, self._sigterm_handler)
        signal.signal(signal.SIGINT,  self._sigterm_handler)

        self.get_logger().info(f'{self._tag} {self._buffer}')
        self.get_logger().info(
            f'{self._tag} Model dir        : {os.path.abspath(self._model_dir)}'
        )
        self.get_logger().info(
            f'{self._tag} TB logs          : tensorboard --logdir '
            f'{os.path.abspath(self._tb_log_dir)}'
        )
        self.get_logger().info(
            f'{self._tag} Experience topic : {exp_topic}'
        )
        self.get_logger().info(
            f'{self._tag} Weight topic     : {weight_topic}'
        )

    # ------------------------------------------------------------------ #
    #  PARAMETER HANDLING                                                  #
    # ------------------------------------------------------------------ #

    def _declare_params(self) -> None:
        rid = self._robot_id
        self.declare_parameter('model_dir',   f'{_N.MODEL_BASE_DIR}/robot_{rid}')
        self.declare_parameter('tb_log_dir',  f'{_N.TB_BASE_DIR}/robot_{rid}')
        self.declare_parameter('csv_log_dir', f'{_N.CSV_BASE_DIR}/robot_{rid}')
        self.declare_parameter('learning_rate',      _N.LEARNING_RATE)
        self.declare_parameter('gamma',              _N.GAMMA)
        self.declare_parameter('epsilon_start',      _N.EPSILON_START)
        self.declare_parameter('epsilon_min',        _N.EPSILON_MIN)
        self.declare_parameter('epsilon_decay',      _N.EPSILON_DECAY)
        self.declare_parameter('memory_size',        _N.MEMORY_SIZE)
        self.declare_parameter('batch_size',         _N.BATCH_SIZE)
        self.declare_parameter('target_update_freq', _N.TARGET_UPDATE_FREQ)
        self.declare_parameter('grad_clip_norm',     1.0)
        self.declare_parameter('watchdog_timeout_sec', _N.WATCHDOG_TIMEOUT_SEC)
        self.declare_parameter('weight_broadcast_freq', _N.WEIGHT_BROADCAST_FREQ)

    def _load_params(self) -> None:
        self._model_dir   = self.get_parameter('model_dir').value
        self._tb_log_dir  = self.get_parameter('tb_log_dir').value
        self._csv_log_dir = self.get_parameter('csv_log_dir').value
        # Buffer lives alongside the checkpoint in model_dir
        self._buffer_path = os.path.join(self._model_dir, _N.BUFFER_FILENAME)
        self._lr          = float(self.get_parameter('learning_rate').value)
        self._gamma       = float(self.get_parameter('gamma').value)
        self._epsilon     = float(self.get_parameter('epsilon_start').value)
        self._eps_min     = float(self.get_parameter('epsilon_min').value)
        self._eps_decay   = float(self.get_parameter('epsilon_decay').value)
        self._memory_size = int(self.get_parameter('memory_size').value)
        self._batch_size  = int(self.get_parameter('batch_size').value)
        self._target_update_freq = int(
            self.get_parameter('target_update_freq').value
        )
        self._grad_clip        = float(self.get_parameter('grad_clip_norm').value)
        self._watchdog_timeout = float(
            self.get_parameter('watchdog_timeout_sec').value
        )
        self._weight_bcast_freq = int(
            self.get_parameter('weight_broadcast_freq').value
        )

    # ------------------------------------------------------------------ #
    #  SETUP                                                               #
    # ------------------------------------------------------------------ #

    def _init_dirs(self) -> None:
        # Convert to absolute paths so background threads work regardless of CWD.
        self._model_dir   = os.path.abspath(self._model_dir)
        self._tb_log_dir  = os.path.abspath(self._tb_log_dir)
        self._csv_log_dir = os.path.abspath(self._csv_log_dir)
        self._buffer_path = os.path.abspath(self._buffer_path)
        for d in (self._model_dir, self._tb_log_dir, self._csv_log_dir):
            os.makedirs(d, exist_ok=True)

    def _init_networks(self) -> None:
        # Limit PyTorch to 1 OS thread per process.  Without this, each trainer
        # spawns N_cores threads for matmul, and 4 trainers × N_cores threads
        # saturate all CPU cores — agent timer callbacks can't get scheduled.
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        # Lock guards policy_net between training thread and checkpoint timer
        self._net_lock = threading.Lock()

        self._policy_net = QNetwork(
            obs_channels=_N.OBS_CHANNELS, num_actions=_N.NUM_ACTIONS
        ).to(self._device)
        self._target_net = QNetwork(
            obs_channels=_N.OBS_CHANNELS, num_actions=_N.NUM_ACTIONS
        ).to(self._device)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()

        self._optimizer = optim.Adam(
            self._policy_net.parameters(), lr=self._lr
        )
        self._load_checkpoint()

    def _init_buffer(self) -> None:
        obs_shape = (_N.MAP_HEIGHT, _N.MAP_WIDTH, _N.OBS_CHANNELS)
        self._buffer = ReplayBuffer(self._memory_size, obs_shape)
        if self._buffer.load(self._buffer_path):
            self.get_logger().info(
                f'{self._tag} Buffer restored: {self._buffer}'
            )

    def _init_logging(self) -> None:
        if _TB_AVAILABLE:
            self._tb: Optional[SummaryWriter] = SummaryWriter(
                log_dir=self._tb_log_dir
            )
        else:
            self._tb = None
            self.get_logger().warning(
                f'{self._tag} TensorBoard unavailable (pip install tensorboard)'
            )

        csv_path = os.path.join(self._csv_log_dir, 'episodes.csv')
        self._csv_path = csv_path
        write_header = not os.path.exists(csv_path)
        self._csv_fh = open(csv_path, 'a', newline='')
        self._csv_writer = csv.DictWriter(
            self._csv_fh, fieldnames=_CSV_FIELDS
        )
        if write_header:
            self._csv_writer.writeheader()

    def _init_episode_state(self) -> None:
        self._ep_reward:   float = 0.0
        self._ep_steps:    int   = 0
        self._ep_minerals: int   = 0
        self._ep_loss_sum: float = 0.0
        self._ep_loss_n:   int   = 0
        self._ep_gnorm_sum: float = 0.0
        self._reward_ma: deque = deque(maxlen=50)

    # ------------------------------------------------------------------ #
    #  EXPERIENCE CALLBACK (ROS2 thread — decode + push only, no training) #
    # ------------------------------------------------------------------ #

    def _exp_cb(self, msg: String) -> None:
        try:
            p = json.loads(msg.data)
            shape = tuple(p['obs_shape'])
            compressed = p.get('compressed', False)
            if compressed:
                obs_bytes      = zlib.decompress(base64.b64decode(p['obs']))
                next_obs_bytes = zlib.decompress(base64.b64decode(p['next_obs']))
            else:
                obs_bytes      = base64.b64decode(p['obs'])
                next_obs_bytes = base64.b64decode(p['next_obs'])
            obs = np.frombuffer(obs_bytes, dtype=np.float16).reshape(shape).astype(np.float32)
            next_obs = np.frombuffer(next_obs_bytes, dtype=np.float16).reshape(shape).astype(np.float32)
            pos      = np.array(p.get('pos',      [0, 0]), dtype=np.float32)
            next_pos = np.array(p.get('next_pos', [0, 0]), dtype=np.float32)

            self._buffer.push(
                obs, pos,
                int(p['action']), float(p['reward']),
                next_obs, next_pos,
                bool(p['done']),
            )
            self._last_exp_time = time.monotonic()
            self._accumulate_episode(p)

        except Exception as exc:
            self.get_logger().warning(
                f'{self._tag} [WARN] _exp_cb: {exc}'
            )

    def _accumulate_episode(self, p: dict) -> None:
        """Track episode metrics; flush when the episode number increments."""
        ep     = int(p.get('ep', 0))
        reward = float(p['reward'])

        if ep != self._current_ep:
            if self._current_ep >= 0:
                self._flush_episode()
            self._current_ep   = ep
            self._ep_reward    = 0.0
            self._ep_steps     = 0
            self._ep_minerals  = 0

        self._ep_reward += reward
        self._ep_steps  += 1
        if reward > 15.0:
            self._ep_minerals += 1

    def _flush_episode(self) -> None:
        """Write completed-episode metrics to TensorBoard and CSV."""
        ep       = self._ep_count
        n_steps  = max(self._ep_steps, 1)
        n_loss   = max(self._ep_loss_n, 1)
        avg_rew  = self._ep_reward / n_steps
        loss_avg = self._ep_loss_sum / n_loss
        gnorm_avg = self._ep_gnorm_sum / n_loss

        self._reward_ma.append(self._ep_reward)
        ma50 = float(np.mean(self._reward_ma))

        if self._tb is not None:
            self._tb.add_scalar('Episode/TotalReward',     self._ep_reward,   ep)
            self._tb.add_scalar('Episode/TotalReward_MA50', ma50,             ep)
            self._tb.add_scalar('Episode/MineralsDetected', self._ep_minerals, ep)
            self._tb.add_scalar('Episode/Steps',            self._ep_steps,   ep)
            self._tb.add_scalar('Episode/Epsilon',          self._epsilon,    ep)
            self._tb.add_scalar('Eval/AvgReward',           avg_rew,          ep)
            self._tb.add_scalar('Eval/AvgRewardPerStep',    avg_rew,          ep)

        self._csv_writer.writerow({
            'episode':       ep,
            'steps':         self._ep_steps,
            'total_reward':  round(self._ep_reward, 4),
            'avg_reward':    round(avg_rew, 4),
            'minerals':      self._ep_minerals,
            'epsilon':       round(self._epsilon, 5),
            'loss_avg':      round(loss_avg, 6),
            'grad_norm_avg': round(gnorm_avg, 4),
            'reward_ma50':   round(ma50, 4),
            'timestamp':     time.strftime('%Y-%m-%dT%H:%M:%S'),
        })
        self._csv_fh.flush()

        self._ep_count    += 1
        self._ep_loss_sum  = 0.0
        self._ep_loss_n    = 0
        self._ep_gnorm_sum = 0.0

    # ------------------------------------------------------------------ #
    #  TRAINING LOOP (background thread — never inside ROS2 callbacks)    #
    # ------------------------------------------------------------------ #

    def _training_loop(self) -> None:
        self.get_logger().info(f'{self._tag} Training thread started')
        # With torch.set_num_threads(1) each CNN step takes ~4-8 s on CPU.
        # Sleep 1 s after each step so the OS scheduler has a guaranteed window
        # to run agent timer callbacks on the same core (time-slicing).
        _TRAIN_INTERVAL = 1.0

        while not self._shutdown_event.is_set():
            if len(self._buffer) < self._batch_size:
                self._shutdown_event.wait(timeout=0.05)
                continue

            batch = self._buffer.sample(self._batch_size)
            if batch is None:
                continue

            loss_val, grad_norm = self._train_step(batch)

            self._step_count    += 1
            self._ep_loss_sum   += loss_val
            self._ep_loss_n     += 1
            self._ep_gnorm_sum  += grad_norm

            self._decay_epsilon()
            self._maybe_update_target()
            self._log_train_step(loss_val, grad_norm)

            if self._step_count % self._weight_bcast_freq == 0:
                self._broadcast_weights()
                self._publish_epsilon()

            # Yield CPU so the agent's 10-Hz decision timer fires on schedule.
            # Uses Event.wait() so SIGTERM is never delayed by this sleep.
            self._shutdown_event.wait(timeout=_TRAIN_INTERVAL)

    def _train_step(self, batch: Tuple) -> Tuple[float, float]:
        """One Bellman gradient update (DQN).

        Bellman target: Q_target = r + γ * max_a' Q_target(s') * (1 - done)
        Loss:           MSE(Q_policy(s, a), Q_target)

        _net_lock guards policy_net against concurrent checkpoint reads.
        """
        obs, pos, actions, rewards, next_obs, next_pos, dones = batch

        # Buffer stores HWC float32; network expects CHW
        obs_t    = torch.FloatTensor(obs).permute(0, 3, 1, 2).to(self._device)
        next_t   = torch.FloatTensor(next_obs).permute(0, 3, 1, 2).to(self._device)
        actions_t = torch.LongTensor(actions).to(self._device)
        rewards_t = torch.FloatTensor(rewards).to(self._device)
        dones_t   = torch.BoolTensor(dones).to(self._device)

        # Target Q-values — no grad, no lock needed
        with torch.no_grad():
            next_q     = self._target_net(next_t)   # (B, A)
            max_next_q = next_q.max(1)[0]            # (B,)

        target_q = rewards_t + self._gamma * max_next_q * ~dones_t

        # Policy forward + backward + step under lock
        with self._net_lock:
            current_q     = self._policy_net(obs_t)               # (B, A)
            current_q_sel = current_q.gather(
                1, actions_t.unsqueeze(1)
            ).squeeze(1)                                           # (B,)

            loss = F.mse_loss(current_q_sel, target_q)

            self._optimizer.zero_grad()
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self._policy_net.parameters(), self._grad_clip
                ).item()
            )
            self._optimizer.step()

        return float(loss.item()), grad_norm

    def _decay_epsilon(self) -> None:
        """Linear epsilon decay — article Table 1: ε -= 9e-7 per training step."""
        if self._epsilon > self._eps_min:
            self._epsilon = max(
                self._epsilon - self._eps_decay, self._eps_min
            )

    def _maybe_update_target(self) -> None:
        """Hard-copy policy → target every TARGET_UPDATE_FREQ steps."""
        if self._step_count % self._target_update_freq == 0:
            self._target_net.load_state_dict(self._policy_net.state_dict())
            self.get_logger().info(
                f'{self._tag} Target net updated at step {self._step_count}'
            )

    def _log_train_step(self, loss: float, grad_norm: float) -> None:
        if self._tb is None or self._step_count % 100 != 0:
            return
        s = self._step_count
        self._tb.add_scalar('Train/Loss',     loss,      s)
        self._tb.add_scalar('Train/GradNorm', grad_norm, s)
        self._tb.add_scalar('Train/Epsilon',  self._epsilon, s)

    # ------------------------------------------------------------------ #
    #  WEIGHT BROADCAST                                                    #
    # ------------------------------------------------------------------ #

    def _broadcast_weights(self) -> None:
        """Serialize policy_net + epsilon → zlib compress → JSON → publish.

        Payload: JSON {"eps": float, "w": base64-zlib-state_dict}
        The agent unpacks epsilon to stay in sync without an extra topic.
        """
        try:
            buf = BytesIO()
            with self._net_lock:
                torch.save(self._policy_net.state_dict(), buf)
            compressed = zlib.compress(buf.getvalue(), level=1)
            encoded    = base64.b64encode(compressed).decode('ascii')
            msg      = String()
            msg.data = json.dumps({'eps': self._epsilon, 'w': encoded},
                                  separators=(',', ':'))
            self._weight_pub.publish(msg)
        except Exception as exc:
            self.get_logger().warning(
                f'{self._tag} [WARN] _broadcast_weights: {exc}'
            )

    def _publish_epsilon(self) -> None:
        msg      = Float32()
        msg.data = float(self._epsilon)
        self._eps_pub.publish(msg)

    # ------------------------------------------------------------------ #
    #  CHECKPOINT                                                          #
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self) -> None:
        """Atomically save policy_net, target_net, optimizer, and counters.

        Also writes epsilon.json so the agent can restore its epsilon on startup
        without waiting for a weight broadcast.
        """
        path = os.path.join(self._model_dir, 'latest.pt')
        tmp  = path + '.tmp'
        with self._net_lock:
            payload = {
                'policy_net_state_dict': self._policy_net.state_dict(),
                'target_net_state_dict': self._target_net.state_dict(),
                'optimizer_state_dict':  self._optimizer.state_dict(),
                'epsilon':       self._epsilon,
                'step_count':    self._step_count,
                'episode_count': self._ep_count,
            }
        torch.save(payload, tmp)
        os.replace(tmp, path)

        # Epsilon+episode file — read by the agent on startup for sync
        eps_path = os.path.join(self._model_dir, 'epsilon.json')
        eps_tmp  = eps_path + '.tmp'
        with open(eps_tmp, 'w') as f:
            json.dump({
                'epsilon': self._epsilon,
                'step':    self._step_count,
                'episode': self._ep_count,
            }, f)
        os.replace(eps_tmp, eps_path)

        self.get_logger().info(
            f'{self._tag} Checkpoint saved — '
            f'ep={self._ep_count} step={self._step_count} '
            f'eps={self._epsilon:.3f}'
        )

    def _load_checkpoint(self) -> None:
        path = os.path.join(self._model_dir, 'latest.pt')
        if not os.path.exists(path):
            self.get_logger().info(f'{self._tag} No checkpoint — starting fresh')
            return
        try:
            ckpt = torch.load(
                path, map_location=self._device, weights_only=True
            )
            self._policy_net.load_state_dict(ckpt['policy_net_state_dict'])
            self._target_net.load_state_dict(ckpt['target_net_state_dict'])
            self._optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            # Reduce epsilon after loading to exploit learned policy more
            self._epsilon  = max(
                self._eps_min, float(ckpt.get('epsilon', 1.0)) * 0.7
            )
            self._step_count = int(ckpt.get('step_count', 0))
            self._ep_count   = int(ckpt.get('episode_count', 0))
            self.get_logger().info(
                f'{self._tag} Checkpoint loaded — '
                f'ep={self._ep_count} step={self._step_count} '
                f'eps={self._epsilon:.3f}'
            )
        except Exception as exc:
            self.get_logger().error(
                f'{self._tag} [ERROR] _load_checkpoint: {exc}'
            )

    def _periodic_checkpoint(self) -> None:
        """Spawn background saves, but only if the previous save has finished.

        Without the try-locks, a slow torch.save() can still be running when
        the next 60-s timer fires.  Both threads would write to the same .tmp
        file, producing a corrupt (truncated or interleaved) gzip archive on
        the next restart.
        """
        if self._step_count == 0:
            return
        if self._checkpoint_saving.acquire(blocking=False):
            threading.Thread(
                target=self._checkpoint_save_thread, daemon=True
            ).start()
        if self._buffer_saving.acquire(blocking=False):
            threading.Thread(
                target=self._buffer_save_thread, daemon=True
            ).start()

    def _checkpoint_save_thread(self) -> None:
        try:
            self._save_checkpoint()
        finally:
            self._checkpoint_saving.release()

    def _buffer_save_thread(self) -> None:
        try:
            self._buffer.save(self._buffer_path)
        finally:
            self._buffer_saving.release()

    # ------------------------------------------------------------------ #
    #  WATCHDOG                                                            #
    # ------------------------------------------------------------------ #

    def _watchdog_check(self) -> None:
        elapsed = time.monotonic() - self._last_exp_time
        if elapsed > self._watchdog_timeout:
            self.get_logger().warning(
                f'{self._tag} [WARN] No experience for {elapsed:.0f}s — '
                'agent may be stalled'
            )

    # ------------------------------------------------------------------ #
    #  LIFECYCLE                                                           #
    # ------------------------------------------------------------------ #

    def _sigterm_handler(self, signum, frame) -> None:
        sig_name = 'SIGTERM' if signum != signal.SIGINT else 'SIGINT'
        self.get_logger().info(
            f'{self._tag} {sig_name} — saving checkpoint + buffer'
        )
        self._shutdown_event.set()
        # Blocking acquire: wait until any in-progress background save finishes.
        # This prevents a race where the background thread and the shutdown save
        # both write to the same .tmp file simultaneously.
        self._checkpoint_saving.acquire(blocking=True)
        self._buffer_saving.acquire(blocking=True)
        try:
            self._save_checkpoint()
            self._buffer.save(self._buffer_path)
        finally:
            self._checkpoint_saving.release()
            self._buffer_saving.release()
        self.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    def destroy_node(self) -> None:
        self._shutdown_event.set()
        if self._tb is not None:
            try:
                self._tb.flush()
                self._tb.close()
            except Exception:
                pass
        try:
            self._csv_fh.close()
        except Exception:
            pass
        super().destroy_node()


def main(argv: Optional[list] = None) -> None:
    rclpy.init(args=argv)
    node = DQNTrainerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
