# data_logger.py
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from datetime import datetime
import json
from pathlib import Path
from threading import Lock

class EnhancedDataLogger:
    """
    Logger avancé DQN
    - CSV append-safe
    - Thread-safe (ROS2)
    - Plots offline only
    """

    def __init__(self, robot_id, log_dir="logs", enable_plots=True):
        self.robot_id = robot_id
        self.log_dir = Path(log_dir)
        self.data_dir = self.log_dir / f"robot_{robot_id}"
        self.enable_plots = enable_plots

        self.lock = Lock()

        # DataFrames mémoire (pour analyse uniquement)
        self.episodes_df = pd.DataFrame()
        self.steps_df = pd.DataFrame()
        self.minerals_df = pd.DataFrame()
        self.training_df = pd.DataFrame()

        self._create_directory_structure()

        if self.enable_plots:
            self._setup_plot_style()

        print(f"📊 Logger prêt pour Robot {robot_id}")
        print(f"📁 {self.data_dir}")

    # ==========================================================
    # INIT
    # ==========================================================

    def _create_directory_structure(self):
        for d in ["csv", "plots/seaborn", "plots/plotly", "reports"]:
            (self.data_dir / d).mkdir(parents=True, exist_ok=True)

    def _setup_plot_style(self):
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["axes.grid"] = True

    # ==========================================================
    # CSV SAFE APPEND
    # ==========================================================

    def _append_csv(self, filepath, row_dict):
        """Append une seule ligne CSV (safe)"""
        with self.lock:
            file_exists = filepath.exists()
            with open(filepath, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row_dict.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row_dict)

    # ==========================================================
    # LOGGING API (UTILISÉ EN RUNTIME)
    # ==========================================================

    def log_episode(self, data: dict):
        csv_path = self.data_dir / "csv" / "episodes.csv"
        self._append_csv(csv_path, data)

        if self.enable_plots:
            self.episodes_df = pd.concat(
                [self.episodes_df, pd.DataFrame([data])],
                ignore_index=True
            )

            if len(self.episodes_df) % 10 == 0:
                self._generate_episode_plots()

    def log_step(self, data: dict):
        csv_path = self.data_dir / "csv" / "steps.csv"
        self._append_csv(csv_path, data)

    def log_mineral(self, data: dict):
        csv_path = self.data_dir / "csv" / "minerals.csv"
        self._append_csv(csv_path, data)

        if self.enable_plots:
            self.minerals_df = pd.concat(
                [self.minerals_df, pd.DataFrame([data])],
                ignore_index=True
            )

            if len(self.minerals_df) % 5 == 0:
                self._generate_mineral_plot()

    def log_training(self, data: dict):
        csv_path = self.data_dir / "csv" / "training.csv"
        self._append_csv(csv_path, data)

        if self.enable_plots:
            self.training_df = pd.concat(
                [self.training_df, pd.DataFrame([data])],
                ignore_index=True
            )

    # ==========================================================
    # ANALYSE & PLOTS (OFFLINE / NON CRITIQUE)
    # ==========================================================

    def _generate_episode_plots(self):
        if len(self.episodes_df) < 2:
            return

        plt.figure(figsize=(14, 10))

        # Récompense
        plt.subplot(2, 2, 1)
        sns.lineplot(
            data=self.episodes_df,
            x="episode",
            y="total_reward",
            marker="o"
        )
        plt.title(f"Robot {self.robot_id} - Récompense")

        # Minéraux
        if "minerals_collected" in self.episodes_df.columns:
            plt.subplot(2, 2, 2)
            sns.lineplot(
                data=self.episodes_df,
                x="episode",
                y="minerals_collected"
            )
            plt.title("Minéraux collectés")

        # Epsilon
        if "epsilon" in self.episodes_df.columns:
            plt.subplot(2, 2, 3)
            sns.lineplot(
                data=self.episodes_df,
                x="episode",
                y="epsilon"
            )
            plt.title("Évolution epsilon")

        plt.tight_layout()
        plt.savefig(
            self.data_dir / "plots/seaborn" / "episode_summary.png",
            dpi=150
        )
        plt.close()

    def _generate_mineral_plot(self):
        if len(self.minerals_df) == 0:
            return

        plt.figure(figsize=(10, 6))
        sns.countplot(
            data=self.minerals_df,
            x="mineral_type"
        )
        plt.title("Distribution des minéraux")
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(
            self.data_dir / "plots/seaborn" / "minerals.png",
            dpi=150
        )
        plt.close()

    # ==========================================================
    # RAPPORT FINAL
    # ==========================================================

    def generate_report(self):
        if len(self.episodes_df) == 0:
            print("❌ Aucune donnée")
            return

        stats = {
            "robot_id": self.robot_id,
            "episodes": len(self.episodes_df),
            "avg_reward": float(self.episodes_df["total_reward"].mean()),
            "max_reward": float(self.episodes_df["total_reward"].max()),
            "min_reward": float(self.episodes_df["total_reward"].min()),
            "final_epsilon": float(self.episodes_df["epsilon"].iloc[-1]),
            "generated_at": datetime.now().isoformat()
        }

        with open(self.data_dir / "reports" / "stats.json", "w") as f:
            json.dump(stats, f, indent=4)

        print("📄 Rapport généré")
