#!/usr/bin/env python3

import subprocess
import sys
import time
import os
import threading
import webbrowser

def main():
    print("🚀 Starting Complete REE Exploration System...")
    print("=" * 60)
    
    # Construction du workspace ROS2
    print("📦 Building ROS2 workspace...")
    build_process = subprocess.run([
        "colcon", "build", "--packages-select",
        "ree_exploration_server", "ree_exploration_agent", 
        "extraterrestrial_world", "spectral_classifier", "generative_ai_layer"
    ], capture_output=True, text=True)
    
    if build_process.returncode != 0:
        print("❌ Build failed!")
        print(build_process.stderr)
        return
    
    print("✅ Build successful!")
    
    # Sourcer l'environnement
    os.system("source install/setup.bash")
    
    processes = []
    
    def run_command(command, description):
        """Exécute une commande dans un processus séparé"""
        print(f"🔄 Starting {description}...")
        process = subprocess.Popen(command, shell=True)
        processes.append((process, description))
        return process
    
    # Lancer Gazebo avec le monde lunaire
    gazebo_process = run_command(
        "ros2 launch extraterrestrial_world moon_base_launch.py",
        "Gazebo Lunar Simulation"
    )
    time.sleep(15)  # Attendre que Gazebo démarre
    
    # Lancer le serveur d'exploration
    server_process = run_command(
        "ros2 run ree_exploration_server server_node",
        "REE Exploration Server"
    )
    time.sleep(5)
    
    # Lancer le coordinateur multi-agents
    coordinator_process = run_command(
        "ros2 run ree_exploration_agent multi_agent_coordinator",
        "Multi-Agent Coordinator"
    )
    time.sleep(3)
    
    # Lancer les agents d'exploration
    for i in range(4):
        agent_process = run_command(
            f"ros2 run ree_exploration_agent agent_node {i}",
            f"Exploration Agent {i}"
        )
        time.sleep(2)
    
    # Lancer le classificateur spectral
    spectral_process = run_command(
        "ros2 run spectral_classifier spectral_classifier_node",
        "Spectral Classifier"
    )
    time.sleep(3)
    
    # Lancer le système IA générative
    ai_process = run_command(
        "ros2 run generative_ai_layer science_ai_system",
        "Generative AI System"
    )
    
    # Lancer le dashboard Streamlit dans un thread séparé
    def start_dashboard():
        print("🌐 Starting Streamlit Dashboard...")
        os.chdir("dashboard")
        subprocess.run(["streamlit", "run", "ree_exploration_dashboard.py", "--server.port=8501"])
    
    dashboard_thread = threading.Thread(target=start_dashboard)
    dashboard_thread.daemon = True
    dashboard_thread.start()
    
    time.sleep(5)
    
    # Ouvrir le dashboard dans le navigateur
    print("🌍 Opening dashboard in browser...")
    webbrowser.open("http://localhost:8501")
    
    print("\n" + "=" * 60)
    print("✅ REE EXPLORATION SYSTEM FULLY OPERATIONAL!")
    print("🤖 Active Components:")
    print("   - 🌕 Gazebo Lunar Simulation")
    print("   - 🗺️ REE Exploration Server") 
    print("   - 🤖 Multi-Agent Coordinator")
    print("   - 🔬 4 Exploration Rovers")
    print("   - 📊 Spectral Classifier")
    print("   - 🧠 Generative AI System")
    print("   - 🌐 Streamlit Dashboard")
    print("\n📋 Monitoring available at: http://localhost:8501")
    print("Press Ctrl+C to shutdown all systems.")
    
    try:
        # Maintenir le système en fonctionnement
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down REE Exploration System...")
        
        # Arrêter tous les processus
        for process, description in processes:
            print(f"⏹️ Stopping {description}...")
            process.terminate()
            process.wait()
        
        print("✅ All systems shutdown complete.")
        print("🎯 Mission accomplished!")

if __name__ == '__main__':
    main()