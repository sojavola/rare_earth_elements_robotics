#!/usr/bin/env python3


import os
import sys
import subprocess
import webbrowser
import threading
import time

def check_dependencies():
    """Vérifie les dépendances nécessaires"""
    try:
        import streamlit
        import plotly
        import pandas
        print("✅ Dépendances Streamlit vérifiées")
        return True
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        return False

def install_dependencies():
    """Installe les dépendances manquantes"""
    print("📦 Installation des dépendances...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "streamlit", "plotly", "pandas", "numpy", "scipy", "scikit-learn"
    ])

def start_dashboard():
    """Lance le dashboard Streamlit"""
    print("🚀 Lancement du Dashboard REE Exploration...")
    
    # Changer vers le répertoire du dashboard
    dashboard_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dashboard_dir)
    
    # Lancer Streamlit
    subprocess.run([
        "streamlit", "run", "ree_exploration_dashboard.py",
        "--server.port=8501", "--server.address=0.0.0.0", "--theme.base=light"
    ])

def open_browser():
    """Ouvre le navigateur automatiquement"""
    time.sleep(3)  # Attendre que Streamlit démarre
    webbrowser.open("http://localhost:8501")

def main():
    print("🎯 REE Exploration Dashboard Launcher")
    print("=" * 50)
    
    # Vérifier les dépendances
    if not check_dependencies():
        print("Installation des dépendances nécessaires...")
        install_dependencies()
    
    # Ouvrir le navigateur dans un thread séparé
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Lancer le dashboard
    try:
        start_dashboard()
    except KeyboardInterrupt:
        print("\n🛑 Dashboard arrêté par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur lors du lancement: {e}")

if __name__ == "__main__":
    main()


