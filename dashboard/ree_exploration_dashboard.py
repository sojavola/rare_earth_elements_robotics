import streamlit as st
import sys
import os

# Ajouter le chemin pour importer les utilitaires
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from data_loader import DataLoader
from visualization import VisualizationEngine
from ros2_connector import ROS2Connector

class REEExplorationDashboard:
    def __init__(self):
        self.setup_page()
        self.data_loader = DataLoader()
        self.visualization = VisualizationEngine()
        self.ros2_connector = ROS2Connector()
        
    def setup_page(self):
        """Configure la page Streamlit"""
        st.set_page_config(
            page_title="REE Exploration Mission Control",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Charger le CSS personnalisé
        self.load_custom_css()
    
    def load_custom_css(self):
        """Charge le CSS personnalisé"""
        css_file = os.path.join(os.path.dirname(__file__), 'assets', 'css', 'custom.css')
        if os.path.exists(css_file):
            with open(css_file) as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        else:
            # CSS par défaut
            st.markdown("""
            <style>
            .main-header {
                font-size: 3rem;
                color: #1f77b4;
                text-align: center;
                margin-bottom: 2rem;
            }
            .metric-card {
                background-color: #f0f2f6;
                padding: 1rem;
                border-radius: 10px;
                border-left: 5px solid #1f77b4;
            }
            </style>
            """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Affiche la barre latérale"""
        with st.sidebar:
            st.image("assets/images/logo.png", width=200)
            st.title("Mission Control")
            
            # Sélecteur de vue
            view_option = st.selectbox(
                "Vue Principale",
                ["Tableau de Bord", "Contrôle Robots", "Analyse Minérale", "Insights IA"]
            )
            
            # Statut de connexion ROS2
            st.subheader("Connexion Système")
            if self.ros2_connector.is_connected():
                st.success("✅ ROS2 Connecté")
            else:
                st.error("❌ ROS2 Déconnecté")
            
            # Contrôles rapides
            st.subheader("Contrôles Rapides")
            if st.button("🔄 Actualiser Données"):
                st.rerun()
                
            if st.button("📊 Générer Rapport"):
                self.generate_report()
    
    def render_main_dashboard(self):
        """Affiche le tableau de bord principal"""
        # En-tête
        st.markdown('<h1 class="main-header">🚀 Mission Control - REE Exploration</h1>', unsafe_allow_html=True)
        
        # Métriques en temps réel
        self.render_realtime_metrics()
        
        # Layout principal
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.render_mineral_map()
            self.render_exploration_progress()
            
        with col2:
            self.render_robot_status()
            self.render_ai_insights()
        
        # Données détaillées
        self.render_detailed_analysis()
    
    def render_realtime_metrics(self):
        """Affiche les métriques en temps réel"""
        st.subheader("📊 Métriques en Temps Réel")
        
        # Récupérer les données ROS2
        mission_data = self.ros2_connector.get_mission_data()
        
        cols = st.columns(5)
        metrics = [
            ("Minéraux Découverts", mission_data.get('minerals_discovered', 0), "+2"),
            ("Zone Explorée", f"{mission_data.get('area_explored', 0)}%", "+5%"),
            ("Échantillons HV", mission_data.get('high_value_samples', 0), "+1"),
            ("Robots Actifs", mission_data.get('active_robots', 0), "0"),
            ("Score Scientifique", f"{mission_data.get('science_score', 0):.0f}", "+25")
        ]
        
        for col, (label, value, delta) in zip(cols, metrics):
            with col:
                st.metric(label=label, value=value, delta=delta)
    
    def render_mineral_map(self):
        """Affiche la carte des minéraux"""
        st.subheader("🗺️ Carte des Concentrations Minérales")
        
        mineral_data = self.ros2_connector.get_mineral_data()
        fig = self.visualization.create_mineral_heatmap(mineral_data)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_exploration_progress(self):
        """Affiche la progression de l'exploration"""
        st.subheader("🧭 Progression de l'Exploration")
        
        progress_data = self.data_loader.get_exploration_progress()
        fig = self.visualization.create_progress_chart(progress_data)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_robot_status(self):
        """Affiche le statut des robots"""
        st.subheader("🤖 Statut des Robots")
        
        robot_data = self.ros2_connector.get_robot_data()
        
        for robot_id, status in robot_data.items():
            with st.expander(f"Robot {robot_id}", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Batterie", f"{status.get('battery', 0)}%")
                    st.metric("État", status.get('status', 'Inconnu'))
                with col2:
                    st.metric("Échantillons", status.get('samples', 0))
                    st.metric("Distance", f"{status.get('distance', 0)}m")
    
    def render_ai_insights(self):
        """Affiche les insights de l'IA générative"""
        st.subheader("🧠 Insights IA Générative")
        
        ai_analysis = self.ros2_connector.get_ai_analysis()
        if ai_analysis:
            st.info(ai_analysis[:500] + "..." if len(ai_analysis) > 500 else ai_analysis)
        else:
            st.warning("En attente des analyses IA...")
    
    def render_detailed_analysis(self):
        """Affiche les analyses détaillées"""
        with st.expander("📈 Analyses Détaillées", expanded=False):
            tab1, tab2, tab3 = st.tabs(["Données Brutes", "Analyses Spectrales", "Export"])
            
            with tab1:
                self.render_raw_data()
            
            with tab2:
                self.render_spectral_analysis()
            
            with tab3:
                self.render_export_section()
    
    def render_raw_data(self):
        """Affiche les données brutes"""
        mission_data = self.ros2_connector.get_mission_data()
        st.dataframe(self.data_loader.format_mission_data(mission_data))
    
    def render_spectral_analysis(self):
        """Affiche l'analyse spectrale"""
        spectral_data = self.ros2_connector.get_spectral_data()
        if spectral_data:
            fig = self.visualization.create_spectral_plot(spectral_data)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_export_section(self):
        """Affiche la section d'export"""
        mission_data = self.ros2_connector.get_mission_data()
        
        st.download_button(
            label="📥 Exporter Rapport JSON",
            data=self.data_loader.export_to_json(mission_data),
            file_name=f"mission_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        if st.button("📊 Générer Rapport PDF"):
            self.generate_pdf_report()
    
    def generate_report(self):
        """Génère un rapport complet"""
        with st.spinner("Génération du rapport..."):
            # Simulation de génération de rapport
            time.sleep(2)
            st.success("Rapport généré avec succès!")
    
    def generate_pdf_report(self):
        """Génère un rapport PDF"""
        st.info("Fonctionnalité PDF en cours de développement...")
    
    def run(self):
        """Lance le dashboard"""
        self.render_sidebar()
        
        # Navigation entre pages
        view_option = st.session_state.get('view_option', 'Tableau de Bord')
        
        if view_option == "Tableau de Bord":
            self.render_main_dashboard()
        elif view_option == "Contrôle Robots":
            self.render_robot_control()
        elif view_option == "Analyse Minérale":
            self.render_mineral_analysis()
        elif view_option == "Insights IA":
            self.render_ai_insights_page()

def main():
    dashboard = REEExplorationDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()