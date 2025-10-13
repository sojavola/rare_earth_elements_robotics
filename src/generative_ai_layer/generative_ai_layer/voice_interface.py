import speech_recognition as sr
import pyttsx3
import threading
import queue
import time

class VoiceCommandInterface:
    def __init__(self, science_ai_system):
        self.science_ai = science_ai_system
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        
        # File d'attente pour les commandes
        self.command_queue = queue.Queue()
        self.is_listening = False
        
        # Configuration de la synthèse vocale
        self._setup_tts()
        
        # Commandes vocales prédéfinies
        self.voice_commands = {
            'analyse': self._handle_analysis_command,
            'rapport': self._handle_report_command,
            'conseils': self._handle_advice_command,
            'statut': self._handle_status_command,
            'explorer': self._handle_exploration_command
        }
    
    def _setup_tts(self):
        """Configure la synthèse vocale"""
        voices = self.tts_engine.getProperty('voices')
        if voices:
            self.tts_engine.setProperty('voice', voices[0].id)  # Première voix disponible
        self.tts_engine.setProperty('rate', 150)  # Vitesse de parole
    
    def start_listening(self):
        """Démarre l'écoute des commandes vocales"""
        self.is_listening = True
        listening_thread = threading.Thread(target=self._listen_loop)
        listening_thread.daemon = True
        listening_thread.start()
        print("🎤 Interface vocale activée - En écoute...")
    
    def stop_listening(self):
        """Arrête l'écoute des commandes vocales"""
        self.is_listening = False
    
    def _listen_loop(self):
        """Boucle d'écoute principale"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            
            while self.is_listening:
                try:
                    print("🎤 Écoute...")
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    
                    # Reconnaissance vocale
                    command = self.recognizer.recognize_google(audio, language='fr-FR')
                    print(f"🗣️ Commande détectée: {command}")
                    
                    # Traitement de la commande
                    self._process_voice_command(command)
                    
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    self.speak("Désolé, je n'ai pas compris la commande.")
                except Exception as e:
                    print(f"Erreur de reconnaissance vocale: {e}")
    
    def _process_voice_command(self, command):
        """Traite une commande vocale"""
        command_lower = command.lower()
        
        # Recherche de mots-clés dans la commande
        for keyword, handler in self.voice_commands.items():
            if keyword in command_lower:
                handler(command)
                return
        
        # Commande non reconnue
        self.speak("Commande non reconnue. Essayez: analyse, rapport, conseils, statut, ou explorer.")
    
    def _handle_analysis_command(self, command):
        """Gère les commandes d'analyse"""
        self.speak("Exécution de l'analyse scientifique...")
        # Ici, vous intégreriez les données actuelles de la mission
        analysis = self.science_ai.analyze_mission_data({}, {}, {})
        self.speak(analysis[:500])  # Limite pour la démo
    
    def _handle_report_command(self, command):
        """Gère les commandes de rapport"""
        self.speak("Génération du rapport de mission...")
        report = self.science_ai.generate_mission_report({}, {}, {})
        self.speak("Rapport généré avec succès.")
    
    def _handle_advice_command(self, command):
        """Gère les commandes de conseils"""
        self.speak("Analyse de la situation pour conseils...")
        advice = self.science_ai.get_real_time_advice({}, {}, {})
        self.speak(advice[:500])
    
    def _handle_status_command(self, command):
        """Gère les commandes de statut"""
        status_message = """
        Statut de la mission: Exploration en cours.
        Robots actifs: 4.
        Minéraux détectés: Oxydes de terres rares, Silicates.
        Zones explorées: 65 pourcent.
        """
        self.speak(status_message)
    
    def _handle_exploration_command(self, command):
        """Gère les commandes d'exploration"""
        if "crater" in command.lower():
            self.speak("Direction vers la zone du cratère pour exploration.")
        elif "nord" in command.lower():
            self.speak("Exploration de la région nord initiée.")
        else:
            self.speak("Zone d'exploration spécifiée. Déploiement des robots.")
    
    def speak(self, text):
        """Fait parler le système"""
        print(f"🤖 AI: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def send_text_command(self, text_command):
        """Permet d'envoyer des commandes texte"""
        print(f"📝 Commande texte: {text_command}")
        self._process_voice_command(text_command)

# Exemple d'utilisation
if __name__ == "__main__":
    ai_system = ScienceAISystem()
    voice_interface = VoiceCommandInterface(ai_system)
    
    # Démarrer l'écoute
    voice_interface.start_listening()
    
    # Exemple de commande texte
    voice_interface.send_text_command("Donne-moi une analyse de la mission")
    
    # Garder le programme actif
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        voice_interface.stop_listening()
        print("Interface vocale arrêtée.")
        