import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

class SpectralDataGenerator:
    def __init__(self, num_wavelengths=100, wavelength_range=(400, 2500)):
        self.num_wavelengths = num_wavelengths
        self.wavelength_range = wavelength_range
        self.wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], num_wavelengths)
        
        # Signatures spectrales de référence basées sur USGS
        self.reference_signatures = {
            'REE_Oxides': self._create_ree_oxide_signature(),
            'REE_Silicates': self._create_ree_silicate_signature(),
            'REE_Phosphates': self._create_ree_phosphate_signature(),
            'REE_Carbonates': self._create_ree_carbonate_signature()
        }
    
    def _create_ree_oxide_signature(self):
        """Crée une signature spectrale réaliste pour les oxydes de REE"""
        base = np.zeros(self.num_wavelengths)
        
        # Pic d'absorption caractéristique des oxydes
        base += self._gaussian_peak(650, 50, 0.8)    # Pic rouge
        base += self._gaussian_peak(1450, 80, 0.6)   # Pic infrarouge
        base += self._gaussian_peak(2200, 100, 0.7)  # Pic infrarouge moyen
        
        # Bruit de fond avec tendance
        background = np.linspace(0.1, 0.3, self.num_wavelengths)
        base += background
        
        return savgol_filter(base, 11, 3)  # Lissage
    
    def _create_ree_silicate_signature(self):
        """Crée une signature spectrale réaliste pour les silicates de REE"""
        base = np.zeros(self.num_wavelengths)
        
        # Signature des silicates
        base += self._gaussian_peak(550, 60, 0.7)    # Pic vert
        base += self._gaussian_peak(1250, 90, 0.5)   # Pic infrarouge
        base += self._gaussian_peak(1900, 120, 0.6)  # Pic infrarouge
        
        background = np.linspace(0.15, 0.25, self.num_wavelengths)
        base += background
        
        return savgol_filter(base, 11, 3)
    
    def _create_ree_phosphate_signature(self):
        """Crée une signature spectrale réaliste pour les phosphates de REE"""
        base = np.zeros(self.num_wavelengths)
        
        # Signature des phosphates
        base += self._gaussian_peak(480, 40, 0.9)    # Pic bleu-vert
        base += self._gaussian_peak(1050, 70, 0.6)   # Pic infrarouge
        base += self._gaussian_peak(2350, 110, 0.5)  # Pic infrarouge
        
        background = np.linspace(0.2, 0.35, self.num_wavelengths)
        base += background
        
        return savgol_filter(base, 11, 3)
    
    def _create_ree_carbonate_signature(self):
        """Crée une signature spectrale réaliste pour les carbonates de REE"""
        base = np.zeros(self.num_wavelengths)
        
        # Signature des carbonates
        base += self._gaussian_peak(600, 55, 0.6)    # Pic orange
        base += self._gaussian_peak(1350, 85, 0.7)   # Pic infrarouge
        base += self._gaussian_peak(2100, 95, 0.8)   # Pic infrarouge
        
        background = np.linspace(0.12, 0.28, self.num_wavelengths)
        base += background
        
        return savgol_filter(base, 11, 3)
    
    def _gaussian_peak(self, center, width, amplitude):
        """Crée un pic gaussien"""
        return amplitude * np.exp(-0.5 * ((self.wavelengths - center) / width) ** 2)
    
    def generate_spectral_dataset(self, num_samples_per_class=1000, noise_level=0.05):
        """Génère un dataset spectral synthétique complet"""
        spectra = []
        labels = []
        mineral_classes = list(self.reference_signatures.keys())
        
        for class_idx, mineral_class in enumerate(mineral_classes):
            base_signature = self.reference_signatures[mineral_class]
            
            for _ in range(num_samples_per_class):
                # Variation aléatoire de la signature de base
                variation = np.random.normal(1.0, 0.1, self.num_wavelengths)
                noisy_spectrum = base_signature * variation
                
                # Ajout de bruit
                noise = np.random.normal(0, noise_level, self.num_wavelengths)
                final_spectrum = noisy_spectrum + noise
                
                # Assurer des valeurs positives
                final_spectrum = np.clip(final_spectrum, 0, 1)
                
                spectra.append(final_spectrum)
                labels.append(class_idx)
        
        return np.array(spectra), np.array(labels), mineral_classes
    
    def add_realistic_noise(self, spectrum, snr_db=20):
        """Ajoute un bruit réaliste avec un rapport signal/bruit spécifié"""
        signal_power = np.mean(spectrum ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(spectrum))
        return spectrum + noise
    
    def save_dataset(self, spectra, labels, filepath):
        """Sauvegarde le dataset généré"""
        dataset_df = pd.DataFrame(spectra, columns=[f'wl_{i}' for i in range(self.num_wavelengths)])
        dataset_df['mineral_class'] = labels
        dataset_df.to_csv(filepath, index=False)
        print(f"Dataset sauvegardé: {filepath}")

# Exemple d'utilisation
if __name__ == "__main__":
    generator = SpectralDataGenerator()
    spectra, labels, classes = generator.generate_spectral_dataset()
    
    print(f"Dataset généré: {len(spectra)} échantillons")
    print(f"Classes: {classes}")
    print(f"Shape des spectres: {spectra.shape}")