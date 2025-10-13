"""
Patch de compatibilité pour Pydantic v2 avec LangChain
"""
import warnings
import os

# Désactiver les warnings temporairement
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Forcer Pydantic v2
os.environ['PYDANTIC_VERSION'] = '2'

try:
    from pydantic import BaseModel, Field, ConfigDict
    from pydantic.v1 import BaseModel as BaseModelV1
    from pydantic.v1 import Field as FieldV1
    
    # Configuration pour la compatibilité
    PYDANTIC_V2 = True
    
    def get_pydantic_config():
        """Retourne la configuration Pydantic compatible"""
        return ConfigDict(extra='ignore', arbitrary_types_allowed=True)
        
except ImportError:
    PYDANTIC_V2 = False
    BaseModel = None
    Field = None
    
    def get_pydantic_config():
        return {}

# Patch pour LangChain
def apply_langchain_patches():
    """Applique les patches de compatibilité pour LangChain"""
    try:
        import langchain
        import langchain_core
        
        # Forcer l'utilisation de Pydantic v2 dans LangChain
        if hasattr(langchain_core, 'pydantic_v1'):
            # Monkey patch pour rediriger vers pydantic v2
            import pydantic
            langchain_core.pydantic_v1 = pydantic
            
        print("✅ Patches de compatibilité Pydantic v2 appliqués")
        
    except Exception as e:
        print(f"⚠️ Impossible d'appliquer les patches: {e}")

# Appliquer les patches au démarrage
apply_langchain_patches()