import streamlit as st
import pandas as pd
import requests
import pickle
from pathlib import Path
import time
from typing import Dict, List, Optional
import google.generativeai as genai
from anthropic import Anthropic
import plotly.graph_objects as go
import plotly.express as px
import re

# --- CONFIGURATION ---
st.set_page_config(
    page_title="IA & Données Éducatives",
    page_icon="🎓",
    layout="wide"
)

# Initialiser le session state
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False

# Créer dossier cache (seulement en local, pas sur Streamlit Cloud)
try:
    CACHE_DIR = Path("data_cache")
    CACHE_DIR.mkdir(exist_ok=True)
    cache_enabled = True
except:
    cache_enabled = False
    st.warning("⚠️ Cache désactivé (système de fichiers en lecture seule)")

# --- GESTION CLÉS API (SUPPORT GEMINI + CLAUDE) ---

def get_api_key(provider: str) -> Optional[str]:
    """
    Récupère la clé API selon la priorité :
    1. Streamlit Secrets (pour déploiement cloud)
    2. Session State (cache en mémoire)
    3. Input utilisateur
    
    Args:
        provider: 'gemini' ou 'claude'
    """
    key_name = f"{provider.upper()}_API_KEY"
    session_key = f'cached_{provider}_api_key'
    
    # Priorité 1 : Streamlit Secrets (configuration cloud)
    try:
        if key_name in st.secrets:
            api_key = st.secrets[key_name]
            st.session_state[session_key] = api_key
            return api_key
    except:
        pass
    
    # Priorité 2 : Session State (cache mémoire)
    if session_key in st.session_state and st.session_state[session_key]:
        return st.session_state[session_key]
    
    # Priorité 3 : Aucune clé disponible
    return None

def save_api_key_to_session(provider: str, api_key: str):
    """Sauvegarde la clé API en mémoire (session state uniquement)."""
    session_key = f'cached_{provider}_api_key'
    st.session_state[session_key] = api_key

# --- CLASSE API SERVICE ---
class WorldBankAPI:
    """Service API Banque Mondiale - Approche REST directe"""
    
    BASE_URL = "https://api.worldbank.org/v2"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Education-Analytics/1.0'})
    
    def _get_cache_path(self, key: str) -> Path:
        if cache_enabled:
            return CACHE_DIR / f"{key}.pkl"
        return None
    
    def _load_cache(self, key: str) -> Optional[pd.DataFrame]:
        """Charge depuis cache si valide (<24h)"""
        if not cache_enabled:
            return None
            
        cache_file = self._get_cache_path(key)
        if cache_file and cache_file.exists():
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            if age_hours < 24:
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    st.warning(f"Impossible de lire le cache {cache_file}: {e}")
        return None
    
    def _save_cache(self, key: str, data: pd.DataFrame):
        """Sauvegarde en cache"""
        if not cache_enabled:
            return
            
        try:
            cache_path = self._get_cache_path(key)
            if cache_path:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
        except Exception as e:
            st.warning(f"Impossible de sauvegarder le cache {key}: {e}")
    
    def fetch_indicator(self, indicator_code: str, countries: List[str], 
                       start_year: int = 2010, end_year: int = 2024) -> pd.DataFrame:
        """Récupère données pour un indicateur via API REST"""
        
        cache_key = f"{indicator_code}_{'_'.join(countries)}_{start_year}_{end_year}"
        
        # Vérifier cache
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached
        
        # Construire requête API
        country_codes = ';'.join(countries)
        url = f"{self.BASE_URL}/country/{country_codes}/indicator/{indicator_code}"
        params = {
            'format': 'json',
            'date': f"{start_year}:{end_year}",
            'per_page': 5000
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Valider structure
            if not isinstance(data, list) or len(data) < 2 or not data[1]:
                return pd.DataFrame()
            
            # Parser les enregistrements
            records = []
            for record in data[1]:
                if not record or record.get('value') is None:
                    continue
                
                country_info = record.get('country', {})
                country_code = country_info.get('id', '')
                
                # Correction : l'API renvoie des codes minuscules
                if not country_code or country_code.upper() not in countries:
                    continue
                
                try:
                    year = int(record.get('date', 0))
                    if start_year <= year <= end_year:
                        records.append({
                            'country_code': country_code.upper(),
                            'country_name': country_info.get('value', ''),
                            'year': year,
                            'value': round(float(record.get('value')), 2),
                            'indicator_code': indicator_code
                        })
                except (ValueError, TypeError):
                    continue
            
            if not records:
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            
            # Sauvegarder en cache
            self._save_cache(cache_key, df)
            
            return df
            
        except Exception as e:
            st.error(f"Erreur API pour {indicator_code}: {str(e)}")
            return pd.DataFrame()

# --- DONNÉES DE DÉMONSTRATION ---
DEMO_RESPONSE = """
### 1. Synthèse des Tendances Clés

* **Disparité entre Inscription et Achèvement :** Le Maroc et le Sénégal affichent des taux de scolarisation primaire élevés (proches de 98-100% ces dernières années), mais le Sénégal montre un taux d'achèvement pour les filles significativement plus bas (environ 70%) comparé au Maroc (environ 90%).
* **Performance du Kenya :** Le Kenya se distingue avec un taux d'achèvement pour les filles (environ 95%) presque aligné sur son excellent taux de scolarisation, indiquant une forte rétention scolaire dans le primaire.
* **Investissement vs Résultats :** Le Sénégal consacre une part plus importante de son PIB à l'éducation (environ 5-6%) que le Maroc (4-5%). Cependant, cet investissement supérieur ne se traduit pas encore par un taux d'achèvement féminin équivalent.

### 2. Interprétation et Anomalies

L'anomalie la plus notable est l'écart important au **Sénégal** entre un taux de scolarisation quasi universel et un taux d'achèvement féminin de seulement 70%. Cela suggère des problèmes systémiques de décrochage scolaire spécifiques aux filles après leur inscription.

Le **Maroc** montre une meilleure efficacité de rétention, mais l'écart de 10 points entre l'inscription et l'achèvement justifie une attention. Le **Kenya** sert de référence positive, montrant qu'un faible écart est possible.

### 3. Recommandations Stratégiques

1.  **[Pour le Sénégal] Lancer une Enquête Qualitative Ciblée :** Les données quantitatives montrent *quoi* (décrochage féminin), mais pas *pourquoi*. Il est recommandé de déployer des enquêtes de terrain pour identifier les causes spécifiques du décrochage des filles entre le début et la fin du cycle primaire.

2.  **[Pour le Maroc] Analyser les Bonnes Pratiques de Rétention :** Analyser les politiques de rétention des 10% d'élèves qui décrochent. Se concentrer sur les régions à plus fort décrochage pour y appliquer des mesures de soutien.

3.  **[Général] Audit d'Efficacité des Dépenses (Sénégal) :** Analyser l'allocation des 5-6% du PIB consacrés à l'éducation pour s'assurer qu'une part suffisante est dirigée vers la rétention scolaire des filles.
"""

# --- PAYS AFRICAINS (54 pays) ---
AFRICAN_COUNTRIES = {
    'Afrique du Sud': 'ZA', 'Algérie': 'DZ', 'Angola': 'AO', 'Bénin': 'BJ',
    'Botswana': 'BW', 'Burkina Faso': 'BF', 'Burundi': 'BI', 'Cameroun': 'CM',
    'Cap-Vert': 'CV', 'Comores': 'KM', 'Congo (Rép. Dém.)': 'CD', 'Congo (Rép.)': 'CG',
    'Côte d\'Ivoire': 'CI', 'Djibouti': 'DJ', 'Égypte': 'EG', 'Érythrée': 'ER',
    'Eswatini': 'SZ', 'Éthiopie': 'ET', 'Gabon': 'GA', 'Gambie': 'GM',
    'Ghana': 'GH', 'Guinée': 'GN', 'Guinée équatoriale': 'GQ', 'Guinée-Bissau': 'GW',
    'Kenya': 'KE', 'Lesotho': 'LS', 'Libéria': 'LR', 'Libye': 'LY',
    'Madagascar': 'MG', 'Malawi': 'MW', 'Mali': 'ML', 'Maroc': 'MA',
    'Maurice': 'MU', 'Mauritanie': 'MR', 'Mozambique': 'MZ', 'Namibie': 'NA',
    'Niger': 'NE', 'Nigéria': 'NG', 'Ouganda': 'UG', 'Rwanda': 'RW',
    'Sao Tomé-et-Principe': 'ST', 'Sénégal': 'SN', 'Seychelles': 'SC', 'Sierra Leone': 'SL',
    'Somalie': 'SO', 'Soudan': 'SD', 'Soudan du Sud': 'SS', 'Tanzanie': 'TZ',
    'Tchad': 'TD', 'Togo': 'TG', 'Tunisie': 'TN', 'Zambie': 'ZM',
    'Zimbabwe': 'ZW'
}

# --- INDICATEURS DISPONIBLES ---
INDICATORS = {
    "Dépenses publiques d'éducation (% du PIB)": "SE.XPD.TOTL.GD.ZS",
    "Taux d'achèvement (Primaire, Filles)": "SE.PRM.CMPT.FE.ZS",
    "Taux de scolarisation (Primaire)": "SE.PRM.NENR"
}

# --- FONCTIONS GRAPHIQUES ---
def create_trend_chart(df: pd.DataFrame, column: str, title: str, yaxis_title: str):
    """Graphique de tendances temporelles"""
    fig = go.Figure()
    
    for pays in df['Pays'].unique():
        data_pays = df[df['Pays'] == pays].sort_values('Année')
        fig.add_trace(go.Scatter(
            x=data_pays['Année'],
            y=data_pays[column],
            mode='lines+markers',
            name=pays,
            line=dict(width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Année",
        yaxis_title=yaxis_title,
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_comparison_chart(df: pd.DataFrame, indicator_cols: List[str], year: int):
    """Graphique de comparaison pour une année donnée"""
    data_year = df[df['Année'] == year]
    
    if data_year.empty:
        return None
    
    fig = go.Figure()
    
    for indicator in indicator_cols:
        fig.add_trace(go.Bar(
            name=indicator,
            x=data_year['Pays'],
            y=data_year[indicator],
            text=data_year[indicator].round(1),
            textposition='auto',
        ))
    
    fig.update_layout(
        title=f"Comparaison des Indicateurs en {year}",
        xaxis_title="Pays",
        yaxis_title="Valeur",
        barmode='group',
        template='plotly_white',
        height=500
    )
    
    return fig

# --- FONCTIONS ANALYSE IA ---

def generate_gemini_analysis(data_csv: str, countries: List[str], 
                             indicators: List[str], api_key: str) -> Optional[str]:
    """Génère une analyse IA via Google Gemini"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        prompt = f"""
Tu es un analyste spécialisé en données éducatives pour l'Afrique. Analyse ce jeu de données et produis un rapport structuré.

**Pays analysés :** {', '.join(countries)}
**Indicateurs :** {', '.join(indicators)}

**Données CSV :**
{data_csv}

**Instructions :**
1. Identifie 3-4 tendances ou patterns clés dans les données
2. Signale toute anomalie ou écart significatif
3. Propose 2-3 recommandations concrètes basées sur les données

Utilise le format markdown avec des sections claires. Sois précis et facile à lire.
"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        st.error(f"Erreur Gemini : {str(e)}")
        return None

def generate_claude_analysis(data_csv: str, countries: List[str], 
                             indicators: List[str], api_key: str) -> Optional[str]:
    """Génère une analyse IA via Claude Anthropic"""
    try:
        client = Anthropic(api_key=api_key)
        
        prompt = f"""Tu es un analyste spécialisé en données éducatives pour l'Afrique. Analyse ce jeu de données et produis un rapport structuré.

**Pays analysés :** {', '.join(countries)}
**Indicateurs :** {', '.join(indicators)}

**Données CSV :**
{data_csv}

**Instructions :**
1. Identifie 3-4 tendances ou patterns clés dans les données
2. Signale toute anomalie ou écart significatif
3. Propose 2-3 recommandations concrètes basées sur les données

Utilise le format markdown avec des sections claires. Sois précis et facile à lire."""
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return message.content[0].text
        
    except Exception as e:
        st.error(f"Erreur Claude : {str(e)}")
        return None

def format_ai_analysis(analysis_text: str):
    """Formatte l'analyse IA avec style"""
    lines = analysis_text.split('\n')
    
    for line in lines:
        if line.strip().startswith('###'):
            st.markdown(f"**{line.strip()}**")
        elif line.strip().startswith('*'):
            st.markdown(line)
        elif line.strip().startswith('-'):
            st.markdown(line)
        elif re.match(r'^\d+\.', line.strip()):
            st.markdown(line)
        elif line.strip():
            st.write(line)

# --- INTERFACE PRINCIPALE ---
st.title("🎓 Analyse IA des Données Éducatives Africaines")
st.markdown("*Propulsé par l'API Banque Mondiale & IA (Gemini/Claude)*")

# --- SIDEBAR : CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # --- SÉLECTION DU FOURNISSEUR IA ---
    st.subheader("🤖 Fournisseur d'IA")
    ai_provider = st.radio(
        "Choisissez votre moteur d'analyse :",
        options=["Gemini (Google)", "Claude (Anthropic)", "Mode Démo (Sans API)"],
        index=0,
        help="Gemini est gratuit jusqu'à 15 req/min. Claude offre une analyse plus approfondie."
    )
    
    # Mapper le choix
    if "Gemini" in ai_provider:
        selected_provider = "gemini"
        provider_name = "Google Gemini"
        api_link = "https://makersuite.google.com/app/apikey"
    elif "Claude" in ai_provider:
        selected_provider = "claude"
        provider_name = "Claude (Anthropic)"
        api_link = "https://console.anthropic.com/account/keys"
    else:
        selected_provider = "demo"
        provider_name = "Mode Démo"
        api_link = None
    
    # --- GESTION DE LA CLÉ API ---
    if selected_provider != "demo":
        st.subheader(f"🔑 Clé API {provider_name}")
        
        # Vérifier si une clé existe déjà
        existing_key = get_api_key(selected_provider)
        
        if existing_key:
            st.success(f"✅ Clé API {provider_name} configurée")
            if st.button(f"🔄 Changer la clé {provider_name}"):
                st.session_state[f'cached_{selected_provider}_api_key'] = None
                st.rerun()
            AI_API_KEY = existing_key
        else:
            st.info(f"Configurez votre clé API {provider_name}")
            if api_link:
                st.markdown(f"[Obtenir une clé API]({api_link})")
            
            api_input = st.text_input(
                "Clé API",
                type="password",
                placeholder="sk-ant-..." if selected_provider == "claude" else "AIzaSy...",
                key=f"{selected_provider}_api_input"
            )
            
            if api_input:
                save_api_key_to_session(selected_provider, api_input)
                AI_API_KEY = api_input
                st.success(f"✅ Clé API {provider_name} sauvegardée en mémoire")
                st.rerun()
            else:
                AI_API_KEY = None
        
        # Mode démo si pas de clé
        use_demo_mode = not AI_API_KEY
        if use_demo_mode:
            st.warning(f"⚠️ Aucune clé API {provider_name} : passage en mode démo")
    else:
        use_demo_mode = True
        AI_API_KEY = None
        st.info("ℹ️ Mode démo activé (données pré-calculées)")
    
    st.markdown("---")
    
    # --- SÉLECTION PAYS ---
    st.subheader("🌍 Pays")
    selected_countries = st.multiselect(
        "Choisissez 2-5 pays africains",
        options=sorted(AFRICAN_COUNTRIES.keys()),
        default=['Maroc', 'Sénégal', 'Kenya'],
        max_selections=5
    )
    
    # --- SÉLECTION INDICATEURS ---
    st.subheader("📊 Indicateurs")
    selected_indicators = st.multiselect(
        "Choisissez 1-3 indicateurs",
        options=list(INDICATORS.keys()),
        default=list(INDICATORS.keys())
    )
    
    # --- PÉRIODE ---
    st.subheader("📅 Période")
    year_range = st.slider(
        "Années",
        min_value=2000,
        max_value=2024,
        value=(2010, 2022)
    )
    
    st.markdown("---")
    
    # --- BOUTON DE LANCEMENT ---
    launch_button = st.button(
        "🚀 Lancer l'Analyse",
        type="primary",
        use_container_width=True
    )

# --- LOGIQUE PRINCIPALE ---
if launch_button:
    # Validation
    if len(selected_countries) < 2:
        st.error("❌ Veuillez sélectionner au moins 2 pays.")
        st.stop()
    
    if not selected_indicators:
        st.error("❌ Veuillez sélectionner au moins 1 indicateur.")
        st.stop()
    
    st.session_state.analysis_running = True

if st.session_state.analysis_running:
    # Préparer les données
    selected_country_codes = [AFRICAN_COUNTRIES[c] for c in selected_countries]
    selected_country_names = selected_countries
    selected_indicator_codes = [INDICATORS[i] for i in selected_indicators]
    selected_indicator_names = selected_indicators
    
    # Afficher les sélections
    with st.expander("📋 Résumé de la configuration", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write("**Pays :**", ", ".join(selected_country_names))
        with col2:
            st.write("**Indicateurs :**", len(selected_indicators))
        with col3:
            st.write("**Période :**", f"{year_range[0]}-{year_range[1]}")
        with col4:
            st.write("**IA :**", f"{'🤖 '+provider_name if not use_demo_mode else '📝 Démo'}")
    
    # Créer barre de progression
    progress_bar = st.progress(0, text="Initialisation...")
    
    # --- ÉTAPE 1 : RÉCUPÉRATION DES DONNÉES ---
    with st.spinner("Étape 1/3 : Récupération des données depuis l'API Banque Mondiale..."):
        api = WorldBankAPI()
        all_data = []
        
        total_calls = len(selected_indicator_codes)
        
        for idx, (indicator_code, indicator_name) in enumerate(zip(selected_indicator_codes, selected_indicator_names)):
            progress = (idx + 1) / total_calls
            progress_bar.progress(progress, text=f"Récupération : {indicator_name}")
            
            df = api.fetch_indicator(
                indicator_code,
                selected_country_codes,
                year_range[0],
                year_range[1]
            )
            
            if not df.empty:
                df['indicator_name'] = indicator_name
                all_data.append(df)
        
        progress_bar.empty()
        
        # --- ÉTAPE 2 : TRAITEMENT & VISUALISATION ---
        with st.spinner("Étape 2/3 : Traitement et visualisation des données..."):
            if not all_data:
                st.error("Aucune donnée récupérée pour les filtres sélectionnés.")
                st.stop()
            
            # Combiner les données
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Pivoter pour format large
            pivot_df = combined_df.pivot_table(
                index=['country_name', 'year'],
                columns='indicator_name',
                values='value',
                aggfunc='first'
            ).reset_index()
            
            pivot_df.columns.name = None
            pivot_df = pivot_df.rename(columns={'country_name': 'Pays', 'year': 'Année'})
            pivot_df = pivot_df.sort_values(['Pays', 'Année'], ascending=[True, False])
            
            # Nettoyer lignes vides
            indicator_cols = [col for col in pivot_df.columns if col not in ['Pays', 'Année']]
            pivot_df = pivot_df.dropna(subset=indicator_cols, how='all')
            
            if pivot_df.empty:
                st.error("Données vides après pivotage. Vérifiez les plages de dates.")
                st.stop()
        
        # Afficher données
        st.subheader("1. Aperçu des Données")
        st.info("💡 **Note :** Le taux de scolarisation peut dépasser 100% (c'est le taux *brut* qui inclut les redoublants et élèves hors âge officiel).")
        
        # --- VISUALISATIONS INTERACTIVES ---
        st.subheader("2. Visualisations des Tendances")
        
        # Créer tabs pour organiser les graphiques
        tab_list = [
            "📈 Dépenses Publiques", 
            "🎓 Taux d'Achèvement (Filles)", 
            "📚 Taux de Scolarisation",
            "📊 Comparaison",
            "📋 Données Brutes"
        ]
        tabs = st.tabs(tab_list)
        
        with tabs[0]: # Dépenses
            if 'Dépenses publiques d\'éducation (% du PIB)' in pivot_df.columns:
                fig1 = create_trend_chart(
                    pivot_df,
                    'Dépenses publiques d\'éducation (% du PIB)',
                    'Évolution des Dépenses Publiques d\'Éducation',
                    'Dépenses (% du PIB)'
                )
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("Indicateur 'Dépenses publiques' non sélectionné ou données non disponibles.")
        
        with tabs[1]: # Achèvement Filles
            if 'Taux d\'achèvement (Primaire, Filles)' in pivot_df.columns:
                fig2 = create_trend_chart(
                    pivot_df,
                    'Taux d\'achèvement (Primaire, Filles)',
                    'Évolution du Taux d\'Achèvement Primaire (Filles)',
                    'Taux d\'achèvement (%)'
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Indicateur 'Taux d'achèvement (Filles)' non sélectionné ou données non disponibles.")
        
        with tabs[2]: # Scolarisation
            if 'Taux de scolarisation (Primaire)' in pivot_df.columns:
                fig3 = create_trend_chart(
                    pivot_df,
                    'Taux de scolarisation (Primaire)',
                    'Évolution du Taux de Scolarisation Primaire',
                    'Taux de scolarisation (%)'
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Indicateur 'Taux de scolarisation' non sélectionné ou données non disponibles.")
        
        with tabs[3]: # Comparaison
            st.markdown("##### 🔬 Comparaison Annuelle")
            
            # Récupérer les années disponibles, triées de la plus récente à la plus ancienne
            available_years = sorted(pivot_df['Année'].unique(), reverse=True)
            
            if not available_years:
                st.warning("Aucune donnée annuelle à comparer.")
            else:
                # Créer le sélecteur d'année
                selected_year = st.selectbox(
                    "Choisissez l'année de comparaison :", 
                    options=available_years,
                    index=0
                )
                
                if selected_year:
                    # Trouver les pays qui ont des données pour cette année
                    data_for_year = pivot_df[pivot_df['Année'] == selected_year]
                    countries_with_data = data_for_year['Pays'].unique()
                    
                    # Comparer avec la liste complète des pays sélectionnés
                    countries_missing_data = [
                        pays for pays in selected_country_names 
                        if pays not in countries_with_data
                    ]
                    
                    if countries_missing_data:
                        st.warning(
                            f"**Données non disponibles pour {selected_year} pour :** "
                            f"{', '.join(countries_missing_data)}"
                        )
                    
                    # Générer le graphique
                    fig4 = create_comparison_chart(
                        pivot_df,
                        indicator_cols,
                        selected_year
                    )
                    
                    if fig4:
                        st.plotly_chart(fig4, use_container_width=True)
                    else:
                        st.info(f"Aucune donnée à afficher pour les pays trouvés en {selected_year}.")
        
        with tabs[4]: # Données brutes
            st.markdown(f"**Pays :** {', '.join(selected_country_names)}  \n**Période :** {year_range[0]}-{year_range[1]}")
            st.dataframe(pivot_df, use_container_width=True)
            
            # Option d'export
            csv = pivot_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les données (CSV)",
                data=csv,
                file_name=f"donnees_education_{'-'.join(selected_country_names[:3])}.csv",
                mime="text/csv",
            )
        
        # --- ÉTAPE 3 : ANALYSE IA ---
        with st.spinner(f"Étape 3/3 : L'IA {provider_name} analyse les tendances et génère les recommandations..."):
            if use_demo_mode:
                st.subheader("3. Analyse & Recommandations (Mode Démo)")
                format_ai_analysis(DEMO_RESPONSE)
                st.info("💡 Configurez une clé API Gemini ou Claude pour des analyses personnalisées.")
            else:
                st.subheader(f"3. Analyse & Recommandations ({provider_name})")
                
                data_csv = pivot_df.to_csv(index=False)
                
                # Appel à l'IA appropriée
                if selected_provider == "gemini":
                    analysis = generate_gemini_analysis(
                        data_csv, 
                        selected_country_names, 
                        selected_indicator_names,
                        AI_API_KEY
                    )
                else:  # claude
                    analysis = generate_claude_analysis(
                        data_csv, 
                        selected_country_names, 
                        selected_indicator_names,
                        AI_API_KEY
                    )
                
                if analysis:
                    format_ai_analysis(analysis)
                    st.success(f"✅ Analyse générée avec succès par {provider_name}.")
                else:
                    st.error(f"L'analyse IA {provider_name} a échoué. Vérifiez la clé API et la console.")

else:
    st.info("👈 Utilisez la barre latérale pour configurer votre analyse et cliquez sur 'Lancer l'Analyse'.")

# --- FOOTER ---
st.sidebar.markdown("---")

# Ajouter un bouton pour réinitialiser
if st.sidebar.button("🔄 Réinitialiser l'Analyse"):
    st.session_state.analysis_running = False
    st.rerun()
    
st.sidebar.markdown("---")

# Afficher le cache seulement si disponible
if cache_enabled:
    st.sidebar.markdown(f"**Cache API :** {len(list(CACHE_DIR.glob('*.pkl')))} fichiers")
    if st.sidebar.button("🗑️ Vider le cache API"):
        count = 0
        for f in CACHE_DIR.glob('*.pkl'):
            try:
                f.unlink()
                count += 1
            except:
                pass
        st.sidebar.success(f"{count} fichiers cache vidés!")
else:
    st.sidebar.markdown("**Cache API :** Désactivé (cloud)")

# Afficher info sur le fournisseur d'IA dans le footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 À propos des IA")
st.sidebar.markdown("""
**Google Gemini:**
- ✅ Gratuit (15 req/min)
- ⚡ Rapide
- 🎯 Bon pour analyses courtes

**Claude (Anthropic):**
- 💎 Payant (crédits gratuits disponibles)
- 🧠 Plus approfondi
- 📊 Excellent pour analyses complexes
""")
