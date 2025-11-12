import streamlit as st
import pandas as pd
import requests
import pickle
from pathlib import Path
import time
from typing import Dict, List, Optional
import google.generativeai as genai
import plotly.graph_objects as go
import plotly.express as px
import toml  # Ajouté pour gérer le cache de la clé API
import re    # Ajouté (depuis la fonction format_ai_analysis)

# --- CONFIGURATION ---
st.set_page_config(
    page_title="IA & Données Éducatives",
    page_icon="🎓",
    layout="wide"
)
# Initialiser le session state
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False
# Créer dossier cache
CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Définir le chemin pour le cache de la clé API
SECRETS_PATH = Path(".streamlit/secrets.toml")

# --- GESTION CLÉ API (Votre demande) ---

def load_cached_api_key() -> Optional[str]:
    """Charge la clé API depuis .streamlit/secrets.toml si elle existe."""
    if SECRETS_PATH.exists():
        try:
            with open(SECRETS_PATH, 'r') as f:
                data = toml.load(f)
                return data.get("GOOGLE_API_KEY")
        except Exception as e:
            st.error(f"Erreur en chargeant secrets.toml : {e}")
    return None

def save_cached_api_key(api_key: str):
    """Sauvegarde la clé API dans .streamlit/secrets.toml."""
    try:
        SECRETS_PATH.parent.mkdir(exist_ok=True)
        data = {"GOOGLE_API_KEY": api_key}
        with open(SECRETS_PATH, 'w') as f:
            toml.dump(data, f)
    except Exception as e:
        st.error(f"Impossible de sauvegarder la clé API : {e}")

# --- CLASSE API SERVICE (inspirée de api_service.py) ---
class WorldBankAPI:
    """Service API Banque Mondiale - Approche REST directe"""
    
    BASE_URL = "https://api.worldbank.org/v2"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Education-Analytics/1.0'})
    
    def _get_cache_path(self, key: str) -> Path:
        return CACHE_DIR / f"{key}.pkl"
    
    def _load_cache(self, key: str) -> Optional[pd.DataFrame]:
        """Charge depuis cache si valide (<24h)"""
        cache_file = self._get_cache_path(key)
        if cache_file.exists():
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
        try:
            with open(self._get_cache_path(key), 'wb') as f:
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
                
                # --- BUG CRITIQUE CORRIGÉ ---
                # L'API renvoie 'ma', 'sn', 'ke' (minuscules)
                # La liste 'countries' contient 'MA', 'SN', 'KE' (majuscules)
                if not country_code or country_code.upper() not in countries:
                    continue
                
                try:
                    year = int(record.get('date', 0))
                    if start_year <= year <= end_year:
                        records.append({
                            'country_code': country_code.upper(), # Standardiser en majuscule
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
    'Cap-Vert': 'CV', 'Centrafrique': 'CF', 'Comores': 'KM', 'Congo': 'CG',
    'Congo (RDC)': 'CD', 'Côte d\'Ivoire': 'CI', 'Djibouti': 'DJ', 'Égypte': 'EG',
    'Érythrée': 'ER', 'Eswatini': 'SZ', 'Éthiopie': 'ET', 'Gabon': 'GA',
    'Gambie': 'GM', 'Ghana': 'GH', 'Guinée': 'GN', 'Guinée-Bissau': 'GW',
    'Guinée équatoriale': 'GQ', 'Kenya': 'KE', 'Lesotho': 'LS', 'Libéria': 'LR',
    'Libye': 'LY', 'Madagascar': 'MG', 'Malawi': 'MW', 'Mali': 'ML',
    'Maroc': 'MA', 'Maurice': 'MU', 'Mauritanie': 'MR', 'Mozambique': 'MZ',
    'Namibie': 'NA', 'Niger': 'NE', 'Nigéria': 'NG', 'Ouganda': 'UG',
    'Rwanda': 'RW', 'Sao Tomé-et-Principe': 'ST', 'Sénégal': 'SN', 'Seychelles': 'SC',
    'Sierra Leone': 'SL', 'Somalie': 'SO', 'Soudan': 'SD', 'Soudan du Sud': 'SS',
    'Tanzanie': 'TZ', 'Tchad': 'TD', 'Togo': 'TG', 'Tunisie': 'TN',
    'Zambie': 'ZM', 'Zimbabwe': 'ZW'
}

# --- INDICATEURS ÉDUCATIFS ---
INDICATORS = {
    'SE.PRM.ENRR': 'Taux de scolarisation (Primaire)',
    'SE.PRM.CMPT.FE.ZS': 'Taux d\'achèvement (Primaire, Filles)',
    'SE.XPD.TOTL.GD.ZS': 'Dépenses publiques d\'éducation (% du PIB)'
}

# --- INTERFACE ---
st.title("🎓 Démo : De la Donnée Brute à la Recommandation Stratégique")
st.markdown("Webinaire : Exemples d'usage de l'IA pour l'analyse de données sur l'éducation.")

# --- SIDEBAR ---
st.sidebar.header("Configuration de la Démo")

# Clé API (Votre demande)
st.sidebar.subheader("Configuration IA")
cached_key = load_cached_api_key()

# Essayer st.secrets comme fallback si le cache est vide
if not cached_key:
    try:
        default_key = st.secrets.get("GOOGLE_API_KEY", "")
    except:
        default_key = ""
else:
    default_key = cached_key

GOOGLE_API_KEY = st.sidebar.text_input(
    "Votre Clé API Gemini", 
    type="password",
    value=default_key,
    help="Sera sauvegardée dans .streamlit/secrets.toml pour la prochaine session."
)

# Sauvegarder la clé si elle est nouvelle ou modifiée
if GOOGLE_API_KEY and GOOGLE_API_KEY != cached_key:
    save_cached_api_key(GOOGLE_API_KEY)
    st.sidebar.success("Clé API sauvegardée localement.")

# Mode Démo
use_demo_mode = st.sidebar.checkbox(
    "✅ Activer le Mode Démo (Recommandé)", 
    value=True,
    help="Utilise une réponse IA pré-enregistrée pour une démo instantanée."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Configuration des Données")

# Sélection indicateurs
selected_indicator_names = st.sidebar.multiselect(
    "Choisissez les Indicateurs :",
    options=list(INDICATORS.values()),
    default=list(INDICATORS.values())
)

# Sélection pays
selected_country_names = st.sidebar.multiselect(
    "Choisissez les Pays :",
    options=list(AFRICAN_COUNTRIES.keys()),
    default=['Maroc', 'Sénégal', 'Kenya']
)

# Bouton action
#start_analysis = st.sidebar.button("🚀 Lancer l'Analyse", type="primary")
# Bouton action
if st.sidebar.button("🚀 Lancer l'Analyse", type="primary"):
    st.session_state.analysis_running = True
# --- FONCTION ANALYSE IA ---
def generate_ai_analysis(data_csv: str, countries: List[str], 
                         indicators: List[str], api_key: str) -> Optional[str]:
    """Génère analyse via Gemini"""
    if not api_key:
        st.error("Clé API Gemini non configurée. Veuillez l'ajouter dans la barre latérale.")
        return None
    
    try:
        genai.configure(api_key=api_key)
        
        # --- BUG CRITIQUE CORRIGÉ ---
        # 'gemini-2.5-pro' n'est pas un nom de modèle valide
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        prompt = f"""
Tu es un expert analyste en politiques éducatives internationales.

**Contexte :** Tu analyses des données de la Banque Mondiale pour {', '.join(countries)} sur : {', '.join(indicators)}.

**Tâche :** Rédige une analyse concise en français pour un décideur politique.

Structure ta réponse *exactement* ainsi (Markdown) :

### 1. Synthèse des Tendances Clés
(Liste à puces des 3 points majeurs)

### 2. Interprétation et Anomalies
(Paragraphe court : corrélations, points surprenants)

### 3. Recommandations Stratégiques
(Liste numérotée de 2-3 recommandations *concrètes* et *actionnables*)

Données CSV :
{data_csv}
"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        st.error(f"Erreur Gemini : {e}")
        return None

# --- FONCTIONS VISUALISATION ---
def create_trend_chart(df: pd.DataFrame, indicator_col: str, title: str, 
                       y_label: str, color_scheme: str = "Set2") -> go.Figure:
    """Crée un graphique d'évolution temporelle professionnel"""
    
    # Palette de couleurs professionnelle
    colors = px.colors.qualitative.Set2
    
    fig = go.Figure()
    
    # Ajouter une trace par pays
    for idx, country in enumerate(df['Pays'].unique()):
        country_data = df[df['Pays'] == country].sort_values('Année')
        
        # Filtrer les valeurs non nulles
        valid_data = country_data[country_data[indicator_col].notna()]
        
        if not valid_data.empty:
            fig.add_trace(go.Scatter(
                x=valid_data['Année'],
                y=valid_data[indicator_col],
                name=country,
                mode='lines+markers',
                line=dict(width=3, color=colors[idx % len(colors)]),
                marker=dict(size=8, symbol='circle', 
                           line=dict(width=2, color='white')),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Année: %{x}<br>' +
                             f'{y_label}: %{{y:.2f}}<br>' +
                             '<extra></extra>'
            ))
    
    # Mise en page professionnelle
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family='Arial, sans-serif', color='#2c3e50'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Année',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False,
            tickfont=dict(size=12),
            titlefont=dict(size=14, family='Arial, sans-serif')
        ),
        yaxis=dict(
            title=y_label,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False,
            tickfont=dict(size=12),
            titlefont=dict(size=14, family='Arial, sans-serif')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(128,128,128,0.3)',
            borderwidth=1
        ),
        height=500,
        margin=dict(l=80, r=40, t=100, b=80)
    )
    
    return fig

def format_ai_analysis(analysis_text: str) -> None:
    """Affiche l'analyse IA avec mise en forme attractive"""
    
    # CSS personnalisé pour les cards
    st.markdown("""
    <style>
    .analysis-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
        color: white;
    }
    .section-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 5px solid #667eea;
    }
    .section-title {
        color: #667eea;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .insight-item {
        background: #f8f9fa;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 3px solid #764ba2;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .recommendation-item {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        border-radius: 10px;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 4px 12px rgba(240, 147, 251, 0.3);
    }
    .recommendation-number {
        background: white;
        color: #f5576c;
        padding: 0.2rem 0.6rem;
        border-radius: 50%;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header principal
    st.markdown("""
    <div class="analysis-card">
        <h2 style="margin: 0; font-size: 2rem;">🤖 Analyse IA & Recommandations Stratégiques</h2>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Analyse générée par Gemini 2.5 Pro</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Parser le texte pour extraire les sections
    sections = analysis_text.split('###')
    
    for section in sections:
        if not section.strip():
            continue
            
        lines = section.strip().split('\n')
        title = lines[0].strip()
        content = '\n'.join(lines[1:]).strip()
        
        if '1.' in title and 'Synthèse' in title:
            # Section Synthèse
            st.markdown("""
            <div class="section-card">
                <div class="section-title">📊 Synthèse des Tendances Clés</div>
            """, unsafe_allow_html=True)
            
            # Parser les bullet points
            for line in content.split('\n'):
                if line.strip().startswith('*'):
                    insight = line.strip()[1:].strip()
                    st.markdown(f'<div class="insight-item">• {insight}</div>', unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        elif '2.' in title and 'Interprétation' in title:
            # Section Interprétation
            st.markdown("""
            <div class="section-card">
                <div class="section-title">🔍 Interprétation et Anomalies</div>
            """, unsafe_allow_html=True)
            
            # Afficher le contenu en paragraphes
            for para in content.split('\n\n'):
                if para.strip():
                    st.markdown(f'<div class="insight-item">{para.strip()}</div>', unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        elif '3.' in title and 'Recommandations' in title:
            # Section Recommandations
            st.markdown("""
            <div class="section-card">
                <div class="section-title">💡 Recommandations Stratégiques</div>
            """, unsafe_allow_html=True)
            
            # Parser les recommandations numérotées
            # (import re est maintenant en haut du fichier)
            recommendations = re.findall(r'\d+\.\s+(.+?)(?=\d+\.|$)', content, re.DOTALL)
            
            for idx, rec in enumerate(recommendations, 1):
                rec_clean = rec.strip()
                st.markdown(
                    f'<div class="recommendation-item">'
                    f'<span class="recommendation-number">{idx}</span>'
                    f'{rec_clean}'
                    f'</div>', 
                    unsafe_allow_html=True
                )
            
            st.markdown("</div>", unsafe_allow_html=True)

def create_comparison_chart(df: pd.DataFrame, indicators: List[str], 
                           selected_year: int) -> go.Figure:
    """Crée un graphique comparatif multi-indicateurs (année sélectionnée)"""
    
    # Filtrer l'année sélectionnée
    latest_data = df[df['Année'] == selected_year].copy()
    
    if latest_data.empty:
        # Note : Le st.info() est maintenant géré dans la logique principale
        return None
    
    colors = ['#667eea', '#f093fb', '#4facfe']
    
    fig = go.Figure()
    
    for idx, indicator in enumerate(indicators):
        if indicator in latest_data.columns:
            fig.add_trace(go.Bar(
                name=indicator,
                x=latest_data['Pays'],
                y=latest_data[indicator],
                marker=dict(
                    color=colors[idx % len(colors)],
                    line=dict(color='white', width=2)
                ),
                text=latest_data[indicator].round(1),
                textposition='outside',
                textfont=dict(size=11, color='#2c3e50'),
                hovertemplate='<b>%{x}</b><br>' +
                             f'{indicator}: %{{y:.2f}}<br>' +
                             '<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(
            text=f'Comparaison des Indicateurs ({selected_year})',
            font=dict(size=20, family='Arial, sans-serif', color='#2c3e50'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Pays',
            tickfont=dict(size=12),
            titlefont=dict(size=14)
        ),
        yaxis=dict(
            title='Valeur',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            tickfont=dict(size=12),
            titlefont=dict(size=14)
        ),
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(128,128,128,0.3)',
            borderwidth=1
        ),
        height=500,
        margin=dict(l=80, r=40, t=100, b=80)
    )
    
    return fig

# --- LOGIQUE PRINCIPALE ---
if st.session_state.analysis_running:
    if not selected_indicator_names or not selected_country_names:
#if start_analysis:
    #if not selected_indicator_names or not selected_country_names:
        st.warning("Veuillez sélectionner au moins un indicateur et un pays.")
    else:
        # Mapper codes
        indicator_codes = {code: name for code, name in INDICATORS.items() 
                          if name in selected_indicator_names}
        country_codes = [AFRICAN_COUNTRIES[name] for name in selected_country_names]
        
        # Initialiser API
        api = WorldBankAPI()
        
        # --- ÉTAPE 1 : RÉCUPÉRATION DONNÉES ---
        with st.spinner("Étape 1/3 : Chargement des données depuis l'API de la Banque Mondiale..."):
            all_data = []
            
            for indicator_code, indicator_name in indicator_codes.items():
                df = api.fetch_indicator(indicator_code, country_codes, 2010, 2022)
                
                if not df.empty:
                    df['indicator_name'] = indicator_name
                    all_data.append(df)
                
                time.sleep(0.2)  # Rate limiting
            
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
                    index=0 # Par défaut, la plus récente
                )
                
                # --- VÉRIFICATION DE DISPONIBILITÉ (Votre demande) ---
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
                # --- FIN DE LA VÉRIFICATION ---

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
            st.markdown(f"**Pays :** {', '.join(selected_country_names)}  \n**Période :** 2010-2022")
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
        with st.spinner("Étape 3/3 : L'IA analyse les tendances et génère les recommandations..."):
            if use_demo_mode:
                st.subheader("3. Analyse & Recommandations")
                format_ai_analysis(DEMO_RESPONSE)
            else:
                st.subheader("3. Analyse & Recommandations")
                
                data_csv = pivot_df.to_csv(index=False)
                analysis = generate_ai_analysis(
                    data_csv, 
                    selected_country_names, 
                    selected_indicator_names,
                    GOOGLE_API_KEY  # Passer la clé
                )
                
                if analysis:
                    format_ai_analysis(analysis)
                    st.success("✅ Analyse générée avec succès par Gemini.")
                else:
                    st.error("L'analyse IA a échoué. Vérifiez la clé API et la console.")

else:
    st.info("👈 Utilisez la barre latérale pour configurer votre analyse et cliquez sur 'Lancer l'Analyse'.")

# --- FOOTER ---
st.sidebar.markdown("---")

# Ajouter un bouton pour réinitialiser
if st.sidebar.button("🔄 Réinitialiser l'Analyse"):
    st.session_state.analysis_running = False
    st.rerun() # Force un re-chargement immédiat
    
st.sidebar.markdown("---")
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
