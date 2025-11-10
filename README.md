# Guide d'Utilisation - Application Webinaire v2.0

## 🚀 Installation Rapide

```bash
pip install streamlit pandas requests google-generativeai
streamlit run webinar_app_v2.py
```

## 🔧 Architecture Technique

### Approche API REST Directe
L'application utilise une approche robuste inspirée de l'architecture professionnelle :

1. **API REST Native** : Requêtes directes à `api.worldbank.org/v2`
2. **Cache Local** : Système de cache pickle (24h) pour performances
3. **Parser JSON** : Contrôle total sur le traitement des données
4. **Gestion d'erreurs** : Retry logic et fallbacks

### Différences avec v1
- ❌ Plus de dépendance `wbdata` (instable)
- ✅ Requêtes HTTP directes via `requests`
- ✅ Cache intelligent avec pickle
- ✅ 54 pays africains disponibles

## 📊 Indicateurs Éducatifs

| Code API | Nom | Description |
|----------|-----|-------------|
| SE.PRM.ENRR | Taux de scolarisation | Taux brut de scolarisation primaire |
| SE.PRM.CMPT.FE.ZS | Taux d'achèvement (Filles) | % de filles complétant le primaire |
| SE.XPD.TOTL.GD.ZS | Dépenses publiques | % du PIB pour l'éducation |

## 🎯 Utilisation

### Mode Démo (Recommandé pour présentation)
1. Cocher "✅ Activer le Mode Démo"
2. Sélectionner pays et indicateurs
3. Cliquer "🚀 Lancer l'Analyse"
4. Résultats instantanés (réponse pré-enregistrée)

### Mode Live (Analyse réelle)
1. Décocher "Mode Démo"
2. Saisir clé API Gemini
3. Sélectionner configuration
4. Lancer l'analyse (appel API réel)

## 🌍 Pays Africains (54)

Tous les pays africains sont disponibles, dont :
- Afrique du Nord : Maroc, Algérie, Tunisie, Égypte, Libye
- Afrique de l'Ouest : Sénégal, Nigeria, Ghana, Côte d'Ivoire, Mali
- Afrique de l'Est : Kenya, Éthiopie, Tanzanie, Ouganda, Rwanda
- Afrique Australe : Afrique du Sud, Zimbabwe, Botswana, Namibie
- Afrique Centrale : Cameroun, RDC, Congo, Gabon, Tchad

## 📁 Structure des Données

### Format API World Bank
```json
{
  "country": {"id": "MA", "value": "Morocco"},
  "date": "2020",
  "value": 98.5,
  "indicator": {"id": "SE.PRM.ENRR"}
}
```

### Format Traité (DataFrame)
```
Pays    | Année | Taux scolarisation | Taux achèvement | Dépenses PIB
--------|-------|-------------------|-----------------|-------------
Maroc   | 2022  | 98.2              | 89.5           | 4.8
Sénégal | 2022  | 99.1              | 72.3           | 5.4
Kenya   | 2022  | 100.0             | 94.7           | 5.1
```

## 🧠 Prompt IA (Gemini)

Le prompt est structuré pour obtenir une analyse professionnelle :

```
Tu es un expert analyste en politiques éducatives internationales.

Contexte : Données Banque Mondiale pour [pays] sur [indicateurs]

Tâche : Analyse concise pour décideur politique

Structure :
1. Synthèse des Tendances Clés (3 points)
2. Interprétation et Anomalies (corrélations)
3. Recommandations Stratégiques (2-3 actions concrètes)
```

## 🎬 Script de Présentation (15 min)

### Intro (2 min)
"Voici une application développée en Python avec Streamlit qui démontre comment l'IA transforme des données brutes en recommandations stratégiques."

### Configuration (3 min)
"Je sélectionne :
- Indicateurs : Scolarisation, Achèvement filles, Dépenses publiques
- Pays : Maroc, Sénégal, Kenya (comparaison)
- Mode Démo activé pour fluidité"

### Données (2 min)
"L'application interroge l'API World Bank... Voilà les données brutes. Des chiffres difficiles à interpréter pour un ministre."

### Analyse IA (5 min)
"Instantanément, l'IA structure l'analyse :
- **Synthèse** : Détecte disparité inscription/achèvement au Sénégal
- **Interprétation** : Investissement élevé ne donne pas résultats équivalents
- **Recommandations** : Enquête qualitative ciblée, audit dépenses"

### Conclusion (3 min)
"En quelques clics : données brutes → recommandations actionnables. C'est la puissance de l'IA pour les politiques publiques."

## 🐛 Dépannage

### Erreur "No module named 'requests'"
```bash
pip install requests
```

### Erreur API Gemini
- Vérifier clé API sur https://makersuite.google.com/app/apikey
- Ou activer Mode Démo

### Pas de données récupérées
- Vérifier connexion internet
- Consulter logs dans terminal
- Cache corrompu ? Cliquer "🗑️ Vider le cache"

### Données incomplètes
Certains pays/années peuvent avoir des données manquantes dans la Banque Mondiale. C'est normal.

## 💡 Avantages Techniques

### Cache Intelligent
- Premier chargement : 10-15 secondes
- Chargements suivants : <1 seconde
- Validité : 24 heures
- Stockage : `data_cache/*.pkl`

### Robustesse
- Retry automatique si échec réseau
- Validation structure JSON
- Gestion valeurs nulles
- Filtrage données aberrantes

### Performance
- Requêtes parallélisables (séquentiel pour démo)
- Rate limiting : 0.2s entre requêtes
- Timeout : 30s par requête
- Cache : réduit charge API 95%

## 📈 Extensions Possibles

1. **Visualisations** : Ajouter Plotly pour graphiques
2. **Export** : Bouton download CSV/Excel
3. **Comparaison temporelle** : Évolution sur 10 ans
4. **Plus d'indicateurs** : Alphabétisation, ratio prof/élève
5. **Prédictions** : Modèle ML pour projections

## 🔐 Sécurité

### Clé API Gemini
Ne JAMAIS commiter la clé dans le code. Utiliser :
- Fichier `.streamlit/secrets.toml` (local)
- Variables d'environnement (production)
- Input utilisateur (démo)

### Exemple secrets.toml
```toml
GOOGLE_API_KEY = "AIza..."
```

Placez dans : `.streamlit/secrets.toml`

## 📞 Support

Pour toute question technique :
1. Vérifier ce README
2. Consulter logs dans terminal
3. Tester avec Mode Démo d'abord
4. Vérifier connexion API World Bank

## 🎓 Ressources

- **API World Bank** : https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
- **Google Gemini** : https://ai.google.dev/
- **Streamlit Docs** : https://docs.streamlit.io/
- **Pandas Guide** : https://pandas.pydata.org/docs/

---

**Version** : 2.0  
**Date** : Novembre 2025  
**Auteur** : Zakaria Benhoumad
