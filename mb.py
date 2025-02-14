import json
import torch
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 🔹 Charger la liste de mots toxiques depuis un fichier JSON
def charger_mots_toxiques(fichier="mots_toxiques.json"):
    with open(fichier, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data["mots_toxiques"])

# Charger les mots toxiques
mots_toxiques = charger_mots_toxiques()

# 🔹 Charger le modèle BERT entraîné
MODEL_NAME = "bert-base-multilingual-cased"  # Remplace avec ton modèle personnalisé si besoin
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Vérifier si un commentaire contient un mot toxique
def est_toxique_dictionnaire(commentaire):
    mots = commentaire.lower().split()
    return any(mot in mots_toxiques for mot in mots)

# Prédire la toxicité avec BERT
def predire_toxicite_bert(commentaire):
    inputs = tokenizer(commentaire, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    return scores[0][1].item()  # Probabilité de toxicité

# 🔹 Interface avec Streamlit
st.title("🛡️ Modération Automatique des Commentaires")
st.write("Entrez un commentaire pour vérifier s'il est toxique.")

# Saisie utilisateur
commentaire = st.text_area("💬 Saisissez votre commentaire ici", "")

if st.button("Analyser"):
    if commentaire.strip():
        # Vérification avec le dictionnaire
        if est_toxique_dictionnaire(commentaire):
            st.error("🚨 Ce commentaire contient des mots interdits ! (Détection par Dictionnaire)")
        else:
            # Prédiction BERT
            score_toxicite = predire_toxicite_bert(commentaire)
            if score_toxicite > 0.5:
                st.warning(f"⚠️ Ce commentaire est probablement!: toxique ({score_toxicite:.2f})")
            else:
                st.success("✅ Ce commentaire est acceptable !: non toxique")
    else:
        st.warning("Veuillez entrer un commentaire avant d'analyser.")

