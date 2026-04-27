from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import httpx
import re
import os
import unicodedata
import pdfplumber
import ollama
import json
import io
import ssl
import socket
import datetime
import time
import requests
from bs4 import BeautifulSoup

from sentence_transformers import SentenceTransformer
import faiss

# ── Normalisation (supprime les accents pour matcher sans accent) ─────────────
def normalize(text: str) -> str:
    """Minuscule + supprime les accents : développeur → developpeur"""
    return unicodedata.normalize('NFD', text.lower()).encode('ascii', 'ignore').decode('ascii')


# ── Dictionnaire d'expansion de domaines ─────────────────────────────────────
# Les clés sont déjà normalisées (sans accents, minuscules)
DOMAIN_EXPANSION = {
    # ── Frontend ──────────────────────────────────────────────────────────────
    "front":             ["angular", "react", "vue", "javascript", "typescript", "frontend", "développeur frontend", "développeur web", "SPA", "composant", "programmation web"],
    "frontend":          ["angular", "react", "vue", "javascript", "typescript", "développeur frontend", "développeur web", "SPA", "composant", "programmation web", "code"],
    "front end":         ["angular", "react", "vue", "javascript", "typescript", "développeur frontend", "programmation web"],
    "dev front":         ["angular", "react", "vue", "javascript", "typescript", "développeur frontend", "SPA", "programmation"],
    "developpeur front": ["angular", "react", "vue", "javascript", "typescript", "développeur frontend", "SPA", "composant"],
    "dev angular":       ["angular", "typescript", "javascript", "rxjs", "frontend", "SPA", "développeur angular"],
    "developpeur angular":["angular", "typescript", "javascript", "rxjs", "frontend", "SPA", "composant angular"],
    "angular":           ["frontend", "typescript", "javascript", "SPA", "rxjs", "composant angular", "développeur angular", "développeur frontend"],
    "react":             ["frontend", "javascript", "typescript", "next.js", "composant react", "développeur react", "développeur frontend"],
    "vue":               ["frontend", "javascript", "typescript", "vuejs", "nuxt", "développeur frontend"],
    "javascript":        ["frontend", "react", "angular", "vue", "typescript", "développeur web"],
    "typescript":        ["frontend", "angular", "react", "javascript", "développeur frontend"],
    "nextjs":            ["frontend", "react", "javascript", "typescript", "développeur frontend"],

    # ── Backend ───────────────────────────────────────────────────────────────
    "back":              ["spring boot", "java", "node.js", "api REST", "microservices", "backend", "développeur backend", "serveur", "base de données"],
    "backend":           ["spring boot", "java", "node.js", "api REST", "microservices", "développeur backend", "serveur", "base de données"],
    "back end":          ["spring boot", "java", "node.js", "api REST", "microservices", "développeur backend", "serveur"],
    "dev back":          ["spring boot", "java", "node.js", "api REST", "microservices", "développeur backend"],
    "developpeur back":  ["spring boot", "java", "node.js", "api REST", "microservices", "développeur backend"],
    "java":              ["spring boot", "backend", "microservices", "api", "jpa", "hibernate", "développeur java", "développeur backend"],
    "spring":            ["spring boot", "java", "backend", "api REST", "microservices", "développeur backend"],
    "spring boot":       ["java", "backend", "api REST", "microservices", "hibernate", "jpa", "développeur backend"],
    "node":              ["node.js", "backend", "javascript", "express", "api REST", "développeur backend"],
    "nodejs":            ["backend", "javascript", "express", "api REST", "développeur backend"],
    "php":               ["backend", "laravel", "symfony", "api REST", "développeur backend", "développeur web"],
    "laravel":           ["php", "backend", "api REST", "développeur backend"],
    "django":            ["python", "backend", "api REST", "développeur backend"],
    "fastapi":           ["python", "backend", "api REST", "développeur backend"],

    # ── Fullstack ─────────────────────────────────────────────────────────────
    "fullstack":         ["frontend", "backend", "angular", "react", "spring boot", "node.js", "javascript", "développeur fullstack"],
    "full stack":        ["frontend", "backend", "angular", "react", "spring boot", "javascript", "développeur fullstack"],
    "developpeur web":   ["frontend", "backend", "javascript", "html", "css", "react", "angular", "développeur web"],

    # ── AI / Data ─────────────────────────────────────────────────────────────
    "ai":                ["intelligence artificielle", "machine learning", "deep learning", "data science", "NLP", "python", "modèle prédictif"],
    "ia":                ["intelligence artificielle", "machine learning", "deep learning", "data science", "NLP", "python"],
    "intelligence artificielle": ["machine learning", "deep learning", "data science", "neural network", "python"],
    "machine learning":  ["intelligence artificielle", "deep learning", "data science", "python", "scikit-learn", "tensorflow"],
    "deep learning":     ["machine learning", "tensorflow", "pytorch", "neural network", "data science"],
    "data science":      ["machine learning", "analyse de données", "python", "statistiques", "big data", "intelligence artificielle"],
    "data":              ["data science", "big data", "analyse", "python", "SQL", "machine learning"],
    "nlp":               ["traitement du langage naturel", "machine learning", "python", "transformers"],
    "data analyst":      ["SQL", "python", "data science", "tableau", "power bi", "analyse de données", "business intelligence"],
    "bi":                ["business intelligence", "power bi", "tableau", "data analyst", "sql", "rapports"],

    # ── UX / Design ───────────────────────────────────────────────────────────
    "ux":                ["user experience", "figma", "maquette", "prototype", "wireframe", "designer"],
    "ui":                ["user interface", "figma", "maquette", "wireframe", "designer"],
    "designer":          ["figma", "maquette", "prototype", "wireframe", "graphisme", "design graphique"],
    "design":            ["figma", "maquette", "prototype", "graphisme", "wireframe"],
    "figma":             ["ux", "ui", "maquette", "prototype", "designer"],

    # ── Mobile ────────────────────────────────────────────────────────────────
    "mobile":            ["android", "ios", "flutter", "react native", "kotlin", "swift", "application mobile", "développeur mobile"],
    "flutter":           ["mobile", "dart", "android", "ios", "application mobile", "développeur mobile"],
    "android":           ["mobile", "kotlin", "java", "application mobile", "développeur mobile"],
    "ios":               ["mobile", "swift", "objective-c", "application mobile", "développeur mobile"],
    "react native":      ["mobile", "javascript", "android", "ios", "application mobile"],
    "developpeur mobile":["flutter", "android", "ios", "react native", "kotlin", "swift", "application mobile"],

    # ── DevOps / Cloud ────────────────────────────────────────────────────────
    "devops":            ["docker", "kubernetes", "ci/cd", "jenkins", "aws", "cloud", "déploiement", "pipeline", "linux"],
    "cloud":             ["aws", "azure", "gcp", "devops", "infrastructure", "docker", "kubernetes"],
    "docker":            ["devops", "kubernetes", "conteneur", "ci/cd", "cloud"],
    "aws":               ["cloud", "devops", "infrastructure", "amazon web services"],
    "linux":             ["système", "administration", "bash", "devops", "serveur"],

    # ── Base de données ───────────────────────────────────────────────────────
    "base de donnees":   ["SQL", "mongodb", "mysql", "postgresql", "oracle", "nosql"],
    "sql":               ["base de données", "mysql", "postgresql", "oracle", "requête"],
    "mongodb":           ["nosql", "base de données", "json", "document"],
    "postgresql":        ["sql", "base de données", "backend"],
    "mysql":             ["sql", "base de données", "backend"],

    # ── Sécurité / Réseau ─────────────────────────────────────────────────────
    "securite":          ["cybersécurité", "pentest", "réseau", "firewall", "cryptographie"],
    "cybersecurite":     ["sécurité", "pentest", "réseau", "firewall", "cryptographie", "ethical hacking"],
    "reseau":            ["réseau informatique", "cisco", "TCP/IP", "firewall", "infrastructure", "wifi"],
    "infrastructure":    ["réseau", "serveur", "cloud", "devops", "système", "administration"],
    "sysadmin":          ["administration système", "linux", "réseau", "infrastructure", "serveur"],

    # ── Expérience ────────────────────────────────────────────────────────────
    "senior":            ["expérimenté", "expert", "5 ans", "7 ans", "10 ans", "lead", "confirmé"],
    "junior":            ["débutant", "0 ans", "1 an", "2 ans", "entry level"],
    "confirme":          ["intermédiaire", "3 ans", "4 ans", "5 ans", "expérience solide"],
    "experimente":       ["senior", "expert", "5 ans expérience", "lead", "confirmé"],
    "debutant":          ["junior", "0 ans", "1 an", "entry level"],

    # ── Lieux / Villes tunisiennes (FR + EN) ─────────────────────────────────
    "tunis":             ["tunis", "tunisie", "capitale", "grand tunis", "tunisia"],
    "nabeul":            ["nabeul", "nabeul-hammamet", "tunisie", "cap bon", "hammamet"],
    "sousse":            ["sousse", "tunisie", "sahel", "port el kantaoui"],
    "sfax":              ["sfax", "tunisie"],
    "monastir":          ["monastir", "tunisie", "sahel"],
    "mahdia":            ["mahdia", "tunisie", "sahel"],
    "bizerte":           ["bizerte", "tunisie", "nord tunisie"],
    "djerba":            ["djerba", "jerba", "tunisie", "medenine"],
    "jerba":             ["djerba", "jerba", "tunisie"],
    "ariana":            ["ariana", "tunis", "grand tunis", "tunisie"],
    "la marsa":          ["la marsa", "tunis", "grand tunis", "tunisie"],
    "hammamet":          ["hammamet", "nabeul", "tunisie", "cap bon"],
    "gabes":             ["gabes", "gabès", "tunisie"],
    "gafsa":             ["gafsa", "tunisie"],
    "kairouan":          ["kairouan", "tunisie"],
    "beja":              ["béja", "beja", "tunisie"],
    "jendouba":          ["jendouba", "tunisie"],
    "zaghouan":          ["zaghouan", "tunisie"],
    "tozeur":            ["tozeur", "tunisie"],
    "medenine":          ["médenine", "medenine", "djerba", "tunisie"],
    "tataouine":         ["tataouine", "tunisie"],
    "kasserine":         ["kasserine", "tunisie"],
    "siliana":           ["siliana", "tunisie"],
    "kebili":            ["kébili", "kebili", "tunisie"],
    "sidi bouzid":       ["sidi bouzid", "tunisie"],
    "tunisie":           ["tunisie", "tunisia", "tunis", "nabeul", "sousse", "sfax", "monastir", "mahdia"],
    "tunisia":           ["tunisie", "tunis", "nabeul", "sousse", "sfax", "monastir"],
    # Remote
    "remote":            ["télétravail", "remote", "à distance", "distanciel", "en ligne"],
    "teletravail":       ["remote", "télétravail", "à distance", "distanciel", "en ligne"],
    "distance":          ["télétravail", "remote", "à distance", "distanciel"],
    "a distance":        ["télétravail", "remote", "distanciel", "en ligne"],

    # ── Traduction anglais → français (termes tech) ───────────────────────────
    "developer":              ["développeur", "dev", "ingénieur logiciel", "programmeur"],
    "java developer":         ["développeur java", "java", "spring boot", "backend", "développeur backend"],
    "frontend developer":     ["développeur frontend", "développeur web", "angular", "react", "vue", "javascript"],
    "front end developer":    ["développeur frontend", "développeur web", "angular", "react", "vue"],
    "backend developer":      ["développeur backend", "serveur", "api rest", "java", "spring boot"],
    "back end developer":     ["développeur backend", "serveur", "api rest", "java", "spring boot"],
    "fullstack developer":    ["développeur fullstack", "frontend", "backend", "angular", "react", "spring"],
    "full stack developer":   ["développeur fullstack", "frontend", "backend", "angular", "react"],
    "web developer":          ["développeur web", "frontend", "backend", "javascript", "html", "css"],
    "mobile developer":       ["développeur mobile", "flutter", "android", "ios", "application mobile"],
    "software engineer":      ["ingénieur logiciel", "développeur", "programmation", "code"],
    "software developer":     ["développeur logiciel", "développeur", "programmation"],
    "data scientist":         ["data science", "machine learning", "python", "analyse données", "intelligence artificielle"],
    "data engineer":          ["ingénieur data", "data", "big data", "pipeline", "sql", "python"],
    "devops engineer":        ["devops", "docker", "kubernetes", "ci/cd", "cloud", "déploiement"],
    "cloud engineer":         ["cloud", "aws", "azure", "gcp", "devops", "infrastructure"],
    "ux designer":            ["designer ux", "designer", "ux", "ui", "figma", "maquette"],
    "ui designer":            ["designer ui", "designer", "ui", "ux", "figma", "maquette"],
    "android developer":      ["développeur android", "android", "kotlin", "mobile", "application mobile"],
    "ios developer":          ["développeur ios", "ios", "swift", "mobile", "application mobile"],
    "flutter developer":      ["développeur flutter", "flutter", "mobile", "dart", "android", "ios"],
    "react developer":        ["développeur react", "react", "frontend", "javascript", "typescript"],
    "angular developer":      ["développeur angular", "angular", "frontend", "typescript", "javascript"],
    "python developer":       ["développeur python", "python", "backend", "django", "fastapi"],
    "php developer":          ["développeur php", "php", "laravel", "symfony", "backend"],
    "node developer":         ["développeur node", "node.js", "backend", "javascript", "api rest"],
    "nodejs developer":       ["développeur node", "node.js", "backend", "javascript"],
    "spring developer":       ["développeur spring", "spring boot", "java", "backend", "microservices"],
    "network engineer":       ["ingénieur réseau", "réseau", "cisco", "infrastructure", "tcp/ip"],
    "security engineer":      ["ingénieur sécurité", "cybersécurité", "pentest", "réseau", "firewall"],
    "machine learning engineer": ["machine learning", "intelligence artificielle", "python", "data science", "modèle prédictif"],
    "business analyst":       ["analyste métier", "analyse", "business intelligence", "consultant"],
    "project manager":        ["chef de projet", "gestion de projet", "management", "scrum", "agile"],
    "scrum master":           ["scrum master", "agile", "scrum", "gestion de projet", "chef de projet"],
    "qa engineer":            ["testeur", "qualité", "qa", "test", "assurance qualité"],
    "test engineer":          ["testeur", "qualité", "qa", "test", "assurance qualité"],
    "embedded developer":     ["développeur embarqué", "embarqué", "c", "c++", "microcontroleur", "iot"],
    "embedded systems":       ["systèmes embarqués", "embarqué", "c", "c++", "microcontroleur"],
}


# ── Détection et filtrage par lieu ───────────────────────────────────────────

# Noms de villes uniquement (pour le filtre dur post-recherche)
CITY_KEYWORDS = {
    "nabeul", "tunis", "sousse", "sfax", "monastir", "mahdia", "bizerte",
    "djerba", "jerba", "ariana", "la marsa", "hammamet", "gabes",
    "gafsa", "kairouan", "beja", "jendouba", "zaghouan", "tozeur",
    "medenine", "tataouine", "kasserine", "siliana", "kebili", "sidi bouzid",
}

REMOTE_KEYWORDS = {"remote", "teletravail", "a distance", "en ligne", "distanciel"}

# Ensemble complet pour détecter qu'une requête est de type "lieu"
LOCATION_QUERY_TRIGGERS = CITY_KEYWORDS | REMOTE_KEYWORDS | {
    "tunisie", "tunisia",
    "lieu", "ville", "location", "based in", "situe", "localise",
    "region", "gouvernorat",
}


def is_location_query(prompt_norm: str) -> bool:
    """Retourne True si la requête contient un indicateur de lieu."""
    return any(loc in prompt_norm for loc in LOCATION_QUERY_TRIGGERS)


def extract_search_cities(prompt_norm: str) -> List[str]:
    """Extrait les noms de villes spécifiques présents dans la requête."""
    return [city for city in CITY_KEYWORDS if city in prompt_norm]


def is_remote_query(prompt_norm: str) -> bool:
    """Retourne True si la requête demande du télétravail/remote."""
    return any(r in prompt_norm for r in REMOTE_KEYWORDS)


# ── Filtrage multi-critères (type de contrat + compétences) ──────────────────

# Variantes normalisées de chaque type de contrat
JOB_TYPE_MAP = {
    "part time":     ["part time", "part-time", "temps partiel", "mi-temps", "partiel"],
    "part-time":     ["part time", "part-time", "temps partiel", "mi-temps"],
    "temps partiel": ["part time", "part-time", "temps partiel", "mi-temps"],
    "full time":     ["full time", "full-time", "temps plein", "plein temps"],
    "full-time":     ["full time", "full-time", "temps plein"],
    "temps plein":   ["full time", "full-time", "temps plein"],
    "freelance":     ["freelance", "independant", "mission freelance"],
    "cdi":           ["cdi", "contrat indefini", "permanent"],
    "cdd":           ["cdd", "contrat determine"],
    "stage":         ["stage", "internship", "stagiaire"],
    "internship":    ["stage", "internship", "stagiaire"],
    "alternance":    ["alternance", "apprentissage"],
    "remote":        ["remote", "teletravail", "a distance", "distanciel", "en ligne"],
    "teletravail":   ["remote", "teletravail", "a distance", "distanciel"],
}

# ── Détection de requêtes hors-sujet ─────────────────────────────────────────
#
# Logique en 2 niveaux :
#   1. La requête doit contenir AU MOINS UN mot TECHNIQUE IT (technologie,
#      langage, métier IT précis, ville, type de contrat...).
#   2. Les mots génériques comme "freelancer", "mission", "poste" seuls
#      ne suffisent PAS — ils ne prouvent pas que la recherche est IT.
#
# Ex: "je veux un freelancer qui maîtrise le biceps"
#   → "freelancer" seul : non-technique → REJETÉ ✓
# Ex: "je cherche un développeur Angular à Tunis"
#   → "developpeur" + "angular" + "tunis" : techniques → ACCEPTÉ ✓

# Mots techniques stricts — suffisants seuls pour valider la requête
_TECH_KEYWORDS = (
    set(DOMAIN_EXPANSION.keys()) |   # angular, react, java, python, docker...
    set(JOB_TYPE_MAP.keys()) |        # full time, freelance, stage, cdi...
    CITY_KEYWORDS | REMOTE_KEYWORDS | # nabeul, tunis, sousse... / remote, teletravail
    {
        # Métiers IT précis (pas "manager" ou "gestionnaire" seuls)
        "developpeur", "developer", "ingenieur", "engineer",
        "technicien", "programmeur", "analyste", "analyst",
        "administrateur", "testeur", "tester", "architecte",
        "devops", "sysadmin", "scrum",
        # Domaine IT non-ambigu
        "informatique", "logiciel", "software", "hardware",
        "web", "application", "code", "api", "microservice",
        "framework", "librairie", "database", "serveur", "server",
        "cloud", "linux", "github", "gitlab", "programmation",
        "developpement", "development", "coding",
        # Données / IA
        "intelligence", "artificielle", "machine", "learning",
        "statistiques", "python", "nosql",
        # Design IT
        "design", "maquette", "prototype", "wireframe", "figma",
        # Contexte
        "tunisie", "tunisia", "teletravail",
        # Niveaux d'expérience (valides si accompagnés d'autre chose,
        # mais ne suffisent pas seuls — traités en bonus uniquement)
    }
)

# Mots qui NE SUFFISENT PAS seuls (trop génériques)
# Si la requête ne contient QUE ces mots → hors-sujet
_WEAK_KEYWORDS = {
    "freelancer", "freelance", "mission", "poste", "emploi", "offre",
    "profil", "recrutement", "embauche", "candidat", "contrat",
    "junior", "senior", "confirme", "experimente", "consultant",
    "stage", "alternance", "internship", "competence", "skill",
    "experience", "projet", "travail", "job",
}

# Precompile word-boundary patterns
_TECH_PATTERNS = [
    re.compile(r'(?<![a-z])' + re.escape(kw) + r'(?![a-z])')
    for kw in _TECH_KEYWORDS
]
_WEAK_PATTERNS = [
    re.compile(r'(?<![a-z])' + re.escape(kw) + r'(?![a-z])')
    for kw in _WEAK_KEYWORDS
]


def is_off_topic(prompt_norm: str) -> bool:
    """
    Retourne True si la requête est hors-sujet IT/freelance.

    Règle :
    - La requête est valide si elle contient au moins un mot TECHNIQUE IT.
    - Les mots génériques ("freelancer", "mission", "poste"...) seuls
      ne suffisent pas — la requête doit aussi contenir un mot technique.
    """
    has_tech  = any(p.search(prompt_norm) for p in _TECH_PATTERNS)
    has_weak  = any(p.search(prompt_norm) for p in _WEAK_PATTERNS)

    if has_tech:
        return False   # contient un vrai mot technique → valide

    if has_weak:
        # Contient seulement des mots génériques ("freelancer biceps") → hors-sujet
        return True

    return True  # aucun mot connu → hors-sujet


# Clés de DOMAIN_EXPANSION à exclure du filtrage compétences (non-tech)
_NON_SKILL_KEYS = (
    CITY_KEYWORDS | REMOTE_KEYWORDS |
    {"tunisie", "tunisia", "senior", "junior", "confirme", "experimente", "debutant",
     "developer", "distance", "a distance", "teletravail", "remote",
     "lieu", "ville", "location", "based in", "situe", "localise",
     "region", "gouvernorat", "en ligne", "distanciel"}
)


def extract_query_filters(prompt_norm: str) -> dict:
    """
    Extrait les filtres de la requête :
    - job_types : types de contrat détectés (ex: ["part time"])
    - skills    : compétences tech détectées (ex: ["java", "angular"])
    """
    job_types = [jt for jt in JOB_TYPE_MAP if jt in prompt_norm]
    skills = [kw for kw in DOMAIN_EXPANSION if kw in prompt_norm and kw not in _NON_SKILL_KEYS]
    return {"job_types": list(dict.fromkeys(job_types)), "skills": list(dict.fromkeys(skills))}


def mission_matches_filter(meta: dict, filter_key: str) -> bool:
    """Vérifie si une mission matche un filtre donné (type de contrat ou compétence)."""
    content_norm = meta.get("content_norm", "")
    type_norm = meta.get("type_norm", "")

    # Filtre type de contrat
    if filter_key in JOB_TYPE_MAP:
        variants = [normalize(v) for v in JOB_TYPE_MAP[filter_key]]
        return any(v in type_norm for v in variants)

    # Filtre compétence : cherche le keyword et ses synonymes dans le contenu de la mission
    if filter_key in content_norm:
        return True
    for syn in DOMAIN_EXPANSION.get(filter_key, []):
        if normalize(syn) in content_norm:
            return True
    return False


def expand_prompt(prompt: str) -> str:
    """Enrichit le prompt — insensible aux accents et variantes."""
    prompt_norm = normalize(prompt)   # sans accents, minuscules
    extra_terms = []

    for keyword, synonyms in DOMAIN_EXPANSION.items():
        if keyword in prompt_norm:
            extra_terms.extend(synonyms)

    if extra_terms:
        unique_extra = list(dict.fromkeys(extra_terms))
        expanded = f"{prompt}. {'. '.join(unique_extra[:20])}."
        print(f"[AI] Prompt expanded: '{prompt}' → +{len(unique_extra)} terms")
        return expanded

    return prompt

app = FastAPI(title="WorkLink AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────
print("[AI] Loading sentence-transformers model...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("[AI] Model loaded.")

DIMENSION = 384

# Index missions
index = faiss.IndexFlatIP(DIMENSION)
mission_ids: List[str] = []
mission_locations: dict = {}   # mission_id → location normalisée (filtre dur lieu)
mission_metadata: dict = {}    # mission_id → {type_norm, content_norm} (filtre multi-critères)

# Index freelancers
freelancer_index = faiss.IndexFlatIP(DIMENSION)
freelancer_ids: List[str] = []
freelancer_texts: dict = {}       # id → texte pour pouvoir ré-indexer
freelancer_locations: dict = {}   # freelancer_id → location normalisée (pour filtre dur)

SPRING_BOOT_URL = os.getenv("SPRING_BOOT_URL", "http://localhost:8080")


# ── Modèles Pydantic ──────────────────────────────────────────────────────────
class MissionIndexRequest(BaseModel):
    id: str
    jobTitle: Optional[str] = ""
    field: Optional[str] = ""
    description: Optional[str] = ""
    requiredSkills: Optional[str] = ""
    technicalEnvironment: Optional[str] = ""
    missionBusinessSector: Optional[str] = ""
    speciality: Optional[str] = ""
    location: Optional[str] = ""
    missionType: Optional[str] = ""


class FreelancerIndexRequest(BaseModel):
    id: str
    currentPosition: Optional[str] = ""
    skills: Optional[List[str]] = []
    bio: Optional[str] = ""
    profileTypes: Optional[List[str]] = []
    yearsOfExperience: Optional[int] = None
    location: Optional[str] = ""
    city: Optional[str] = ""
    country: Optional[str] = ""


class SearchRequest(BaseModel):
    prompt: str
    top_k: int = 10


class SearchResult(BaseModel):
    mission_id: str
    score: float


class FreelancerSearchResult(BaseModel):
    freelancer_id: str
    score: float


# ── Utilitaires ───────────────────────────────────────────────────────────────
def strip_html(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'<[^>]+>', ' ', text).strip()


def mission_to_text(m: MissionIndexRequest) -> str:
    title = m.jobTitle or ""
    skills = strip_html(m.requiredSkills or "")
    tech = strip_html(m.technicalEnvironment or "")
    description = strip_html(m.description or "")
    field = m.field or ""
    speciality = m.speciality or ""
    location = m.location or ""

    # Déduire le type de poste depuis le titre + compétences (normalisé sans accents)
    title_lower = normalize(title)
    skills_lower = normalize(skills + " " + tech)

    context_tags = []

    # Tags frontend — uniquement si outils de dev frontend présents dans titre ou skills
    frontend_dev_kw = ["angular", "react", "vue", "javascript", "typescript", "frontend", "front-end", "nextjs", "rxjs"]
    if any(k in title_lower or k in skills_lower for k in frontend_dev_kw):
        context_tags.append("développeur frontend développeur web javascript typescript composant SPA code développement programmation")

    # Tags backend
    backend_kw = ["java", "spring", "node", "django", "flask", "fastapi", "backend", "back-end", "api rest", "microservice", "hibernate", "jpa"]
    if any(k in title_lower or k in skills_lower for k in backend_kw):
        context_tags.append("développeur backend serveur api microservices base de données programmation code")

    # Tags AI/Data
    ai_kw = ["machine learning", "deep learning", "data science", "nlp", "ia", "ai", "tensorflow", "pytorch"]
    if any(k in title_lower or k in skills_lower for k in ai_kw):
        context_tags.append("intelligence artificielle data science machine learning modèle prédictif python")

    # Tags UX/Design — uniquement si figma ou ux/ui dans le TITRE (pas skills)
    ux_title_kw = ["ux", "ui", "figma", "designer", "design graphique", "product designer"]
    if any(k in title_lower for k in ux_title_kw):
        context_tags.append("designer ux ui maquette prototype figma wireframe expérience utilisateur graphisme non-développeur")

    # Tags Mobile
    mobile_kw = ["flutter", "android", "ios", "react native", "kotlin", "swift", "mobile"]
    if any(k in title_lower or k in skills_lower for k in mobile_kw):
        context_tags.append("développeur mobile application android ios programmation")

    context = ". ".join(context_tags)

    mission_type = m.missionType or ""

    return (
        f"{title}. {title}. {title}. "
        f"Poste: {title}. "
        f"Type de contrat: {mission_type}. {mission_type}. "
        f"Domaine: {field}. Spécialité: {speciality}. "
        f"Lieu: {location}. {location}. {location}. {location}. "
        f"Compétences requises: {skills}. "
        f"Environnement technique: {tech}. "
        f"{context}. "
        f"{description}."
    )


def add_to_index(mission_id: str, text: str, location: str = "", mission_type: str = "", content: str = ""):
    vec = model.encode(text, normalize_embeddings=True)
    vec = np.array([vec], dtype=np.float32)
    index.add(vec)
    mission_ids.append(mission_id)
    mission_locations[mission_id] = normalize(location)
    mission_metadata[mission_id] = {
        "type_norm":    normalize(mission_type),
        "content_norm": normalize(content),
    }


# ── Utilitaires Freelancer ─────────────────────────────────────────────────────
PROFILE_TYPE_TEXT = {
    "STUDIES_DEVELOPMENT":        "développeur développement logiciel études programmation code",
    "BI_DATA":                    "data analyst business intelligence big data analyse données",
    "NEW_TECHNOLOGIES":           "nouvelles technologies innovation tech digital",
    "SYSTEMS_INFRASTRUCTURE":     "système infrastructure réseau administration système",
    "TESTING_QUALITY":            "qualité test QA assurance qualité testeur",
    "ERP_CRM":                    "ERP CRM gestion entreprise SAP Oracle",
    "BUSINESS_CONSULTING":        "consultant business conseil stratégie management",
    "INDUSTRIAL_IT_ELECTRONICS":  "industriel électronique IT embarqué",
    "SYSTEM_RESOURCES":           "ressources système administration support",
    "OFFICE_SUPPORT":             "support bureautique assistance helpdesk",
}


def freelancer_to_text(f: FreelancerIndexRequest) -> str:
    position = f.currentPosition or ""
    skills_list = f.skills or []
    skills_csv = ", ".join(skills_list)
    skills_expanded = " ".join(skills_list)
    bio = f.bio or ""
    exp = f.yearsOfExperience
    location = f.location or ""
    city = f.city or ""
    country = f.country or ""
    # Construire le bloc localisation (dédupliqué)
    location_parts = list(dict.fromkeys(p for p in [city, location, country] if p))
    location_block = ". ".join(location_parts)

    # Normaliser sans accents pour la détection des mots-clés
    position_lower = normalize(position)
    skills_lower = normalize(skills_csv)

    # ── Expérience en texte lisible ──────────────────────────────────────────
    exp_text = ""
    if exp is not None:
        if exp == 0:
            exp_text = "0 an d'expérience junior débutant"
        elif exp == 1:
            exp_text = "1 an d'expérience junior"
        elif exp <= 2:
            exp_text = f"{exp} ans d'expérience junior débutant entry level"
        elif exp <= 5:
            exp_text = f"{exp} ans d'expérience confirmé intermédiaire"
        else:
            exp_text = f"{exp} ans d'expérience senior expérimenté expert"

    # ── Context tags (détection normalisée sans accents) ─────────────────────
    context_tags = []

    frontend_kw = ["angular", "react", "vue", "javascript", "typescript", "frontend", "front-end", "nextjs", "rxjs", "svelte"]
    if any(k in position_lower or k in skills_lower for k in frontend_kw):
        context_tags.append("développeur frontend développeur web javascript typescript angular react vue composant SPA programmation code interface web")

    backend_kw = ["java", "spring", "node", "django", "flask", "fastapi", "backend", "back-end", "api", "microservice", "hibernate", "php", "laravel", "symfony", "dotnet", "csharp"]
    if any(k in position_lower or k in skills_lower for k in backend_kw):
        context_tags.append("développeur backend serveur api REST microservices base de données programmation code spring boot java node")

    ai_kw = ["machine learning", "deep learning", "data science", "nlp", "ai", "ia", "tensorflow", "pytorch", "scikit", "keras"]
    if any(k in position_lower or k in skills_lower for k in ai_kw):
        context_tags.append("intelligence artificielle data science machine learning modèle prédictif python analyse données big data")

    ux_kw = ["ux", "ui", "figma", "designer", "design graphique", "product designer", "maquette"]
    if any(k in position_lower for k in ux_kw):
        context_tags.append("designer ux ui maquette prototype figma wireframe graphisme expérience utilisateur")

    mobile_kw = ["flutter", "android", "ios", "react native", "kotlin", "swift", "mobile", "dart"]
    if any(k in position_lower or k in skills_lower for k in mobile_kw):
        context_tags.append("développeur mobile application android ios flutter kotlin swift programmation")

    devops_kw = ["docker", "kubernetes", "devops", "ci/cd", "jenkins", "aws", "azure", "terraform", "ansible"]
    if any(k in position_lower or k in skills_lower for k in devops_kw):
        context_tags.append("devops cloud infrastructure docker kubernetes déploiement pipeline aws azure")

    data_kw = ["sql", "mongodb", "postgresql", "mysql", "oracle", "power bi", "tableau", "bi", "data analyst"]
    if any(k in position_lower or k in skills_lower for k in data_kw):
        context_tags.append("data analyst base de données sql business intelligence analyse données rapports")

    for pt in (f.profileTypes or []):
        pt_text = PROFILE_TYPE_TEXT.get(pt, "")
        if pt_text:
            context_tags.append(pt_text)

    context = ". ".join(context_tags)

    # Poste répété 5x pour qu'il domine le vecteur
    position_block = f"{position}. " * 5
    # Skills répétés 2x : une fois en liste, une fois individuellement
    skills_block = f"Compétences: {skills_csv}. {skills_expanded}. {skills_expanded}. "

    return (
        f"{position_block}"
        f"Poste: {position}. "
        f"Localisation: {location_block}. {location_block}. {location_block}. {location_block}. "
        f"{skills_block}"
        f"Expérience: {exp_text}. "
        f"{context}. "
        f"Bio: {bio}."
    )


def add_freelancer_to_index(freelancer_id: str, text: str, location: str = ""):
    vec = model.encode(text, normalize_embeddings=True)
    vec = np.array([vec], dtype=np.float32)
    freelancer_index.add(vec)
    freelancer_ids.append(freelancer_id)
    freelancer_texts[freelancer_id] = text
    freelancer_locations[freelancer_id] = normalize(location)


def rebuild_freelancer_index():
    """Reconstruit l'index FAISS freelancer depuis freelancer_texts."""
    global freelancer_index
    freelancer_index = faiss.IndexFlatIP(DIMENSION)
    for fid in freelancer_ids:
        text = freelancer_texts.get(fid, "")
        vec = model.encode(text, normalize_embeddings=True)
        vec = np.array([vec], dtype=np.float32)
        freelancer_index.add(vec)


# ── Startup : charger missions + freelancers depuis Spring Boot ───────────────
@app.on_event("startup")
async def startup():
    async with httpx.AsyncClient() as client:
        # Missions
        try:
            resp = await client.get(f"{SPRING_BOOT_URL}/api/missions/public/all", timeout=30.0)
            if resp.status_code == 200:
                for m in resp.json():
                    req = MissionIndexRequest(
                        id=m.get("id", ""),
                        jobTitle=m.get("jobTitle"),
                        field=m.get("field"),
                        description=m.get("description"),
                        requiredSkills=m.get("requiredSkills"),
                        technicalEnvironment=m.get("technicalEnvironment"),
                        missionBusinessSector=m.get("missionBusinessSector"),
                        speciality=m.get("speciality"),
                        location=m.get("location"),
                        missionType=m.get("missionType"),
                    )
                    if req.id and req.id not in mission_ids:
                        add_to_index(
    req.id, mission_to_text(req), req.location or "", req.missionType or "",
    f"{req.jobTitle} {req.requiredSkills} {req.technicalEnvironment} {req.field} {req.speciality}"
)
                print(f"[AI] Indexed {len(mission_ids)} missions at startup")
        except Exception as e:
            print(f"[AI] Warning: could not load missions at startup: {e}")

        # Freelancers
        try:
            resp = await client.get(f"{SPRING_BOOT_URL}/api/freelancer/public/all", timeout=30.0)
            if resp.status_code == 200:
                for f in resp.json():
                    req = FreelancerIndexRequest(
                        id=f.get("id", ""),
                        currentPosition=f.get("currentPosition"),
                        skills=f.get("skills") or [],
                        bio=f.get("bio"),
                        profileTypes=f.get("profileTypes") or [],
                        yearsOfExperience=f.get("yearsOfExperience"),
                        location=f.get("location"),
                        city=f.get("city"),
                        country=f.get("country"),
                    )
                    if req.id and req.id not in freelancer_ids:
                        loc = ", ".join(filter(None, [req.city or "", req.location or "", req.country or ""]))
                        add_freelancer_to_index(req.id, freelancer_to_text(req), loc)
                print(f"[AI] Indexed {len(freelancer_ids)} freelancers at startup")
        except Exception as e:
            print(f"[AI] Warning: could not load freelancers at startup: {e}")
            print("[AI] Freelancers will be indexed via /index-freelancer")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/index-mission")
def index_mission(req: MissionIndexRequest):
    """Appelé par Spring Boot après création d'une mission."""
    if req.id in mission_ids:
        return {"status": "already_indexed", "id": req.id, "total": len(mission_ids)}
    add_to_index(
    req.id, mission_to_text(req), req.location or "", req.missionType or "",
    f"{req.jobTitle} {req.requiredSkills} {req.technicalEnvironment} {req.field} {req.speciality}"
)
    print(f"[AI] Mission indexed: {req.id} | Total: {len(mission_ids)}")
    return {"status": "indexed", "id": req.id, "total": len(mission_ids)}


@app.post("/search", response_model=List[SearchResult])
def search_missions(req: SearchRequest):
    """Recherche sémantique par prompt utilisateur."""
    if len(mission_ids) == 0:
        return []

    prompt_norm = normalize(req.prompt)

    # Rejeter les requêtes hors domaine IT/freelance
    if is_off_topic(prompt_norm):
        print(f"[AI] OFF-TOPIC rejected: '{req.prompt}'")
        raise HTTPException(
            status_code=422,
            detail="Votre recherche semble hors du contexte de la plateforme. "
                   "Veuillez rechercher des compétences IT, des postes ou des technologies "
                   "(ex: développeur Angular, Java senior, Data Scientist à Tunis...)."
        )

    location_query = is_location_query(prompt_norm)

    enriched_prompt = expand_prompt(req.prompt)
    query_vec = model.encode(enriched_prompt, normalize_embeddings=True)
    query_vec = np.array([query_vec], dtype=np.float32)

    k = min(req.top_k, len(mission_ids))
    scores, indices = index.search(query_vec, k)

    # Seuil absolu minimum — plus bas pour les requêtes de lieu
    ABSOLUTE_MIN = 0.18 if location_query else 0.22
    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and float(score) > ABSOLUTE_MIN:
            candidates.append((mission_ids[idx], float(score)))

    if not candidates:
        print(f"[AI] Search: '{req.prompt}' → 0 results")
        return []

    # ── Filtre dur par ville si des villes spécifiques sont détectées ───────────
    detected_cities = extract_search_cities(prompt_norm)
    if detected_cities:
        candidates = [
            (mid, s) for mid, s in candidates
            if any(city in mission_locations.get(mid, "") for city in detected_cities)
        ]
        if not candidates:
            print(f"[AI] Search [CITY FILTER]: '{req.prompt}' → 0 results after city filter {detected_cities}")
            return []

    best_score = candidates[0][1]
    dynamic_ratio = 0.50 if location_query else 0.80
    dynamic_threshold = max(ABSOLUTE_MIN, best_score * dynamic_ratio)

    # Appliquer le seuil dynamique
    candidates = [(mid, s) for mid, s in candidates if s >= dynamic_threshold]
    if not candidates:
        return []

    # ── Re-classement multi-filtres (type de contrat + compétences) ──────────
    query_filters = extract_query_filters(prompt_norm)
    all_filters = query_filters["job_types"] + query_filters["skills"]

    if len(all_filters) >= 2:
        # Compter combien de filtres chaque mission satisfait
        full_match, partial_match = [], []
        for mid, s in candidates:
            meta = mission_metadata.get(mid, {})
            matched = sum(1 for f in all_filters if mission_matches_filter(meta, f))
            if matched >= len(all_filters):
                full_match.append((mid, s))
            elif matched > 0:
                partial_match.append((mid, s))
        # Full match d'abord, puis partial (chacun trié par score décroissant)
        candidates = full_match + partial_match
        print(f"[AI] Multi-filter [{all_filters}]: {len(full_match)} full, {len(partial_match)} partial")

    seen_ids = set()
    results = []
    for mid, s in candidates:
        if mid not in seen_ids:
            seen_ids.add(mid)
            results.append(SearchResult(mission_id=mid, score=round(s * 100, 1)))

    mode = f"CITY={detected_cities}" if detected_cities else ("LOCATION" if location_query else "SEMANTIC")
    print(f"[AI] Search [{mode}]: '{req.prompt}' → {len(results)} results "
          f"(best={round(best_score*100,1)}%, threshold={round(dynamic_threshold*100,1)}%)")
    return results


@app.post("/index-freelancer")
def index_freelancer(req: FreelancerIndexRequest):
    """Appelé par Spring Boot après création/mise à jour d'un freelancer."""
    text = freelancer_to_text(req)
    loc = ", ".join(filter(None, [req.city or "", req.location or "", req.country or ""]))
    if req.id in freelancer_ids:
        # Mise à jour : on met à jour le texte, la location et on reconstruit l'index
        freelancer_texts[req.id] = text
        freelancer_locations[req.id] = normalize(loc)
        rebuild_freelancer_index()
        print(f"[AI] Freelancer re-indexed: {req.id} | Total: {len(freelancer_ids)}")
        return {"status": "re_indexed", "id": req.id, "total": len(freelancer_ids)}
    add_freelancer_to_index(req.id, text, loc)
    print(f"[AI] Freelancer indexed: {req.id} | Total: {len(freelancer_ids)}")
    return {"status": "indexed", "id": req.id, "total": len(freelancer_ids)}


@app.post("/search-freelancers", response_model=List[FreelancerSearchResult])
def search_freelancers(req: SearchRequest):
    """Recherche sémantique de freelancers par prompt entreprise."""
    if len(freelancer_ids) == 0:
        return []

    prompt_norm = normalize(req.prompt)

    # Rejeter les requêtes hors domaine IT/freelance
    if is_off_topic(prompt_norm):
        print(f"[AI] OFF-TOPIC rejected: '{req.prompt}'")
        raise HTTPException(
            status_code=422,
            detail="Votre recherche semble hors du contexte de la plateforme. "
                   "Veuillez rechercher des compétences IT, des postes ou des technologies "
                   "(ex: développeur React, ingénieur DevOps, Data Analyst junior...)."
        )

    location_query = is_location_query(prompt_norm)

    enriched_prompt = expand_prompt(req.prompt)
    query_vec = model.encode(enriched_prompt, normalize_embeddings=True)
    query_vec = np.array([query_vec], dtype=np.float32)

    k = min(req.top_k, len(freelancer_ids))
    scores, indices = freelancer_index.search(query_vec, k)

    # Seuil absolu minimum — plus bas pour les requêtes de lieu
    ABSOLUTE_MIN = 0.20 if location_query else 0.28
    candidates = [
        (freelancer_ids[idx], float(score))
        for score, idx in zip(scores[0], indices[0])
        if idx >= 0 and float(score) >= ABSOLUTE_MIN
    ]

    if not candidates:
        print(f"[AI] Freelancer search: '{req.prompt}' → 0 results")
        return []

    # ── Filtre dur par ville si des villes spécifiques sont détectées ───────────
    detected_cities = extract_search_cities(prompt_norm)
    if detected_cities:
        candidates = [
            (fid, s) for fid, s in candidates
            if any(city in freelancer_locations.get(fid, "") for city in detected_cities)
        ]
        if not candidates:
            print(f"[AI] Freelancer search [CITY FILTER]: '{req.prompt}' → 0 results after city filter {detected_cities}")
            return []

    best_score = candidates[0][1]
    # Pour une recherche par lieu : seuil dynamique souple (50%)
    # Pour une recherche normale : seuil modéré (60%)
    dynamic_ratio = 0.50 if location_query else 0.60
    dynamic_threshold = max(ABSOLUTE_MIN, best_score * dynamic_ratio)

    seen_fids = set()
    results = []
    for fid, s in candidates:
        if s >= dynamic_threshold and fid not in seen_fids:
            seen_fids.add(fid)
            results.append(FreelancerSearchResult(freelancer_id=fid, score=round(s * 100, 1)))

    mode = f"CITY={detected_cities}" if detected_cities else ("LOCATION" if location_query else "SEMANTIC")
    print(f"[AI] Freelancer search [{mode}]: '{req.prompt}' → {len(results)} results "
          f"(best={round(best_score*100,1)}%, threshold={round(dynamic_threshold*100,1)}%)")
    return results


@app.post("/extract-cv")
async def extract_cv(file: UploadFile = File(...)):
    """Extract structured CV data from a PDF using Ollama llama3.2."""
    try:
        content = await file.read()

        # Extract raw text from PDF
        text = ""
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        # Truncate to avoid exceeding llama3.2 context limit
        if len(text) > 8000:
            text = text[:8000]

        print(f"[AI-EXTRACT] Extracted {len(text)} chars from PDF")

        prompt = f"""You are a CV data extractor. Your ONLY job is to copy information that EXPLICITLY appears in the CV text below.
STRICT RULES — violation is not allowed:
1. NEVER invent, infer, guess, complete, or add ANY information not present word-for-word in the CV.
2. If a field is absent from the CV, return null (for strings/objects) or [] (for arrays). NEVER fabricate a value.
3. Copy text EXACTLY as written in the CV — do not paraphrase, summarize, or rewrite.
4. Return ONLY a valid JSON object, no text before or after.

=== BIO / PROFILE SUMMARY (CRITICAL) ===
Look for a short personal introduction section. It may be labeled:
French: Profil, À propos, A propos, Résumé, Resume, Présentation, Présentation professionnelle,
  Synthèse, Synthèse professionnelle, Objectif, Objectif professionnel, Qui suis-je,
  Introduction, Pitch, Accroche, Accroche professionnelle, Mon profil, Profil professionnel,
  Résumé de profil, Description, Aperçu
English: Profile, About, About me, Summary, Professional Summary, Career Summary,
  Objective, Career Objective, Overview, Introduction, Personal Statement, Executive Summary
If you find such a section, copy its FULL EXACT text as the bio value — do NOT shorten or summarize it.
If no such section exists, return null for bio.
IMPORTANT: Do NOT confuse bio with work experience descriptions. Bio is always at the TOP of the CV.

=== WORK EXPERIENCE vs PROJECTS (CRITICAL) ===
WORK EXPERIENCE = paid professional jobs at a company or client.
  Indicators: company name, job title, dates of employment, salary, contract type.
  Labels: Expérience professionnelle, Expériences, Experience, Work Experience, Employment, Parcours professionnel

PROJECTS = personal, academic, freelance or open-source projects that are NOT paid employment.
  Indicators: project name, technologies used, GitHub link, description of what was built.
  Labels: Projets, Projets personnels, Projets académiques, Réalisations, Portfolio,
          Projects, Personal Projects, Academic Projects, Side Projects, Open Source
RULE: If an entry has a PROJECT NAME (not a job title) and/or a technology list but no company employer,
put it in "projects", NOT in "workExperience".
If projects section is empty or absent, return empty array [].

=== SKILLS EXTRACTION (EXHAUSTIVE — CRITICAL) ===
You MUST extract EVERY SINGLE skill listed in the CV. Do NOT stop early. Do NOT truncate.
Search through ALL of these possible section labels:
French: Compétences, Compétences techniques, Compétences informatiques, Technologies,
  Technologies maîtrisées, Savoir-faire, Outils, Langages de programmation, Frameworks,
  Environnement technique, Outils & Technologies, Stack technique, Compétences clés,
  Logiciels, Maîtrise technique, Compétences métier, Hard skills, Soft skills,
  Connaissances, Acquis techniques, Technologies utilisées, Langages & Frameworks
English: Skills, Technical Skills, Technologies, Tools, Tech Stack, Key Skills,
  Core Competencies, Programming Languages, Frameworks, Libraries, Platforms,
  Hard Skills, Soft Skills, Expertise, Proficiencies, Technical Expertise
Also extract skills mentioned inside work experience descriptions and project descriptions.
Return each skill as a separate string EXACTLY as written in the CV.
Do NOT add skills that are not explicitly written in the CV.

=== CERTIFICATIONS vs EDUCATION (CRITICAL) ===
CERTIFICATIONS are professional credentials issued by tech or industry bodies:
  Examples: AWS, Azure, Google Cloud, Cisco CCNA, Oracle, Microsoft, Scrum Master, PMP,
            TOEFL, IELTS, DELF, DALF, CompTIA, CFA, PMI, Coursera, Udemy certificates
EDUCATION is academic degrees from schools/universities:
  Examples: Licence, Master, Doctorat, Ingénieur, BTS, DUT, Baccalauréat, Bachelor, MBA,
            Diplôme national → these go ONLY in education, NEVER in certifications.
If certifications section is empty or absent, return empty array [].

=== DATE RULES ===
- Format: YYYY-MM-DD. If only year known: YYYY-01-01. If year+month: YYYY-MM-01.
- French months: Janvier=01, Février=02, Mars=03, Avril=04, Mai=05, Juin=06,
  Juillet=07, Août=08, Septembre=09, Octobre=10, Novembre=11, Décembre=12
- "Présent" / "Aujourd'hui" / "Current" / "En cours" / "Now" → isCurrent: true, endDate: null
- If a date is missing or unknown → use null, never use empty string "".

=== LANGUAGE MAPPING (CRITICAL) ===
Scan the ENTIRE CV for any section labeled: Langues, Languages, Compétences linguistiques,
Langues parlées, Spoken languages, or similar.
Extract EVERY language found — do not stop after 2, list ALL of them.
Use these exact mappings:
- Français / French / FR → "FRENCH"
- Anglais / English / EN → "ENGLISH"
- Arabe / Arabic / AR / العربية → "ARABIC"
- Espagnol / Spanish → "SPANISH"
- Allemand / German / Deutsch → "GERMAN"
- Italien / Italian → "ITALIAN"
- Portugais / Portuguese → "PORTUGUESE"
- Chinois / Chinese / Mandarin → "CHINESE"
- Japonais / Japanese → "JAPANESE"
- Any other language → skip it
Allowed values ONLY: FRENCH, ENGLISH, ARABIC, SPANISH, GERMAN, ITALIAN, PORTUGUESE, CHINESE, JAPANESE

=== JSON STRUCTURE (return exactly this structure) ===
{{
  "bio": "full profile/summary text from CV top section, or null",
  "workExperience": [
    {{
      "jobTitle": "exact job title",
      "company": "company or client name",
      "startDate": "YYYY-MM-DD or null",
      "endDate": "YYYY-MM-DD or null",
      "isCurrent": false,
      "description": "responsibilities and achievements"
    }}
  ],
  "projects": [
    {{
      "name": "project name",
      "description": "what was built, goals, outcomes",
      "technologies": ["tech1", "tech2", "tech3"],
      "url": "GitHub or project URL, or null"
    }}
  ],
  "education": [
    {{
      "diploma": "exact degree/diploma name",
      "institution": "school or university name",
      "year": 2020,
      "description": "specialization or field of study, or empty string"
    }}
  ],
  "certifications": [
    {{
      "name": "exact certification name",
      "issuer": "issuing organization",
      "issueDate": "YYYY-MM-DD or null",
      "expiryDate": "YYYY-MM-DD or null"
    }}
  ],
  "skills": ["skill1", "skill2", "skill3", "... ALL skills without exception"],
  "languages": ["FRENCH", "ENGLISH"]
}}

FINAL REMINDER:
- NEVER invent or guess any data — if it is not written in the CV, return null or []
- bio: copy EXACTLY the text from the Profil / À propos / Summary section at the TOP
- projects: any named project that is NOT a paid job goes here, NOT in workExperience
- skills: copy EVERY skill EXACTLY as written — do not add, do not skip
- dates: always null (never "") when not explicitly written in the CV

CV TEXT:
{text}"""

        response = ollama.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': prompt}],
            format='json',
            options={
                'temperature': 0,
                'seed': 42,
                'num_ctx': 4096,
            }
        )

        # Compatible avec ollama >= 0.2.x (objet Pydantic) et < 0.2.x (dict)
        try:
            raw = response.message.content.strip()
        except AttributeError:
            raw = response['message']['content'].strip()

        print(f"[AI-EXTRACT] Raw LLM response (first 300 chars): {raw[:300]}")

        # Find the JSON object in the response (LLM sometimes adds extra text)
        start = raw.find('{')
        end = raw.rfind('}') + 1

        if start == -1 or end <= start:
            print(f"[AI-EXTRACT] No JSON in response: {raw[:300]}")
            raise HTTPException(status_code=500, detail="AI could not produce structured data from this CV")

        json_str = raw[start:end]
        result = json.loads(json_str)

        # ── Post-validation : nettoyage des données extraites ─────────────────

        # 0. Sanitisation des dates (évite les formats invalides → 400 côté Spring Boot)
        def sanitize_date(value) -> str | None:
            """Normalise une date extraite par le LLM en YYYY-MM-DD ou None."""
            if value is None:
                return None
            text = str(value).strip()
            if not text or text.lower() in ("null", "n/a", "unknown", "present",
                                             "présent", "en cours", "aujourd'hui"):
                return None
            # Déjà au format YYYY-MM-DD
            import re as _re
            if _re.match(r'^\d{4}-\d{2}-\d{2}$', text):
                # Vérifier que le mois et le jour sont valides
                parts = text.split('-')
                month, day = int(parts[1]), int(parts[2])
                if month < 1: text = f"{parts[0]}-01-{parts[2]}"
                if month > 12: text = f"{parts[0]}-12-{parts[2]}"
                if day < 1: text = f"{parts[0]}-{parts[1]}-01"
                if day > 31: text = f"{parts[0]}-{parts[1]}-01"
                return text
            # Format YYYY-MM
            if _re.match(r'^\d{4}-\d{2}$', text):
                return f"{text}-01"
            # Année seule YYYY
            if _re.match(r'^\d{4}$', text):
                return f"{text}-01-01"
            # Format DD/MM/YYYY ou MM/YYYY
            if _re.match(r'^\d{2}/\d{2}/\d{4}$', text):
                parts = text.split('/')
                return f"{parts[2]}-{parts[1]}-{parts[0]}"
            if _re.match(r'^\d{2}/\d{4}$', text):
                parts = text.split('/')
                return f"{parts[1]}-{parts[0]}-01"
            # Impossible à parser → null (évite le 400)
            print(f"[AI-EXTRACT] Unrecognized date format: '{text}' → null")
            return None

        # Nettoyer les dates dans workExperience
        for exp in result.get("workExperience", []):
            exp["startDate"] = sanitize_date(exp.get("startDate"))
            exp["endDate"]   = sanitize_date(exp.get("endDate"))

        # Nettoyer les dates dans certifications
        for cert in result.get("certifications", []):
            cert["issueDate"]  = sanitize_date(cert.get("issueDate"))
            cert["expiryDate"] = sanitize_date(cert.get("expiryDate"))

        # 1. Filtrer les langues invalides (le LLM peut retourner "Français" au lieu de "FRENCH")
        VALID_LANGUAGES = {"FRENCH", "ENGLISH", "ARABIC", "SPANISH", "GERMAN",
                           "ITALIAN", "PORTUGUESE", "CHINESE", "JAPANESE", "OTHER"}
        LANGUAGE_FIX = {
            # French
            "FRANÇAIS": "FRENCH", "FRANCAIS": "FRENCH", "FRENCH": "FRENCH",
            "FR": "FRENCH", "LANGUE FRANÇAISE": "FRENCH",
            # English
            "ANGLAIS": "ENGLISH", "ENGLISH": "ENGLISH", "EN": "ENGLISH",
            "LANGUE ANGLAISE": "ENGLISH",
            # Arabic
            "ARABE": "ARABIC", "ARABIC": "ARABIC", "AR": "ARABIC",
            "LANGUE ARABE": "ARABIC", "ARABE DIALECTAL": "ARABIC",
            "ARABE CLASSIQUE": "ARABIC", "ARABE MODERNE": "ARABIC",
            # German
            "ALLEMAND": "GERMAN", "GERMAN": "GERMAN", "DEUTSCH": "GERMAN",
            "LANGUE ALLEMANDE": "GERMAN",
            # Spanish
            "ESPAGNOL": "SPANISH", "SPANISH": "SPANISH", "ESPAÑOL": "SPANISH",
            # Italian
            "ITALIEN": "ITALIAN", "ITALIAN": "ITALIAN",
            # Portuguese
            "PORTUGAIS": "PORTUGUESE", "PORTUGUESE": "PORTUGUESE",
            # Chinese
            "CHINOIS": "CHINESE", "CHINESE": "CHINESE", "MANDARIN": "CHINESE",
            "MANDARIN CHINESE": "CHINESE",
            # Japanese
            "JAPONAIS": "JAPANESE", "JAPANESE": "JAPANESE",
        }
        raw_langs = result.get("languages", [])
        cleaned_langs = []
        for lang in raw_langs:
            normalized = lang.upper().strip()
            fixed = LANGUAGE_FIX.get(normalized, normalized)
            if fixed in VALID_LANGUAGES and fixed not in cleaned_langs and fixed != "OTHER":
                cleaned_langs.append(fixed)
        result["languages"] = cleaned_langs

        # 2. Filtrer les certifications qui sont en réalité des diplômes académiques
        ACADEMIC_KEYWORDS = [
            "licence", "master", "doctorat", "ingénieur", "ingenieur", "bts", "dut",
            "baccalauréat", "baccalaureat", "bachelor", "mba", "iut", "diplôme",
            "diplome", "deug", "deust", "licence professionnelle", "magistère",
            "licence pro", "master pro", "master recherche"
        ]
        raw_certifs = result.get("certifications", [])
        cleaned_certifs = []
        for cert in raw_certifs:
            name_lower = cert.get("name", "").lower()
            if not any(kw in name_lower for kw in ACADEMIC_KEYWORDS):
                cleaned_certifs.append(cert)
            else:
                print(f"[AI-EXTRACT] Removed academic entry from certifications: {cert.get('name')}")
        result["certifications"] = cleaned_certifs

        # 3. Séparer les projets mal classés dans workExperience
        # Si une entrée workExperience n'a pas de "company" ou a une description de projet → déplacer
        PROJECT_INDICATORS = ["github", "gitlab", "projet", "project", "application", "app",
                               "portfolio", "open source", "personnel", "académique", "academic"]
        work_cleaned = []
        extra_projects = []
        for exp in result.get("workExperience", []):
            company = (exp.get("company") or "").strip()
            job_title = (exp.get("jobTitle") or "").strip().lower()
            desc = (exp.get("description") or "").lower()
            # Indicateur : pas de company ET titre ressemble à un projet
            if not company and any(ind in job_title or ind in desc for ind in PROJECT_INDICATORS):
                # Convertir en projet
                extra_projects.append({
                    "name": exp.get("jobTitle", ""),
                    "description": exp.get("description", ""),
                    "technologies": [],
                    "url": None,
                })
                print(f"[AI-EXTRACT] Moved from workExp to projects: {exp.get('jobTitle')}")
            else:
                work_cleaned.append(exp)
        result["workExperience"] = work_cleaned

        # Fusionner avec les projets déjà extraits
        existing_projects = result.get("projects", [])
        if not isinstance(existing_projects, list):
            existing_projects = []
        # Normaliser les projets (s'assurer que tous les champs existent)
        all_projects = []
        for proj in existing_projects + extra_projects:
            all_projects.append({
                "name":         proj.get("name", ""),
                "description":  proj.get("description", ""),
                "technologies": proj.get("technologies") if isinstance(proj.get("technologies"), list) else [],
                "url":          proj.get("url") or None,
            })
        result["projects"] = all_projects

        # 4. S'assurer que bio est null (pas "") si vide
        if not result.get("bio") or str(result.get("bio", "")).strip() == "":
            result["bio"] = None

        # 5. Dédupliquer les skills
        raw_skills = result.get("skills", [])
        if isinstance(raw_skills, list):
            seen = set()
            deduped = []
            for sk in raw_skills:
                sk_norm = str(sk).strip()
                if sk_norm and sk_norm.lower() not in seen:
                    seen.add(sk_norm.lower())
                    deduped.append(sk_norm)
            result["skills"] = deduped

        print(f"[AI-EXTRACT] Final: {len(result.get('workExperience', []))} jobs, "
              f"{len(result.get('projects', []))} projects, "
              f"{len(result.get('education', []))} edu, "
              f"{len(cleaned_certifs)} certifs, "
              f"{len(result.get('skills', []))} skills, "
              f"bio={'yes' if result.get('bio') else 'no'}, "
              f"languages={result.get('languages', [])}")

        return result

    except json.JSONDecodeError as e:
        print(f"[AI-EXTRACT] JSON parse error: {e}")
        raise HTTPException(status_code=500, detail="AI returned invalid JSON — try again")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[AI-EXTRACT] Error: {e}")
        raise HTTPException(status_code=500, detail=f"CV extraction failed: {str(e)}")



# ── Mission Matching ──────────────────────────────────────────────────────────

class WorkExperienceItem(BaseModel):
    jobTitle: Optional[str] = ""
    company: Optional[str] = ""
    description: Optional[str] = ""
    isCurrent: Optional[bool] = False

class ProjectItem(BaseModel):
    name: Optional[str] = ""
    description: Optional[str] = ""
    technologies: Optional[List[str]] = []

class MatchMissionRequest(BaseModel):
    freelancerSkills: Optional[List[str]] = []
    freelancerBio: Optional[str] = ""
    freelancerPosition: Optional[str] = ""
    freelancerExperience: Optional[int] = None
    workExperience: Optional[List[WorkExperienceItem]] = []
    projects: Optional[List[ProjectItem]] = []
    missionTitle: Optional[str] = ""
    missionDescription: Optional[str] = ""
    missionRequiredSkills: Optional[str] = ""
    missionTechnicalEnvironment: Optional[str] = ""

class MatchMissionResponse(BaseModel):
    score: int
    skillScore: int
    semanticScore: int
    matchedSkills: List[str]
    missingSkills: List[str]
    recommendation: str
    explanation: str


@app.post("/match-mission", response_model=MatchMissionResponse)
def match_mission(req: MatchMissionRequest):
    """
    Calcule la compatibilité entre un freelancer et une mission.
    Retourne un score (0-100), les compétences matchées/manquantes et une recommandation.
    """
    try:
        print("[MATCH] Starting mission matching...")

        # ── 1. Skill Match Score ──────────────────────────────────────────────
        raw_required = strip_html(req.missionRequiredSkills or "")
        required_skills = [s.strip() for s in re.split(r'[,;|\n]+', raw_required) if s.strip()]

        freelancer_skills_norm = [normalize(s) for s in (req.freelancerSkills or [])]
        required_skills_norm = [normalize(s) for s in required_skills]

        matched_skills = []
        missing_skills = []

        for i, req_skill_norm in enumerate(required_skills_norm):
            original = required_skills[i] if i < len(required_skills) else req_skill_norm
            found = False
            for fl_skill_norm in freelancer_skills_norm:
                # Exact match ou substring match (ex: "React" dans "ReactJS")
                if req_skill_norm in fl_skill_norm or fl_skill_norm in req_skill_norm:
                    found = True
                    break
            if found:
                matched_skills.append(original)
            else:
                missing_skills.append(original)

        skill_score = int((len(matched_skills) / len(required_skills)) * 100) if required_skills else 0
        print(f"[MATCH] Skill score: {skill_score}% ({len(matched_skills)}/{len(required_skills)})")

        # ── 2. Semantic Similarity Score ──────────────────────────────────────
        # Construire texte CV du freelancer
        exp_parts = []
        for we in (req.workExperience or [])[:3]:
            if we.jobTitle or we.company:
                part = f"{we.jobTitle or ''} at {we.company or ''}".strip()
                if we.description:
                    part += f": {we.description[:200]}"
                exp_parts.append(part)

        proj_parts = []
        for p in (req.projects or [])[:3]:
            if p.name:
                tech_str = ", ".join(p.technologies or [])
                part = f"{p.name}: {p.description or ''} ({tech_str})"
                proj_parts.append(part[:200])

        cv_text = " | ".join(filter(None, [
            req.freelancerPosition or "",
            req.freelancerBio or "",
            "Compétences: " + ", ".join(req.freelancerSkills or []),
            "Expérience: " + " | ".join(exp_parts),
            "Projets: " + " | ".join(proj_parts),
        ]))

        # Construire texte Mission
        mission_text = " | ".join(filter(None, [
            req.missionTitle or "",
            req.missionDescription or "",
            "Compétences requises: " + (req.missionRequiredSkills or ""),
            "Environnement technique: " + (req.missionTechnicalEnvironment or ""),
        ]))

        cv_vec = model.encode(cv_text, normalize_embeddings=True)
        mission_vec = model.encode(mission_text, normalize_embeddings=True)
        semantic_sim = float(np.dot(cv_vec, mission_vec))  # cosine sim (vecteurs normalisés)
        semantic_score = int(max(0, min(100, semantic_sim * 100)))
        print(f"[MATCH] Semantic score: {semantic_score}%")

        # ── 3. Score final ────────────────────────────────────────────────────
        final_score = int(round(0.55 * skill_score + 0.45 * semantic_score))
        final_score = max(0, min(100, final_score))
        print(f"[MATCH] Final score: {final_score}%")

        # ── 4. LLM Recommendation via llama3.2 ───────────────────────────────
        matched_str = ", ".join(matched_skills) if matched_skills else "aucune"
        missing_str = ", ".join(missing_skills[:8]) if missing_skills else "aucune"
        skills_str = ", ".join((req.freelancerSkills or [])[:15])
        exp_str = f"{req.freelancerExperience} ans" if req.freelancerExperience is not None else "non précisée"

        llm_prompt = f"""You are an expert tech recruiter. Analyze the compatibility between this freelancer and this job mission.

FREELANCER:
- Current position: {req.freelancerPosition or 'not specified'}
- Experience: {exp_str}
- Skills: {skills_str}
- Bio: {(req.freelancerBio or '')[:300]}

MISSION: {req.missionTitle or 'not specified'}
- Description: {(req.missionDescription or '')[:400]}
- Required skills: {raw_required[:300]}

PRE-CALCULATED ANALYSIS:
- Compatibility score: {final_score}%
- Matched skills ({len(matched_skills)}): {matched_str}
- Missing skills ({len(missing_skills)}): {missing_str}

Reply ONLY with valid JSON (no text before or after) with exactly these 2 fields:
{{
  "recommendation": "APPLY" or "APPLY WITH RESERVATIONS" or "DO NOT APPLY",
  "explanation": "2-3 sentences maximum explaining why, in English, concise and direct"
}}"""

        try:
            response = ollama.chat(
                model='llama3.2',
                messages=[{'role': 'user', 'content': llm_prompt}],
                options={"temperature": 0.1}
            )
            raw = response['message']['content'].strip()
            # Extraire le JSON
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                llm_data = json.loads(json_match.group())
                recommendation = llm_data.get("recommendation", "").strip().upper()
                explanation = llm_data.get("explanation", "").strip()
                # Valider la recommendation
                valid_recs = ["APPLY", "APPLY WITH RESERVATIONS", "DO NOT APPLY"]
                if recommendation not in valid_recs:
                    recommendation = "APPLY" if final_score >= 60 else ("APPLY WITH RESERVATIONS" if final_score >= 40 else "DO NOT APPLY")
            else:
                raise ValueError("No JSON found in LLM response")
        except Exception as llm_err:
            print(f"[MATCH] LLM error (using fallback): {llm_err}")
            # Fallback basé sur le score
            if final_score >= 65:
                recommendation = "APPLY"
                explanation = f"Your profile is a strong match for this mission with {final_score}% compatibility. You have {len(matched_skills)} out of {len(required_skills)} required skills."
            elif final_score >= 40:
                recommendation = "APPLY WITH RESERVATIONS"
                explanation = f"Your profile is a partial match ({final_score}%). Some key skills are missing: {missing_str[:100]}."
            else:
                recommendation = "DO NOT APPLY"
                explanation = f"Your profile does not sufficiently match this mission ({final_score}%). Too many required skills are absent from your profile."

        print(f"[MATCH] Recommendation: {recommendation}")

        return MatchMissionResponse(
            score=final_score,
            skillScore=skill_score,
            semanticScore=semantic_score,
            matchedSkills=matched_skills,
            missingSkills=missing_skills,
            recommendation=recommendation,
            explanation=explanation,
        )

    except Exception as e:
        print(f"[MATCH] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Matching failed: {str(e)}")


@app.post("/match-mission-quick", response_model=MatchMissionResponse)
def match_mission_quick(req: MatchMissionRequest):
    """
    Version rapide sans LLM : skill matching + semantic similarity uniquement.
    Retourne les résultats en ~1-2 secondes avec une recommandation basée sur les scores.
    """
    try:
        print("[MATCH-QUICK] Starting fast mission matching...")

        # ── 1. Skill Match Score ──────────────────────────────────────────────
        raw_required = strip_html(req.missionRequiredSkills or "")
        required_skills = [s.strip() for s in re.split(r'[,;|\n]+', raw_required) if s.strip()]

        freelancer_skills_norm = [normalize(s) for s in (req.freelancerSkills or [])]
        required_skills_norm = [normalize(s) for s in required_skills]

        matched_skills = []
        missing_skills = []

        for i, req_skill_norm in enumerate(required_skills_norm):
            original = required_skills[i] if i < len(required_skills) else req_skill_norm
            found = False
            for fl_skill_norm in freelancer_skills_norm:
                if req_skill_norm in fl_skill_norm or fl_skill_norm in req_skill_norm:
                    found = True
                    break
            if found:
                matched_skills.append(original)
            else:
                missing_skills.append(original)

        skill_score = int((len(matched_skills) / len(required_skills)) * 100) if required_skills else 0

        # ── 2. Semantic Similarity Score ──────────────────────────────────────
        exp_parts = []
        for we in (req.workExperience or [])[:3]:
            if we.jobTitle or we.company:
                part = f"{we.jobTitle or ''} at {we.company or ''}".strip()
                if we.description:
                    part += f": {we.description[:200]}"
                exp_parts.append(part)

        proj_parts = []
        for p in (req.projects or [])[:3]:
            if p.name:
                tech_str = ", ".join(p.technologies or [])
                part = f"{p.name}: {p.description or ''} ({tech_str})"
                proj_parts.append(part[:200])

        cv_text = " | ".join(filter(None, [
            req.freelancerPosition or "",
            req.freelancerBio or "",
            "Compétences: " + ", ".join(req.freelancerSkills or []),
            "Expérience: " + " | ".join(exp_parts),
            "Projets: " + " | ".join(proj_parts),
        ]))

        mission_text = " | ".join(filter(None, [
            req.missionTitle or "",
            req.missionDescription or "",
            "Compétences requises: " + (req.missionRequiredSkills or ""),
            "Environnement technique: " + (req.missionTechnicalEnvironment or ""),
        ]))

        cv_vec = model.encode(cv_text, normalize_embeddings=True)
        mission_vec = model.encode(mission_text, normalize_embeddings=True)
        semantic_sim = float(np.dot(cv_vec, mission_vec))
        semantic_score = int(max(0, min(100, semantic_sim * 100)))

        # ── 3. Score final ────────────────────────────────────────────────────
        final_score = int(round(0.55 * skill_score + 0.45 * semantic_score))
        final_score = max(0, min(100, final_score))

        # ── 4. Recommandation basée sur les scores (sans LLM) ─────────────────
        missing_str = ", ".join(missing_skills[:8]) if missing_skills else "none"
        if final_score >= 65:
            recommendation = "APPLY"
            explanation = f"Your profile is a strong match for this mission with {final_score}% compatibility. You have {len(matched_skills)} out of {len(required_skills)} required skills."
        elif final_score >= 40:
            recommendation = "APPLY WITH RESERVATIONS"
            explanation = f"Your profile is a partial match ({final_score}%). Some key skills are missing: {missing_str[:100]}."
        else:
            recommendation = "DO NOT APPLY"
            explanation = f"Your profile does not sufficiently match this mission ({final_score}%). Too many required skills are absent from your profile."

        print(f"[MATCH-QUICK] Done. Score: {final_score}%, Recommendation: {recommendation}")

        return MatchMissionResponse(
            score=final_score,
            skillScore=skill_score,
            semanticScore=semantic_score,
            matchedSkills=matched_skills,
            missingSkills=missing_skills,
            recommendation=recommendation,
            explanation=explanation,
        )

    except Exception as e:
        print(f"[MATCH-QUICK] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Quick matching failed: {str(e)}")


# ── Rank Candidates ───────────────────────────────────────────────────────────

class CandidateProfile(BaseModel):
    applicationId: str
    freelancerId: str
    skills: Optional[List[str]] = []
    bio: Optional[str] = ""
    currentPosition: Optional[str] = ""
    yearsOfExperience: Optional[int] = None
    profileTypes: Optional[List[str]] = []
    workExperience: Optional[List[dict]] = []
    projects: Optional[List[dict]] = []
    certifications: Optional[List[dict]] = []
    education: Optional[List[dict]] = []
    rating: Optional[float] = None
    portfolioUrl: Optional[str] = None
    cvUrl: Optional[str] = None


class RankCandidatesRequest(BaseModel):
    missionId: str
    missionTitle: Optional[str] = ""
    missionDescription: Optional[str] = ""
    missionRequiredSkills: Optional[str] = ""
    missionTechnicalEnvironment: Optional[str] = ""
    missionYearsOfExperience: Optional[int] = None
    candidates: List[CandidateProfile]


class CandidateRankResult(BaseModel):
    applicationId: str
    freelancerId: str
    rank: int
    totalScore: float
    skillScore: float
    experienceScore: float
    semanticScore: float
    completenessScore: float
    matchedSkills: List[str]
    missingSkills: List[str]


@app.post("/rank-candidates", response_model=List[CandidateRankResult])
def rank_candidates(req: RankCandidatesRequest):
    """Classe les candidats d'une mission par score AI (skills + expérience + sémantique + complétude)."""
    if not req.candidates:
        return []

    # Encoder le texte mission une seule fois
    mission_text = " | ".join(filter(None, [
        req.missionTitle or "",
        req.missionDescription or "",
        req.missionRequiredSkills or "",
        req.missionTechnicalEnvironment or "",
    ]))
    mission_vec = model.encode(mission_text, normalize_embeddings=True)

    # Parser les skills requis (requiredSkills + technicalEnvironment)
    raw_required = strip_html(req.missionRequiredSkills or "") + " " + strip_html(req.missionTechnicalEnvironment or "")
    required_skills = [s.strip() for s in re.split(r'[,;|\n]+', raw_required) if s.strip()]
    required_skills_norm = [normalize(s) for s in required_skills]

    results = []

    for candidate in req.candidates:

        # ── 1. Skill Score (40%) ──────────────────────────────────────────────
        freelancer_skills_norm = [normalize(s) for s in (candidate.skills or [])]
        matched, missing = [], []
        for i, req_norm in enumerate(required_skills_norm):
            original = required_skills[i] if i < len(required_skills) else req_norm
            found = any(req_norm in fs or fs in req_norm for fs in freelancer_skills_norm)
            (matched if found else missing).append(original)
        skill_score = (len(matched) / len(required_skills) * 100) if required_skills else 50.0

        # ── 2. Experience Score (25%) ─────────────────────────────────────────
        required_exp = req.missionYearsOfExperience or 0
        candidate_exp = candidate.yearsOfExperience or 0
        if required_exp == 0:
            experience_score = 100.0
        elif candidate_exp >= required_exp:
            experience_score = min(100.0, 100.0 + (candidate_exp - required_exp) * 5)
        else:
            experience_score = max(0.0, (candidate_exp / required_exp) * 100)

        # ── 3. Semantic Score (25%) ───────────────────────────────────────────
        exp_parts = []
        for we in (candidate.workExperience or [])[:3]:
            title = we.get("jobTitle", "") or ""
            company = we.get("company", "") or ""
            desc = (we.get("description", "") or "")[:150]
            exp_parts.append(f"{title} at {company}: {desc}")

        proj_parts = []
        for p in (candidate.projects or [])[:3]:
            techs = ", ".join(p.get("technologies", []) or [])
            proj_parts.append(f"{p.get('name','')}: {(p.get('description','') or '')[:100]} ({techs})")

        cv_text = " | ".join(filter(None, [
            candidate.currentPosition or "",
            candidate.bio or "",
            "Skills: " + ", ".join(candidate.skills or []),
            " | ".join(exp_parts),
            " | ".join(proj_parts),
        ]))
        cv_vec = model.encode(cv_text, normalize_embeddings=True)
        semantic_score = float(np.dot(cv_vec, mission_vec)) * 100
        semantic_score = max(0.0, min(100.0, semantic_score))

        # ── 4. Complétude profil (10%) ────────────────────────────────────────
        completeness = 0.0
        if candidate.bio and len((candidate.bio or "").strip()) > 20:   completeness += 20
        if candidate.certifications:                                      completeness += 15
        if candidate.workExperience:                                      completeness += 20
        if candidate.projects:                                            completeness += 15
        if candidate.education:                                           completeness += 10
        if candidate.portfolioUrl:                                        completeness += 10
        if candidate.cvUrl:                                               completeness += 5
        if candidate.rating and candidate.rating >= 4.0:                 completeness += 5

        # ── Score final pondéré ───────────────────────────────────────────────
        total = round(
            0.40 * skill_score +
            0.25 * experience_score +
            0.25 * semantic_score +
            0.10 * completeness,
            1
        )

        results.append(CandidateRankResult(
            applicationId=candidate.applicationId,
            freelancerId=candidate.freelancerId,
            rank=0,
            totalScore=total,
            skillScore=round(skill_score, 1),
            experienceScore=round(experience_score, 1),
            semanticScore=round(semantic_score, 1),
            completenessScore=round(completeness, 1),
            matchedSkills=matched,
            missingSkills=missing,
        ))

    # Trier par score décroissant et assigner les rangs
    results.sort(key=lambda x: x.totalScore, reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1

    print(f"[RANK] Mission {req.missionId}: {len(results)} candidates ranked, best={results[0].totalScore}%")
    return results


@app.post("/recommend-missions", response_model=List[SearchResult])
def recommend_missions(req: FreelancerIndexRequest):
    """
    Recommande uniquement les missions vraiment compatibles avec le profil freelancer.
    Score combiné : 55% skills match + 45% similarité sémantique.
    Filtre strict : score combiné >= 42 ET au moins 1 skill correspondant.
    """
    if len(mission_ids) == 0:
        return []

    # ── 1. Premier passage FAISS — top 20 candidats sémantiques ──────────────
    profile_text = freelancer_to_text(req)
    query_vec = model.encode(profile_text, normalize_embeddings=True)
    query_vec = np.array([query_vec], dtype=np.float32)

    k = min(20, len(mission_ids))
    faiss_scores, faiss_indices = index.search(query_vec, k)

    # Seuil sémantique de base (élimine les missions vraiment hors sujet)
    SEMANTIC_MIN = 0.22
    candidates = [
        (mission_ids[idx], float(score))
        for score, idx in zip(faiss_scores[0], faiss_indices[0])
        if idx >= 0 and float(score) >= SEMANTIC_MIN
    ]

    if not candidates:
        print(f"[AI] Recommend: freelancer {req.id} → 0 results (semantic filter)")
        return []

    # ── 2. Skill matching sur chaque candidat ────────────────────────────────
    freelancer_skills_norm = [normalize(s) for s in (req.skills or [])]
    position_norm = normalize(req.currentPosition or "")
    # Élargir avec le poste (ex: "angular developer" → skills angular, typescript…)
    expanded_position = normalize(expand_prompt(req.currentPosition or ""))

    results = []
    for mid, sem_score in candidates:
        meta = mission_metadata.get(mid, {})
        content_norm = meta.get("content_norm", "")

        # Compter les skills du freelancer présents dans la mission
        matched = 0
        for skill_norm in freelancer_skills_norm:
            if skill_norm and (skill_norm in content_norm):
                matched += 1
            else:
                # Chercher aussi via les synonymes du dictionnaire d'expansion
                for syn in DOMAIN_EXPANSION.get(skill_norm, []):
                    if normalize(syn) in content_norm:
                        matched += 1
                        break

        total_skills = max(len(freelancer_skills_norm), 1)
        skill_score = (matched / total_skills) * 100

        # Bonus si le titre de poste matche le contenu de la mission
        if position_norm and position_norm in content_norm:
            skill_score = min(100, skill_score + 15)
        elif any(term in content_norm for term in expanded_position.split()[:8] if len(term) > 3):
            skill_score = min(100, skill_score + 8)

        # Score combiné
        sem_pct = sem_score * 100
        combined = round(0.55 * skill_score + 0.45 * sem_pct, 1)

        # Filtre strict : score combiné >= 42 ET au moins 1 skill ou position match
        has_skill_match = matched > 0 or (position_norm and position_norm in content_norm)
        if combined >= 42 and has_skill_match:
            results.append((mid, combined))

    if not results:
        print(f"[AI] Recommend: freelancer {req.id} → 0 results after skill filter")
        return []

    # ── 3. Dédupliquer (même mission indexée 2x), trier, top 8 ─────────────
    seen = set()
    unique_results = []
    for mid, score in sorted(results, key=lambda x: x[1], reverse=True):
        if mid not in seen:
            seen.add(mid)
            unique_results.append((mid, score))
    results = unique_results[:8]

    print(f"[AI] Recommend: freelancer {req.id} → {len(results)} compatible missions "
          f"(best={results[0][1]}%)")
    return [SearchResult(mission_id=mid, score=score) for mid, score in results]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed_missions": len(mission_ids),
        "indexed_freelancers": len(freelancer_ids),
        "model": "paraphrase-multilingual-MiniLM-L12-v2"
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPANY TRUST SCORE
# ═══════════════════════════════════════════════════════════════════════════════

# Domaines d'email considérés comme génériques (non professionnels)
# ══════════════════════════════════════════════════════════════════════════════
#  COMPANY TRUST SCORE — 3-Node AI Workflow
#  Node 1 : Website + Domain Age + Social Media (python-whois + requests + BS4)
#  Node 2 : Email Validation + Matching       (email-validator + regex)
#  Node 3 : AI Scoring                        (Ollama llama3.2 + rule fallback)
# ══════════════════════════════════════════════════════════════════════════════

GENERIC_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "live.com",
    "mail.com", "icloud.com", "aol.com", "yandex.com", "protonmail.com",
    "tutanota.com", "inbox.com", "gmx.com", "zoho.com", "yahoo.fr",
    "hotmail.fr", "orange.fr", "free.fr", "sfr.fr", "laposte.net",
    "msn.com", "rediffmail.com", "gmx.de", "web.de", "libero.it"
}

class CompanyTrustRequest(BaseModel):
    company_id: str
    company_name: str
    email: str
    website_url: Optional[str] = None
    trade_register: Optional[str] = None
    description: Optional[str] = None
    business_sector: Optional[str] = None
    manager_name: Optional[str] = None
    manager_email: Optional[str] = None
    number_of_employees: Optional[int] = None
    foundation_date: Optional[str] = None
    address: Optional[str] = None

class CompanyTrustResponse(BaseModel):
    company_id: str
    trust_score: int
    details: dict
    label: str

# ─── Node 1 : Website + Domain Age + Social Media ────────────────────────────

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

def _whois_lookup(domain: str) -> dict:
    """Isolé dans un thread avec timeout pour éviter le blocage."""
    try:
        import whois as python_whois
        w = python_whois.whois(domain)
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if creation:
            if hasattr(creation, 'tzinfo') and creation.tzinfo:
                creation = creation.replace(tzinfo=None)
            age_years = (datetime.datetime.now() - creation).days / 365.25
            return {"age_years": round(age_years, 1), "created": str(creation.date())}
    except Exception as e:
        pass
    return {"age_years": None, "created": None}

def _resolve_ip_with_fallback(hostname: str) -> tuple[str | None, str]:
    """
    Resolve hostname to (IPv4, actual_hostname_used).
    Tries: local DNS → Google DNS (bare domain) → Google DNS (www. prefix) → Google DNS (without www.)
    Returns (ip, resolved_hostname) or (None, hostname).
    """
    candidates = [hostname]
    if not hostname.startswith("www."):
        candidates.append("www." + hostname)
    else:
        candidates.append(hostname[4:])  # without www

    # 1. Local DNS
    for h in candidates:
        try:
            results = socket.getaddrinfo(h, None, socket.AF_INET, socket.SOCK_STREAM)
            if results:
                return results[0][4][0], h
        except socket.gaierror:
            pass

    # 2. Google DNS fallback
    try:
        import dns.resolver
        resolver = dns.resolver.Resolver()
        resolver.nameservers = ['8.8.8.8', '8.8.4.4', '1.1.1.1']
        resolver.timeout = 5
        resolver.lifetime = 5
        for h in candidates:
            try:
                answers = resolver.resolve(h, 'A')
                return str(answers[0]), h
            except Exception:
                continue
    except Exception:
        pass

    return None, hostname


def _fetch_url_robust(url: str, headers: dict, timeout: int = 12) -> requests.Response | None:
    """
    Fetch a URL with SSL and DNS fallback.
    - Tries verify=True first, then verify=False (for self-signed certs)
    - If local DNS fails, resolves via Google DNS and connects to IP directly
      with Host header set (bypasses local DNS restriction for .tn etc.)
    """
    import warnings
    from urllib.parse import urlparse

    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    for verify_ssl in [True, False]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resp = requests.get(url, headers=headers, timeout=timeout,
                                    allow_redirects=True, verify=verify_ssl)
            return resp
        except requests.exceptions.SSLError:
            if verify_ssl:
                continue  # retry without SSL verify
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            err_str = str(e)
            if "getaddrinfo" in err_str or "NameResolution" in err_str or "11001" in err_str:
                break  # DNS error — try Google DNS below
            if not verify_ssl:
                break
            continue
        except Exception:
            break

    # ── Google DNS fallback ────────────────────────────────────────────────
    if not hostname:
        return None

    ip, resolved_host = _resolve_ip_with_fallback(hostname)
    if not ip:
        return None

    print(f"[TRUST-N1] DNS fallback: {hostname} → {resolved_host} @ {ip}")

    scheme = parsed.scheme
    port = parsed.port or (443 if scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query

    ip_url = f"{scheme}://{ip}:{port}{path}"
    req_headers = dict(headers)
    req_headers["Host"] = resolved_host  # Use the hostname that actually resolved

    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resp = requests.get(ip_url, headers=req_headers, timeout=timeout,
                                allow_redirects=True, verify=False)
        return resp
    except Exception as e:
        print(f"[TRUST-N1] DNS-fallback request error: {e}")

    return None


def node1_website_analysis(url: str, company_name: str) -> dict:
    """
    Vérifie existence du site, SSL, âge du domaine (whois),
    et scrape les liens réseaux sociaux depuis la homepage.
    Score max : 55 pts
    """
    result = {
        "website_exists": False,
        "ssl_valid": False,
        "domain_age_years": None,
        "response_time_ms": None,
        "social_media": {
            "linkedin": False,
            "facebook": False,
            "twitter": False,
            "instagram": False,
        },
        "social_links_found": [],
        "score": 0,
        "details": []
    }

    if not url or not url.strip():
        result["details"].append("no_website_provided")
        return result

    url = url.strip()
    if not url.startswith("http"):
        url = "https://" + url

    from urllib.parse import urlparse
    parsed = urlparse(url)
    domain = parsed.netloc or parsed.path
    # Strip port if present
    domain = domain.split(":")[0]

    # ── 1. SSL check (with DNS fallback) ─────────────────────────────────────
    ssl_checked = False
    # Resolve IP first (handles .tn and other locally-unresolvable domains)
    resolved_ip, resolved_hostname = _resolve_ip_with_fallback(domain)
    connect_host = resolved_ip if resolved_ip else domain
    ssl_sni = resolved_hostname if resolved_ip else domain

    for strict in [True, False]:
        try:
            ctx = ssl.create_default_context() if strict else ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            if not strict:
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            with socket.create_connection((connect_host, 443), timeout=6) as sock:
                with ctx.wrap_socket(sock, server_hostname=ssl_sni) as ssock:
                    ssock.getpeercert()
                    result["ssl_valid"] = strict
                    result["score"] += 10
                    result["details"].append("ssl_valid" if strict else "ssl_self_signed")
                    ssl_checked = True
                    break
        except Exception:
            if strict:
                continue
    if not ssl_checked:
        result["details"].append("ssl_missing_or_invalid")

    # ── 2. Site reachable + scrape social links ───────────────────────────────
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }
    homepage_html = None

    # Build candidate URLs: prefer https, also try with/without www
    candidates = [url]
    if url.startswith("https://") and not domain.startswith("www."):
        candidates.append(f"https://www.{domain}")
    if url.startswith("https://"):
        candidates.append(url.replace("https://", "http://"))
    elif url.startswith("http://"):
        candidates.append(url.replace("http://", "https://"))

    for attempt_url in candidates:
        t0 = time.time()
        resp = _fetch_url_robust(attempt_url, headers, timeout=12)
        if resp is not None and resp.status_code < 500:
            elapsed = int((time.time() - t0) * 1000)
            result["website_exists"] = True
            result["response_time_ms"] = elapsed
            result["score"] += 15
            result["details"].append("website_reachable")
            if resp.status_code < 400:
                homepage_html = resp.text
            break

    if not result["website_exists"]:
        result["details"].append("website_unreachable")

    # ── 3. Social media detection (scrape homepage) ───────────────────────────
    if homepage_html:
        try:
            soup = BeautifulSoup(homepage_html, "html.parser")

            # Collect hrefs from <a> tags AND src/content from meta/iframe
            all_hrefs = [a.get("href", "") or "" for a in soup.find_all("a", href=True)]
            # Also check page raw text for social URLs (some sites embed them in JS/data attrs)
            raw_lower = homepage_html.lower()

            sm_map = {
                "linkedin":  ["linkedin.com/company/", "linkedin.com/in/"],
                "facebook":  ["facebook.com/"],
                "twitter":   ["twitter.com/", "x.com/"],
                "instagram": ["instagram.com/"],
                "youtube":   ["youtube.com/channel/", "youtube.com/c/", "youtube.com/@"],
            }
            # Initialize all platforms as None (not detected)
            for plat in sm_map:
                result["social_media"][plat] = None

            EXCLUDE = {"share", "intent", "login", "sharer", "watch?v=", "playlist", "hashtag"}

            for platform, patterns in sm_map.items():
                best_url = None

                # 1. Check <a href> links — prefer exact profile links
                for href in all_hrefs:
                    href_l = href.lower()
                    if not any(p in href_l for p in patterns):
                        continue
                    if any(ex in href_l for ex in EXCLUDE):
                        continue
                    # Normalize URL: ensure it's absolute
                    if href.startswith("//"):
                        href = "https:" + href
                    elif href.startswith("/"):
                        continue  # relative path, skip
                    if not href.startswith("http"):
                        continue
                    best_url = href.rstrip("/")
                    break

                # 2. Fallback: extract URL from raw HTML (JS vars, data attrs, og: meta)
                if not best_url:
                    for p in patterns:
                        idx = raw_lower.find(p)
                        while idx != -1:
                            # Walk back to find start of URL
                            start = idx
                            for c in range(min(50, idx), 0, -1):
                                ch = homepage_html[idx - c]
                                if ch in ('"', "'", '(', ' ', '\n', '='):
                                    start = idx - c + 1
                                    break
                            # Walk forward to end of URL
                            end = idx + len(p)
                            for ch in homepage_html[end:end + 120]:
                                if ch in ('"', "'", ')', ' ', '\n', '\\'):
                                    break
                                end += 1
                            candidate = homepage_html[start:end].strip().strip('"\'')
                            if candidate.startswith("http") and p in candidate.lower():
                                if not any(ex in candidate.lower() for ex in EXCLUDE):
                                    best_url = candidate.rstrip("/")
                                    break
                            idx = raw_lower.find(p, idx + 1)
                        if best_url:
                            break

                if best_url:
                    result["social_media"][platform] = best_url
                    if best_url not in result["social_links_found"]:
                        result["social_links_found"].append(best_url)

            social_count = sum(1 for v in result["social_media"].values() if v)
            if social_count >= 3:
                result["score"] += 20
                result["details"].append(f"social_media_strong_{social_count}_platforms")
            elif social_count == 2:
                result["score"] += 13
                result["details"].append(f"social_media_good_2_platforms")
            elif social_count == 1:
                result["score"] += 6
                result["details"].append("social_media_minimal_1_platform")
            else:
                result["details"].append("no_social_media_detected")

        except Exception as e:
            print(f"[TRUST-N1] Social scraping error: {e}")

    # ── 4. Domain age via python-whois (thread + 8s timeout) ─────────────────
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_whois_lookup, domain)
            whois_data = future.result(timeout=8)
        age_years = whois_data.get("age_years")
        if age_years is not None:
            result["domain_age_years"] = age_years
            if age_years >= 5:
                result["score"] += 10
                result["details"].append(f"domain_age_{int(age_years)}y_established")
            elif age_years >= 2:
                result["score"] += 6
                result["details"].append(f"domain_age_{int(age_years)}y")
            elif age_years >= 0.5:
                result["score"] += 3
                result["details"].append(f"domain_age_recent")
            else:
                result["details"].append("domain_very_new")
        else:
            result["details"].append("whois_no_date")
    except FutureTimeoutError:
        print(f"[TRUST-N1] WHOIS timeout for {domain}")
        result["details"].append("whois_timeout")
    except Exception as e:
        print(f"[TRUST-N1] WHOIS error: {e}")
        result["details"].append("whois_error")

    print(f"[TRUST-N1] {domain} → score={result['score']}, social={result['social_media']}, age={result.get('domain_age_years')}y")
    return result


# ─── Node 2 : Email Validation + Matching ────────────────────────────────────

def node2_email_analysis(email: str, company_name: str, website_url: str) -> dict:
    """
    Valide l'email (format + MX records), vérifie si pro ou générique,
    et teste le matching email ↔ site ↔ nom d'entreprise.
    Score max : 55 pts
    """
    result = {
        "email_valid_format": False,
        "has_mx_records": False,
        "is_professional": False,
        "matches_website": False,
        "matches_company_name": False,
        "email_domain": "",
        "score": 0,
        "details": []
    }

    if not email:
        result["details"].append("no_email_provided")
        return result

    email = email.lower().strip()
    domain = email.split("@")[-1] if "@" in email else ""
    result["email_domain"] = domain

    # ── 1. Format + MX records ────────────────────────────────────────────────
    try:
        from email_validator import validate_email, EmailNotValidError
        validated = validate_email(email, check_deliverability=True)
        result["email_valid_format"] = True
        result["has_mx_records"] = True
        result["score"] += 10
        result["details"].append("email_valid_with_mx")
    except Exception:
        try:
            from email_validator import validate_email, EmailNotValidError
            validated = validate_email(email, check_deliverability=False)
            result["email_valid_format"] = True
            result["score"] += 4
            result["details"].append("email_valid_format_only")
        except Exception:
            result["details"].append("email_format_invalid")

    # ── 2. Professional domain ────────────────────────────────────────────────
    if domain and domain not in GENERIC_EMAIL_DOMAINS:
        result["is_professional"] = True
        result["score"] += 20
        result["details"].append("professional_email_domain")
    else:
        result["details"].append("generic_email_domain_detected")

    # ── 3. Email domain ↔ website domain ─────────────────────────────────────
    if website_url and domain:
        ws = website_url.lower()
        for prefix in ["https://", "http://", "www."]:
            ws = ws.replace(prefix, "")
        ws_domain = ws.split("/")[0].split(":")[0]

        # Exact match or subdomain
        if domain == ws_domain or domain.endswith("." + ws_domain) or ws_domain.endswith("." + domain):
            result["matches_website"] = True
            result["score"] += 15
            result["details"].append("email_exactly_matches_website")
        else:
            # Partial match (e.g. contact@myco.fr vs www.mycompany.fr)
            ws_base = ws_domain.split(".")[0]
            em_base = domain.split(".")[0]
            if ws_base and em_base and (ws_base in em_base or em_base in ws_base):
                result["matches_website"] = True
                result["score"] += 8
                result["details"].append("email_partially_matches_website")

    # ── 4. Email domain ↔ company name ───────────────────────────────────────
    if company_name and domain:
        name_slug = re.sub(r"[^a-z0-9]", "", company_name.lower())
        domain_base = re.sub(r"[^a-z0-9]", "", domain.split(".")[0])

        if name_slug and domain_base:
            if name_slug in domain_base or domain_base in name_slug:
                result["matches_company_name"] = True
                result["score"] += 10
                result["details"].append("email_matches_company_name")
            elif len(name_slug) >= 4 and len(domain_base) >= 4:
                # Fuzzy: at least 4 chars in common at start
                common = sum(1 for a, b in zip(name_slug, domain_base) if a == b)
                if common >= 4:
                    result["matches_company_name"] = True
                    result["score"] += 5
                    result["details"].append("email_fuzzy_matches_company_name")

    print(f"[TRUST-N2] {email} → score={result['score']}, pro={result['is_professional']}, mx={result['has_mx_records']}")
    return result


# ─── Node 3 : AI Scoring (Ollama llama3.2 + rule-based fallback) ─────────────

def node3_ai_scoring(company_data: dict, node1: dict, node2: dict) -> dict:
    """
    Utilise Ollama (llama3.2, local, gratuit) pour analyser tous les signaux
    et produire un score final raisonné.
    Fallback automatique sur scoring par règles si Ollama est indisponible.
    """

    # ── Rule-based score (fallback + base de référence) ───────────────────────
    rule_score = node1["score"] + node2["score"]

    # Bonus données entreprise
    if company_data.get("trade_register") and len(str(company_data["trade_register"]).strip()) >= 3:
        rule_score += 15
    if company_data.get("address") and len(str(company_data["address"]).strip()) > 5:
        rule_score += 5
    if company_data.get("number_of_employees") and int(company_data["number_of_employees"]) > 0:
        rule_score += 5
    desc = company_data.get("description") or ""
    word_count = len(desc.split())
    if word_count >= 50:
        rule_score += 8
    elif word_count >= 20:
        rule_score += 4
    if company_data.get("manager_email") and company_data.get("email"):
        if company_data["manager_email"].lower().strip() != company_data["email"].lower().strip():
            rule_score += 5
    if company_data.get("foundation_date"):
        try:
            fd = datetime.date.fromisoformat(str(company_data["foundation_date"]))
            if fd < datetime.date.today() and fd.year >= 1850:
                rule_score += 5
        except Exception:
            pass

    rule_score = min(100, rule_score)

    # ── Ollama AI scoring ─────────────────────────────────────────────────────
    social_found = [k for k, v in node1.get("social_media", {}).items() if v]
    social_str = ", ".join(social_found) if social_found else "none detected"
    domain_age = node1.get("domain_age_years")
    age_str = f"{domain_age} years old" if domain_age is not None else "unknown"

    social_found = [k for k, v in node1.get("social_media", {}).items() if v]
    prompt = (
        f"Rate company trust 0-100. Return JSON only.\n"
        f"Name:{company_data.get('company_name')} Sector:{company_data.get('business_sector','?')} "
        f"Founded:{company_data.get('foundation_date','?')} Employees:{company_data.get('number_of_employees','?')}\n"
        f"Website:{'OK' if node1['website_exists'] else 'MISSING'} "
        f"SSL:{'OK' if node1['ssl_valid'] else 'MISSING'} "
        f"DomainAge:{node1.get('domain_age_years','?')}y "
        f"Social:{social_found if social_found else 'none'}\n"
        f"Email:{company_data.get('email')} Pro:{'yes' if node2['is_professional'] else 'no'} "
        f"MX:{'yes' if node2['has_mx_records'] else 'no'} "
        f"MatchSite:{'yes' if node2['matches_website'] else 'no'}\n"
        f"TradeReg:{'yes' if company_data.get('trade_register') else 'no'} "
        f"Address:{'yes' if company_data.get('address') else 'no'}\n"
        f"RuleScore:{rule_score}/100\n"
        f'Output: {{"score":<0-100>,"label":"TRUSTED|REVIEW|SUSPICIOUS","reasoning":"<8 words>"}}'
    )

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 55},
        )

        raw = response["message"]["content"] if isinstance(response, dict) else response.message.content
        raw = raw.strip()

        # Extract JSON robustly
        json_match = re.search(r'\{[^{}]*"score"\s*:\s*\d+[^{}]*\}', raw, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            ai_score = max(0, min(100, int(parsed.get("score", rule_score))))

            # Blend: 65% AI + 35% rule-based for stability
            blended = int(round(0.65 * ai_score + 0.35 * rule_score))
            blended = max(0, min(100, blended))

            print(f"[TRUST-N3] AI={ai_score}, rule={rule_score}, blended={blended}")
            return {
                "final_score": blended,
                "ai_score": ai_score,
                "rule_score": rule_score,
                "reasoning": parsed.get("reasoning", ""),
                "method": "ai_blended"
            }

    except Exception as e:
        print(f"[TRUST-N3] Ollama unavailable ({e}), using rule-based score")

    # Pure rule-based fallback
    return {
        "final_score": rule_score,
        "ai_score": None,
        "rule_score": rule_score,
        "reasoning": "Scored via technical verification (AI model unavailable).",
        "method": "rule_based"
    }


# ─── Main endpoint ────────────────────────────────────────────────────────────

@app.post("/company/trust-score", response_model=CompanyTrustResponse)
def compute_company_trust_score(req: CompanyTrustRequest):
    """
    Workflow 3-nœuds — Node1 + Node2 en PARALLÈLE, puis Node3 (AI scoring).
    Timeout global : ~25 secondes maximum.
    """
    t_start = time.time()
    print(f"\n[TRUST] ══ Starting for: {req.company_name} ══")

    company_data = {
        "company_name":        req.company_name,
        "email":               req.email,
        "website_url":         req.website_url,
        "business_sector":     req.business_sector,
        "trade_register":      req.trade_register,
        "description":         req.description,
        "number_of_employees": req.number_of_employees,
        "foundation_date":     str(req.foundation_date) if req.foundation_date else None,
        "address":             req.address,
        "manager_email":       req.manager_email,
    }

    # ── Node 1 + Node 2 en parallèle (max 15s chacun) ─────────────────────
    node1_result = {"website_exists": False, "ssl_valid": False, "domain_age_years": None,
                    "social_media": {}, "social_links_found": [], "score": 0, "details": ["skipped"]}
    node2_result = {"email_valid_format": False, "has_mx_records": False, "is_professional": False,
                    "matches_website": False, "matches_company_name": False,
                    "email_domain": "", "score": 0, "details": ["skipped"]}

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(node1_website_analysis, req.website_url or "", req.company_name)
        f2 = pool.submit(node2_email_analysis, req.email, req.company_name, req.website_url or "")
        try:
            node1_result = f1.result(timeout=15)
        except FutureTimeoutError:
            print("[TRUST] Node1 timeout (>15s)")
            node1_result["details"] = ["node1_timeout"]
        except Exception as e:
            print(f"[TRUST] Node1 error: {e}")
        try:
            node2_result = f2.result(timeout=15)
        except FutureTimeoutError:
            print("[TRUST] Node2 timeout (>15s)")
            node2_result["details"] = ["node2_timeout"]
        except Exception as e:
            print(f"[TRUST] Node2 error: {e}")

    print(f"[TRUST] Nodes 1+2 done in {int((time.time()-t_start)*1000)}ms")

    # ── Node 3 : AI scoring (max 20s, puis fallback règles) ───────────────
    node3_result = {"final_score": 0, "method": "error"}
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            f3 = pool.submit(node3_ai_scoring, company_data, node1_result, node2_result)
            node3_result = f3.result(timeout=20)
    except FutureTimeoutError:
        print("[TRUST] Node3 (Ollama) timeout — using rule-based fallback")
        # Calcul fallback direct
        score = node1_result.get("score", 0) + node2_result.get("score", 0)
        if req.trade_register: score += 15
        if req.address: score += 5
        if req.number_of_employees: score += 5
        if req.description and len(req.description.split()) >= 30: score += 8
        if req.manager_email and req.email and req.manager_email.lower() != req.email.lower(): score += 5
        node3_result = {"final_score": min(100, score), "method": "rule_based_timeout_fallback",
                        "reasoning": "Scored via rules (AI timeout).", "ai_score": None, "rule_score": min(100, score)}
    except Exception as e:
        print(f"[TRUST] Node3 error: {e}")
        node3_result = {"final_score": min(100, node1_result.get("score",0) + node2_result.get("score",0)),
                        "method": "rule_based_error_fallback", "reasoning": str(e), "ai_score": None, "rule_score": 0}

    final_score = node3_result["final_score"]
    label = "TRUSTED" if final_score >= 75 else ("REVIEW" if final_score >= 45 else "SUSPICIOUS")

    print(f"[TRUST] ══ {req.company_name}: {final_score}/100 → {label} in {int((time.time()-t_start)*1000)}ms ══\n")

    return CompanyTrustResponse(
        company_id=req.company_id,
        trust_score=final_score,
        label=label,
        details={
            "node1_website":  node1_result,
            "node2_email":    node2_result,
            "node3_scoring":  node3_result,
        }
    )
