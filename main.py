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
