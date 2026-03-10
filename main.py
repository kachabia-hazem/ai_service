from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import httpx
import re
import os
import unicodedata

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

    results = [
        SearchResult(mission_id=mid, score=round(s * 100, 1))
        for mid, s in candidates
    ]

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

    results = [
        FreelancerSearchResult(freelancer_id=fid, score=round(s * 100, 1))
        for fid, s in candidates
        if s >= dynamic_threshold
    ]

    mode = f"CITY={detected_cities}" if detected_cities else ("LOCATION" if location_query else "SEMANTIC")
    print(f"[AI] Freelancer search [{mode}]: '{req.prompt}' → {len(results)} results "
          f"(best={round(best_score*100,1)}%, threshold={round(dynamic_threshold*100,1)}%)")
    return results


@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed_missions": len(mission_ids),
        "indexed_freelancers": len(freelancer_ids),
        "model": "paraphrase-multilingual-MiniLM-L12-v2"
    }
