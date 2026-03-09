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
}


def expand_prompt(prompt: str) -> str:
    """Enrichit le prompt — insensible aux accents et variantes."""
    prompt_norm = normalize(prompt)   # sans accents, minuscules
    extra_terms = []

    for keyword, synonyms in DOMAIN_EXPANSION.items():
        if keyword in prompt_norm:
            extra_terms.extend(synonyms)

    if extra_terms:
        unique_extra = list(dict.fromkeys(extra_terms))
        # On garde plus de termes (20 au lieu de 12)
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

# Index freelancers
freelancer_index = faiss.IndexFlatIP(DIMENSION)
freelancer_ids: List[str] = []
freelancer_texts: dict = {}  # id → texte pour pouvoir ré-indexer

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


class FreelancerIndexRequest(BaseModel):
    id: str
    currentPosition: Optional[str] = ""
    skills: Optional[List[str]] = []
    bio: Optional[str] = ""
    profileTypes: Optional[List[str]] = []
    yearsOfExperience: Optional[int] = None


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

    return (
        f"{title}. {title}. {title}. "
        f"Poste: {title}. "
        f"Domaine: {field}. Spécialité: {speciality}. "
        f"Compétences requises: {skills}. "
        f"Environnement technique: {tech}. "
        f"{context}. "
        f"{description}."
    )


def add_to_index(mission_id: str, text: str):
    vec = model.encode(text, normalize_embeddings=True)
    vec = np.array([vec], dtype=np.float32)
    index.add(vec)
    mission_ids.append(mission_id)


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
        f"{skills_block}"
        f"Expérience: {exp_text}. "
        f"{context}. "
        f"Bio: {bio}."
    )


def add_freelancer_to_index(freelancer_id: str, text: str):
    vec = model.encode(text, normalize_embeddings=True)
    vec = np.array([vec], dtype=np.float32)
    freelancer_index.add(vec)
    freelancer_ids.append(freelancer_id)
    freelancer_texts[freelancer_id] = text


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
                    )
                    if req.id and req.id not in mission_ids:
                        add_to_index(req.id, mission_to_text(req))
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
                    )
                    if req.id and req.id not in freelancer_ids:
                        add_freelancer_to_index(req.id, freelancer_to_text(req))
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
    add_to_index(req.id, mission_to_text(req))
    print(f"[AI] Mission indexed: {req.id} | Total: {len(mission_ids)}")
    return {"status": "indexed", "id": req.id, "total": len(mission_ids)}


@app.post("/search", response_model=List[SearchResult])
def search_missions(req: SearchRequest):
    """Recherche sémantique par prompt utilisateur."""
    if len(mission_ids) == 0:
        return []

    enriched_prompt = expand_prompt(req.prompt)
    query_vec = model.encode(enriched_prompt, normalize_embeddings=True)
    query_vec = np.array([query_vec], dtype=np.float32)

    k = min(req.top_k, len(mission_ids))
    scores, indices = index.search(query_vec, k)

    # Collecter tous les candidats avec seuil minimum absolu
    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and float(score) > 0.22:
            candidates.append((mission_ids[idx], float(score)))

    if not candidates:
        print(f"[AI] Search: '{req.prompt}' → 0 results")
        return []

    # Seuil dynamique plus strict : 80% du meilleur score
    best_score = candidates[0][1]
    dynamic_threshold = best_score * 0.80

    results = [
        SearchResult(
            mission_id=mid,
            score=round(s * 100, 1)
        )
        for mid, s in candidates
        if s >= dynamic_threshold
    ]

    print(f"[AI] Search: '{req.prompt}' → {len(results)} results (best={round(best_score*100,1)}%, threshold={round(dynamic_threshold*100,1)}%)")
    return results


@app.post("/index-freelancer")
def index_freelancer(req: FreelancerIndexRequest):
    """Appelé par Spring Boot après création/mise à jour d'un freelancer."""
    text = freelancer_to_text(req)
    if req.id in freelancer_ids:
        # Mise à jour : on met à jour le texte et on reconstruit l'index
        freelancer_texts[req.id] = text
        rebuild_freelancer_index()
        print(f"[AI] Freelancer re-indexed: {req.id} | Total: {len(freelancer_ids)}")
        return {"status": "re_indexed", "id": req.id, "total": len(freelancer_ids)}
    add_freelancer_to_index(req.id, text)
    print(f"[AI] Freelancer indexed: {req.id} | Total: {len(freelancer_ids)}")
    return {"status": "indexed", "id": req.id, "total": len(freelancer_ids)}


@app.post("/search-freelancers", response_model=List[FreelancerSearchResult])
def search_freelancers(req: SearchRequest):
    """Recherche sémantique de freelancers par prompt entreprise."""
    if len(freelancer_ids) == 0:
        return []

    enriched_prompt = expand_prompt(req.prompt)
    query_vec = model.encode(enriched_prompt, normalize_embeddings=True)
    query_vec = np.array([query_vec], dtype=np.float32)

    k = min(req.top_k, len(freelancer_ids))
    scores, indices = freelancer_index.search(query_vec, k)

    # 1. Collecter tous les candidats au-dessus du seuil absolu minimum
    ABSOLUTE_MIN = 0.28
    candidates = [
        (freelancer_ids[idx], float(score))
        for score, idx in zip(scores[0], indices[0])
        if idx >= 0 and float(score) >= ABSOLUTE_MIN
    ]

    if not candidates:
        print(f"[AI] Freelancer search: '{req.prompt}' → 0 results")
        return []

    # 2. Seuil dynamique modéré : 60% du meilleur score
    #    → garde tous les profils similaires, élimine ceux vraiment hors-sujet
    best_score = candidates[0][1]
    dynamic_threshold = max(ABSOLUTE_MIN, best_score * 0.60)

    results = [
        FreelancerSearchResult(freelancer_id=fid, score=round(s * 100, 1))
        for fid, s in candidates
        if s >= dynamic_threshold
    ]

    print(f"[AI] Freelancer search: '{req.prompt}' → {len(results)} results "
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
