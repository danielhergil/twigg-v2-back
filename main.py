import os
import json
import uuid
import asyncio
import re
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Body, Depends, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from jose import jwt
from jose.exceptions import ExpiredSignatureError, JWTError
from dotenv import load_dotenv
from fastapi import Path
from bs4 import BeautifulSoup


# ---- Load .env (local/dev) ----
load_dotenv()

# Optional: Supabase client
try:
    from supabase import create_client, Client as SupabaseClient
except Exception:  # pragma: no cover
    create_client = None
    SupabaseClient = None

APP_NAME = "coursegen-fastapi"

# OpenRouter config
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# Video search config - completely free alternatives
VIDEO_SEARCH_ENABLED = os.getenv("VIDEO_SEARCH_ENABLED", "true").lower() == "true"

# Supabase config
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Auth config (Supabase) with fallback derived from SUPABASE_URL
SUPABASE_JWKS_URL = os.getenv("SUPABASE_JWKS_URL") or (
    f"{SUPABASE_URL.rstrip('/')}/auth/v1/jwks" if SUPABASE_URL else None
)
SUPABASE_ISS = os.getenv("SUPABASE_ISS") or (
    f"{SUPABASE_URL.rstrip('/')}/auth/v1" if SUPABASE_URL else None
)
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

app = FastAPI(title="Course Generator API", version="1.1.1")

# CORS for local dev (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Models
# ---------------------------
class GenerateDraftIn(BaseModel):
    courseTitle: str
    level: str
    durationWeeks: int = Field(..., gt=0)
    prompt: str
    language: Optional[str] = Field(
        None,
        description="Force output language (e.g., 'es', 'en'). If omitted, model must mirror the body language.",
    )
    # createdBy removed from trust path; we will use JWT user id


class PublishIn(BaseModel):
    draftId: str
    # createdBy removed; we will use JWT user id

# Review models
class ReviewSubmission(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    title: Optional[str] = None
    body: str = Field(..., min_length=10, max_length=2000)
    is_anonymous: bool = False

class ReviewUpdate(BaseModel):
    rating: Optional[int] = Field(None, ge=1, le=5)
    title: Optional[str] = None
    body: Optional[str] = Field(None, min_length=10, max_length=2000)
    is_anonymous: Optional[bool] = None

class InstructorResponse(BaseModel):
    response: str = Field(..., min_length=10, max_length=1000)

class HelpfulVote(BaseModel):
    is_helpful: bool


# ---------------------------
# Auth helpers (Supabase JWT via JWKS)
# ---------------------------
_JWKS_CACHE: Optional[Dict[str, Any]] = None

async def _load_jwks() -> Dict[str, Any]:
    global _JWKS_CACHE
    if _JWKS_CACHE is None:
        if not SUPABASE_JWKS_URL:
            raise HTTPException(status_code=500, detail="SUPABASE_JWKS_URL not configured")
        headers = {}
        if SUPABASE_ANON_KEY:
            headers["apikey"] = SUPABASE_ANON_KEY
            headers["Authorization"] = f"Bearer {SUPABASE_ANON_KEY}"
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            r = await client.get(SUPABASE_JWKS_URL, headers=headers)
            try:
                r.raise_for_status()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Cannot fetch JWKS: {e}")
            _JWKS_CACHE = r.json()
    return _JWKS_CACHE

async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.split(" ", 1)[1]

    # Leer cabecera sin verificar para detectar algoritmo y KID
    try:
        header = jwt.get_unverified_header(token)
    except JWTError:
        raise HTTPException(status_code=401, detail="Malformed token header")

    alg = header.get("alg", "HS256")  # tu token actual viene con HS256
    kid = header.get("kid")
    iss = SUPABASE_ISS or (f"{SUPABASE_URL.rstrip('/')}/auth/v1" if SUPABASE_URL else None)

    # --- Ruta HS*: validar con el JWT SECRET del proyecto (HMAC) ---
    if alg.startswith("HS"):
        secret = os.getenv("SUPABASE_JWT_SECRET")
        if not secret:
            raise HTTPException(status_code=500, detail="SUPABASE_JWT_SECRET not configured (HS256)")
        audience = os.getenv("SUPABASE_JWT_AUDIENCE", "authenticated")
        try:
            claims = jwt.decode(
                token,
                secret,
                algorithms=[alg],          # HS256
                audience=audience,
                issuer=iss if iss else None,
            )
        except ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except JWTError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    # --- Ruta ES*/RS*: validar con JWKS (cuando uses Signing Keys) ---
    else:
        jwks = await _load_jwks()
        keys = jwks.get("keys", [])
        key = next((k for k in keys if k.get("kid") == kid), None) or (keys[0] if keys else None)
        if not key:
            raise HTTPException(status_code=401, detail="Matching JWK not found")
        try:
            claims = jwt.decode(
                token,
                key,                       # dict JWK; requiere python-jose[cryptography]
                algorithms=[alg],          # ES256 o RS256
                options={"verify_aud": False},
                issuer=iss if iss else None,
            )
        except ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except JWTError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    uid = claims.get("sub")
    if not uid:
        raise HTTPException(status_code=401, detail="Token missing sub (user id)")

    return {"id": uid, "claims": claims}


# ---------------------------
# Helpers
# ---------------------------

async def search_youtube_video_free(query: str, language: str = "en") -> Optional[str]:
    """
    Free YouTube video search using web scraping.
    No API key required, completely free to use.
    """
    if not VIDEO_SEARCH_ENABLED:
        return None
    
    try:
        # Encode the search query
        encoded_query = urllib.parse.quote_plus(query)
        
        # Add language parameter for better results
        lang_param = f"&lr=lang_{language}" if language != "en" else ""
        
        # YouTube search URL with filters for educational content
        search_url = f"https://www.youtube.com/results?search_query={encoded_query}{lang_param}&sp=EgIQAQ%253D%253D"  # Filter for videos only
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': f'{language}-US,{language};q=0.5' if language != "en" else 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0), follow_redirects=True) as client:
            response = await client.get(search_url, headers=headers)
            response.raise_for_status()
            
            # Parse the HTML response
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find video links using multiple methods for robustness
            video_patterns = [
                r'"videoId":"([a-zA-Z0-9_-]{11})"',
                r'/watch\?v=([a-zA-Z0-9_-]{11})',
                r'watch\?v=([a-zA-Z0-9_-]{11})'
            ]
            
            video_ids = []
            for pattern in video_patterns:
                matches = re.findall(pattern, response.text)
                video_ids.extend(matches)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_video_ids = []
            for video_id in video_ids:
                if video_id not in seen and len(video_id) == 11:  # YouTube video IDs are always 11 characters
                    seen.add(video_id)
                    unique_video_ids.append(video_id)
            
            if unique_video_ids:
                # Validate video quality by checking if multiple videos were found
                # (indicates relevant search results)
                if len(unique_video_ids) >= 2:
                    # Additional content validation: check if the search context matches expected content
                    if _validate_video_content_relevance(query, response.text):
                        # Return the first valid video URL
                        return f"https://www.youtube.com/watch?v={unique_video_ids[0]}"
                # If only one video found, be more cautious - might not be relevant
                
    except Exception as e:
        # Silently fail but log for debugging in development
        if os.getenv("DEBUG", "false").lower() == "true":
            print(f"Video search failed for query '{query}': {e}")
        pass
    
    return None

async def search_video_alternative_sources(query: str, language: str = "en") -> Optional[str]:
    """
    Alternative video sources for educational content.
    Uses multiple free platforms as fallback.
    """
    if not VIDEO_SEARCH_ENABLED:
        return None
    
    # Try YouTube first
    youtube_result = await search_youtube_video_free(query, language)
    if youtube_result:
        return youtube_result
    
    # Could add other free video platforms here:
    # - Vimeo search (free)
    # - Internet Archive videos
    # - Educational platform searches
    
    return None

def _validate_video_content_relevance(search_query: str, search_results_html: str) -> bool:
    """
    Validate that video search results are actually relevant to the search query.
    Helps prevent irrelevant content like kitchen videos for camping courses.
    """
    query_lower = search_query.lower()
    results_lower = search_results_html.lower()
    
    # Define conflicting content patterns that should disqualify results
    conflict_patterns = {
        # If searching for camping/outdoor content, reject kitchen/cooking content
        "camping": ["kitchen", "cooking", "recipe", "baking", "chef", "culinary", "food preparation"],
        "tent": ["kitchen", "interior design", "home decoration", "cooking", "recipe"],
        "outdoor": ["interior", "indoor", "kitchen", "home design", "cooking"],
        "hiking": ["kitchen", "cooking", "recipe", "interior design"],
        
        # If searching for cooking content, reject automotive content
        "cooking": ["automotive", "car repair", "mechanic", "engine"],
        "baking": ["automotive", "car repair", "mechanic", "engine"],
        "kitchen": ["automotive", "car repair", "mechanic", "camping equipment"],
        
        # If searching for automotive content, reject cooking content
        "automotive": ["cooking", "baking", "recipe", "kitchen", "culinary"],
        "car": ["cooking", "baking", "recipe", "kitchen design"],
        "mechanic": ["cooking", "baking", "recipe", "kitchen"],
        
        # General conflicts
        "art": ["automotive", "car repair", "kitchen appliance"],
        "music": ["automotive", "cooking equipment", "car repair"]
    }
    
    # Check if the search query contains any keywords that have known conflicts
    for query_keyword, conflicting_terms in conflict_patterns.items():
        if query_keyword in query_lower:
            # Count how many conflicting terms appear in results
            conflict_count = sum(1 for term in conflicting_terms if term in results_lower)
            relevant_count = results_lower.count(query_keyword)
            
            # If we find more conflicting content than relevant content, reject
            if conflict_count > relevant_count and conflict_count >= 3:
                if os.getenv("DEBUG", "false").lower() == "true":
                    print(f"Video search rejected: found {conflict_count} conflicting terms vs {relevant_count} relevant terms for '{query_keyword}'")
                return False
    
    # Additional specific validation for common mismatches
    if any(outdoor_term in query_lower for outdoor_term in ["camping", "tent", "outdoor", "hiking"]):
        # For outdoor queries, reject if we see too many kitchen/cooking indicators
        kitchen_indicators = ["recipe", "cooking", "kitchen design", "culinary", "chef", "food", "ingredient"]
        kitchen_count = sum(1 for indicator in kitchen_indicators if indicator in results_lower)
        if kitchen_count >= 5:  # Too many cooking-related results
            return False
    
    return True

def create_contextual_search_query(lesson_title: str, course_title: str, lesson_content: str = "") -> str:
    """
    Create a contextual search query that focuses on the specific technique/skill being taught.
    Avoids generic terms and focuses on actionable content.
    """
    lesson_lower = lesson_title.lower()
    course_lower = course_title.lower()
    content_lower = lesson_content.lower()
    
    # Extract the core technique or skill from the lesson
    technique_keywords = []
    
    # For cooking/bakery courses
    if any(cook_term in course_lower for cook_term in ["baking", "bakery", "cooking", "culinary", "chef"]):
        cooking_techniques = [
            "knead", "fold", "whip", "beat", "cream", "proof", "rise",
            "mixing", "dough", "batter", "icing", "frosting", "decoration",
            "piping", "rolling", "shaping", "cutting", "measuring", "tempering"
        ]
        technique_keywords = [kw for kw in cooking_techniques if kw in lesson_lower or kw in content_lower]
        if technique_keywords:
            return f"how to {technique_keywords[0]} baking cooking tutorial"
        else:
            # Look for specific baked goods or techniques
            for item in ["bread", "cake", "cookie", "pastry", "pie", "tart", "muffin"]:
                if item in lesson_lower:
                    return f"how to make {item} baking tutorial"
    
    # For mechanical courses
    elif any(mech_term in course_lower for mech_term in ["mechanic", "automotive", "car", "engine", "repair"]):
        mech_techniques = ["change", "replace", "install", "remove", "repair", "fix", "diagnostic", "maintenance"]
        technique_keywords = [kw for kw in mech_techniques if kw in lesson_lower or kw in content_lower]
        if technique_keywords:
            return f"how to {technique_keywords[0]} car automotive tutorial"
    
    # For art courses
    elif any(art_term in course_lower for art_term in ["art", "drawing", "painting", "design"]):
        art_techniques = ["draw", "paint", "sketch", "shade", "blend", "mix", "brush", "technique"]
        technique_keywords = [kw for kw in art_techniques if kw in lesson_lower or kw in content_lower]
        if technique_keywords:
            return f"{technique_keywords[0]} art technique tutorial"
    
    # For camping/outdoor equipment courses
    elif any(outdoor_term in course_lower for outdoor_term in ["camping", "tent", "outdoor", "hiking", "survival", "adventure", "equipment", "gear"]):
        camping_techniques = [
            "setup", "assembly", "pitching", "folding", "packing", "installation",
            "maintenance", "repair", "care", "storage", "operation", "use"
        ]
        technique_keywords = [kw for kw in camping_techniques if kw in lesson_lower or kw in content_lower]
        
        # Look for specific equipment types
        equipment_types = [
            "tent", "sleeping bag", "backpack", "stove", "lantern", "compass",
            "rope", "tarp", "shelter", "cookware", "water filter", "headlamp"
        ]
        equipment_keywords = [eq for eq in equipment_types if eq in lesson_lower or eq in content_lower]
        
        if technique_keywords and equipment_keywords:
            return f"how to {technique_keywords[0]} {equipment_keywords[0]} camping tutorial"
        elif equipment_keywords:
            return f"{equipment_keywords[0]} camping gear review setup tutorial"
        elif technique_keywords:
            return f"camping {technique_keywords[0]} outdoor tutorial"
        else:
            # Focus on structure and classification for camping equipment
            if any(classifier in lesson_lower for classifier in ["classifying", "classification", "structure", "type", "category"]):
                equipment_focus = next((eq for eq in equipment_types if eq in course_lower), "equipment")
                return f"camping {equipment_focus} types classification structure guide"
            else:
                return f"camping outdoor equipment tutorial"
    
    # Default: focus on the specific lesson technique + course context
    # Remove common filler words
    lesson_clean = lesson_lower.replace("how to", "").replace("introduction to", "").replace("basics of", "")
    course_clean = course_lower.replace("course", "").replace("class", "").replace("training", "")
    
    return f"{lesson_clean.strip()} {course_clean.strip()} tutorial".strip()

def should_include_video(lesson_title: str, course_title: str, lesson_content: str = "") -> bool:
    lesson_lower = lesson_title.lower()
    course_lower = course_title.lower()
    content_lower = lesson_content.lower()
    combined_text = f"{lesson_lower} {content_lower}".strip()
    
    # STRICT EXCLUSIONS - Never recommend videos for these
    exclusion_patterns = [
        # Introductory/Overview lessons
        "welcome", "introduction", "overview", "getting started", "course outline",
        "what you will learn", "learning objectives", "course goals", "expectations",
        "global vision", "general view", "course structure", "syllabus",
        
        # Theoretical/Conceptual lessons
        "theory", "concepts", "principles", "definitions", "terminology",
        "history", "background", "context", "evolution", "origins",
        "fundamentals", "basics", "foundation", "understanding",
        
        # Assessment/Evaluation
        "quiz", "test", "exam", "assessment", "evaluation", "review",
        "summary", "conclusion", "wrap up", "final thoughts",
        
        # Abstract topics
        "philosophy", "ethics", "methodology", "approach", "mindset",
        "importance", "benefits", "advantages", "why", "reasons",
        
        # Study/Learning methods
        "how to study", "learning tips", "study guide", "preparation",
        "study methods", "learning strategies", "note taking"
    ]
    
    # Check for exclusion patterns first
    for pattern in exclusion_patterns:
        if pattern in combined_text:
            return False
    
    # COURSE-SPECIFIC STRICT MATCHING
    # Only recommend videos if there's a strong match between course type and lesson content
    
    # Cooking/Bakery courses
    if any(cook_term in course_lower for cook_term in ["baking", "bakery", "cooking", "culinary", "chef", "kitchen", "recipe"]):
        cooking_video_keywords = [
            # Specific techniques that benefit from visual demonstration
            "knead", "fold", "whip", "beat", "cream", "proof", "rise",
            "temperature", "texture", "consistency", "mixing technique",
            "dough preparation", "batter", "icing", "frosting", "decoration",
            "piping", "rolling", "shaping", "cutting technique", "measuring",
            "ingredient preparation", "equipment use", "oven technique",
            "timing", "doneness test", "presentation", "plating"
        ]
        return any(keyword in combined_text for keyword in cooking_video_keywords)
    
    # Mechanical/Automotive courses
    elif any(mech_term in course_lower for mech_term in ["mechanic", "automotive", "car", "engine", "repair", "maintenance"]):
        mechanical_video_keywords = [
            "change", "replace", "install", "remove", "repair", "fix",
            "diagnostic", "troubleshoot", "assembly", "disassembly",
            "tools usage", "procedure", "step by step", "technique",
            "maintenance", "inspection", "adjustment", "calibration"
        ]
        return any(keyword in combined_text for keyword in mechanical_video_keywords)
    
    # Arts/Crafts courses
    elif any(art_term in course_lower for art_term in ["art", "craft", "drawing", "painting", "design", "creative"]):
        art_video_keywords = [
            "technique", "brush stroke", "color mixing", "shading", "blending",
            "texture", "composition", "perspective", "drawing", "sketching",
            "painting", "tool usage", "medium", "application", "demonstration"
        ]
        return any(keyword in combined_text for keyword in art_video_keywords)
    
    # Music courses
    elif any(music_term in course_lower for music_term in ["music", "instrument", "piano", "guitar", "violin", "singing"]):
        music_video_keywords = [
            "technique", "fingering", "posture", "breathing", "bow hold",
            "chord", "scale", "rhythm", "timing", "practice", "exercise",
            "performance", "playing", "demonstration"
        ]
        return any(keyword in combined_text for keyword in music_video_keywords)
    
    # Programming courses - Very selective
    elif any(prog_term in course_lower for prog_term in ["programming", "coding", "software", "development", "javascript", "python", "java", "c++", "html", "css"]):
        # Only for very practical, implementation-focused lessons
        programming_video_keywords = [
            "live coding", "code walkthrough", "debugging session", "setup", "installation",
            "deployment", "build process", "project creation", "IDE usage",
            "framework setup", "database connection", "API integration",
            "testing implementation", "code review", "refactoring example"
        ]
        return any(keyword in combined_text for keyword in programming_video_keywords)
    
    # Sports/Fitness courses
    elif any(sport_term in course_lower for sport_term in ["sport", "fitness", "exercise", "workout", "training", "yoga"]):
        sports_video_keywords = [
            "technique", "form", "movement", "exercise", "drill", "practice",
            "demonstration", "posture", "breathing", "routine", "workout"
        ]
        return any(keyword in combined_text for keyword in sports_video_keywords)
    
    # Medical/Health courses
    elif any(med_term in course_lower for med_term in ["medical", "health", "nursing", "therapy", "clinical"]):
        medical_video_keywords = [
            "procedure", "technique", "examination", "treatment", "demonstration",
            "clinical skill", "patient care", "hands-on", "practice", "protocol"
        ]
        return any(keyword in combined_text for keyword in medical_video_keywords)
    
    # Camping/Outdoor Equipment courses
    elif any(outdoor_term in course_lower for outdoor_term in ["camping", "tent", "outdoor", "hiking", "survival", "adventure", "equipment", "gear"]):
        camping_video_keywords = [
            "setup", "assembly", "installation", "pitching", "folding", "packing",
            "demonstration", "technique", "procedure", "step by step", "hands-on",
            "practical", "use", "operation", "maintenance", "care", "storage",
            "safety", "proper use", "equipment handling", "gear setup",
            "field test", "real world", "outdoor use", "weather protection",
            "durability test", "weight", "portability", "space efficiency"
        ]
        return any(keyword in combined_text for keyword in camping_video_keywords)
    
    # Default: Be very conservative - only recommend if explicitly practical
    ultra_practical_keywords = [
        "hands-on demonstration", "step-by-step guide", "visual tutorial",
        "practical exercise", "real-world application", "case study demonstration"
    ]
    
    return any(keyword in combined_text for keyword in ultra_practical_keywords)

def supabase_client() -> Optional["SupabaseClient"]:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY or not create_client:
        return None
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

async def openrouter_chat(
    messages: List[Dict[str, Any]], *, json_response: bool = True, temperature: float = 0.4
) -> Dict[str, Any]:
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # Optional attribution headers; safe to omit in local dev
        **({"HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER")} if os.getenv("OPENROUTER_HTTP_REFERER") else {}),
        **({"X-Title": os.getenv("OPENROUTER_X_TITLE", APP_NAME)} if os.getenv("OPENROUTER_X_TITLE") else {}),
    }
    body = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if json_response:
        body["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        r = await client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions", headers=headers, json=body
        )
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"OpenRouter error: {r.text}")
        data = r.json()
    try:
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception:
        # Fallback: return raw string (surface for debugging)
        return {"raw": data}

# Event Stream formatting (SSE)
def sse_event(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"

# ---------------------------
# Draft Generation (module-by-module SSE)
# ---------------------------
@app.post("/drafts/stream")
async def generate_draft_stream(
    body: GenerateDraftIn = Body(...),
    user=Depends(get_current_user),
):
    draft_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    async def event_generator():
        # 1) Announce start
        yield sse_event(
            "draft_started",
            {
                "draftId": draft_id,
                "startedAt": created_at,
            },
        )

        # 2) Ask the model for a plan (module outline)
        sys = {
            "role": "system",
            "content": (
                "You are a course writer that ALWAYS responds in pure JSON only. "
                "You will first create an outline of modules, then (when asked) expand a single module at a time. "
                "All output must be in the requested language. If no language is provided, mirror the language of the user's courseTitle."
            ),
        }
        user_outline = {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "outline",
                    "courseTitle": body.courseTitle,
                    "level": body.level,
                    "durationWeeks": body.durationWeeks,
                    "prompt": body.prompt,
                    "language": body.language,
                    "schema": {
                        "outline": {
                            "courseTitle": "string",
                            "level": "string",
                            "durationWeeks": "number",
                            "description": "string",
                            "modules": [
                                {
                                    "moduleNumber": "number",
                                    "moduleTitle": "string",
                                    "weeks": ["number", "number"],
                                }
                            ],
                        }
                    },
                },
                ensure_ascii=False,
            ),
        }
        outline = await openrouter_chat([sys, user_outline])
        if "raw" in outline:
            yield sse_event("error", {"message": "LLM returned non-JSON outline", "data": outline})
            return

        # 3) Emit outline immediately and store a draft shell in Supabase (optional)
        yield sse_event("outline", outline)

        sb = supabase_client()
        if sb:
            try:
                sb.table("course_drafts").insert(
                    {
                        "id": draft_id,
                        "status": "generating",
                        "draft": {
                            "draftId": draft_id,
                            "draft": {"id": draft_id, **outline, "modules": []},
                        },
                        "created_by": user.get("id"),
                    }
                ).execute()
            except Exception:
                pass

        # 4) For each module in outline, expand and stream
        modules = outline.get("modules", [])
        assembled_modules: List[Dict[str, Any]] = []

        for mod in modules:
            mod_prompt = {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "expand_module",
                        "language": body.language,
                        "course": {
                            "courseTitle": outline.get("courseTitle", body.courseTitle),
                            "level": outline.get("level", body.level),
                            "durationWeeks": outline.get("durationWeeks", body.durationWeeks),
                            "description": outline.get("description", body.prompt),
                        },
                        "module": mod,
                        "schema": {
                            "moduleNumber": "number",
                            "moduleTitle": "string",
                            "weeks": ["number", "number"],
                            "topics": [
                                {
                                    "topicTitle": "string",
                                    "lessons": [
                                        {
                                            "lessonTitle": "string",
                                            "theory": "string (extensive, detailed explanation with examples and practical applications - minimum 200 words)",
                                            "practicalTips": "string (actionable tips and best practices)",
                                            "videoRecommended": "boolean (true if a video would be helpful for this lesson)",
                                            "tests": [
                                                {
                                                    "question": "string",
                                                    "options": ["string"],
                                                    "answer": "string",
                                                    "solution": "string",
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        },
                    },
                    ensure_ascii=False,
                ),
            }
            expanded = await openrouter_chat([sys, mod_prompt])
            if "raw" in expanded:
                yield sse_event("error", {"message": "LLM returned non-JSON module", "data": expanded})
                return

            assembled_modules.append(expanded)
            # Stream this module to the client
            yield sse_event("module", expanded)

            # Push partial progress to Supabase draft
            if sb:
                try:
                    current = {
                        "draftId": draft_id,
                        "draft": {
                            "id": draft_id,
                            "courseTitle": outline.get("courseTitle", body.courseTitle),
                            "level": outline.get("level", body.level),
                            "durationWeeks": outline.get("durationWeeks", body.durationWeeks),
                            "description": outline.get("description", body.prompt),
                            "modules": assembled_modules,
                            "status": "generating",
                            "createdAt": created_at,
                            "updatedAt": datetime.now(timezone.utc).isoformat(),
                            "createdBy": user.get("id"),
                        },
                    }
                    sb.table("course_drafts").update({"draft": current}).eq("id", draft_id).execute()
                except Exception:
                    pass

            await asyncio.sleep(0)

        # 5) Finish draft and emit complete
        final_draft = {
            "draftId": draft_id,
            "draft": {
                "id": draft_id,
                "courseTitle": outline.get("courseTitle", body.courseTitle),
                "level": outline.get("level", body.level),
                "durationWeeks": outline.get("durationWeeks", body.durationWeeks),
                "description": outline.get("description", body.prompt),
                "modules": assembled_modules,
                "status": "draft",
                "createdAt": created_at,
                "updatedAt": datetime.now(timezone.utc).isoformat(),
                "createdBy": user.get("id"),
            },
        }

        if sb:
            try:
                sb.table("course_drafts").update(
                    {"status": "draft", "draft": final_draft}
                ).eq("id", draft_id).execute()
            except Exception:
                pass

        yield sse_event("complete", final_draft)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------
# Publish Route (persist to your existing Supabase schema)
# ---------------------------
@app.post("/drafts/publish")
async def publish_course(body: PublishIn, user=Depends(get_current_user)):
    sb = supabase_client()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    # 1) Cargar draft
    resp = (
        sb.table("course_drafts")
        .select("draft, created_by, status")
        .eq("id", body.draftId)
        .single()
        .execute()
    )
    draft_row = getattr(resp, "data", None) or getattr(resp, "json", None)
    if not draft_row:
        raise HTTPException(status_code=404, detail="Draft not found")
    if draft_row.get("created_by") and draft_row["created_by"] != user.get("id"):
        raise HTTPException(status_code=403, detail="You do not own this draft")

    draft = draft_row.get("draft")
    if not draft or "draft" not in draft:
        raise HTTPException(status_code=400, detail="Invalid draft payload")

    d = draft["draft"]

    # 2) Insert/Update course
    course_id = d.get("id") or str(uuid.uuid4())
    course_payload = {
        "id": course_id,
        "user_id": user.get("id"),
        "title": d["courseTitle"],
        "description": d.get("description"),
        "thumbnail_url": None,
        "is_published": True,
        "level": d.get("level"),
        "duration_weeks": d.get("durationWeeks"),
        "locale": d.get("language"),
    }
    try:
        sb.table("courses").insert(course_payload).execute()
    except Exception:
        sb.table("courses").update(
            {
                "title": course_payload["title"],
                "description": course_payload["description"],
                "is_published": True,
                "level": course_payload.get("level"),
                "duration_weeks": course_payload.get("duration_weeks"),
                "locale": course_payload.get("locale"),
            }
        ).eq("id", course_id).execute()

    # 3) Modules & Topics
    module_rows: List[Dict[str, Any]] = []
    topic_rows: List[Dict[str, Any]] = []

    for m in d.get("modules", []):
        module_id = str(uuid.uuid4())
        module_rows.append(
            {
                "id": module_id,
                "course_id": course_id,
                "module_number": m.get("moduleNumber"),
                "title": m.get("moduleTitle"),
                "weeks": m.get("weeks"),
            }
        )
        for t in m.get("topics", []):
            topic_id = str(uuid.uuid4())
            topic_rows.append(
                {
                    "id": topic_id,
                    "module_id": module_id,
                    "title": t.get("topicTitle"),
                }
            )

    if module_rows:
        sb.table("modules").insert(module_rows).execute()
    if topic_rows:
        sb.table("topics").insert(topic_rows).execute()

    # índice para localizar topic_id por (module_id, topicTitle)
    topic_index = {(tr["module_id"], tr["title"]): tr["id"] for tr in topic_rows}

    # 4) Lessons & Tests
    lesson_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    for m in d.get("modules", []):
        # encontrar el module_id recién insertado por module_number
        module_id = next(
            (mr["id"] for mr in module_rows if mr["module_number"] == m.get("moduleNumber")),
            None,
        )
        if not module_id:
            # si no lo encontramos, saltamos este módulo para evitar inserts huérfanos
            continue

        for t in m.get("topics", []):
            topic_id = topic_index.get((module_id, t.get("topicTitle")))
            if not topic_id:
                # seguridad: si por algún motivo no existe, lo creamos on-the-fly
                topic_id = str(uuid.uuid4())
                sb.table("topics").insert(
                    {"id": topic_id, "module_id": module_id, "title": t.get("topicTitle")}
                ).execute()
                topic_index[(module_id, t.get("topicTitle"))] = topic_id

            for idx, l in enumerate(t.get("lessons", []), start=1):
                lesson_id = str(uuid.uuid4())
                
                # Get YouTube video URL if recommended and API is available
                video_url = None
                lesson_title = l.get("lessonTitle", "")
                theory_content = l.get("theory", "")
                practical_tips = l.get("practicalTips", "")
                
                # Combine theory and practical tips for content
                full_content = f"{theory_content}\n\n{practical_tips}".strip()
                if not full_content:
                    full_content = theory_content or l.get("content", "")
                
                # Check if video is recommended and search for it
                if l.get("videoRecommended", False) or should_include_video(
                    lesson_title, 
                    d.get("courseTitle", ""), 
                    theory_content
                ):
                    # Create contextual search query focused on the specific technique
                    search_query = create_contextual_search_query(
                        lesson_title, 
                        d.get("courseTitle", ""), 
                        theory_content
                    )
                    video_url = await search_video_alternative_sources(search_query, d.get("language", "en"))
                
                lesson_rows.append(
                    {
                        "id": lesson_id,
                        "topic_id": topic_id,
                        "course_id": course_id,                      # tu tabla lo exige
                        "title": lesson_title,
                        "content": full_content,
                        "video_url": video_url,                      # Save YouTube URL
                        "language": d.get("language"),
                        "estimated_minutes": None,
                        "position": idx,                              # <-- FIX: NOT NULL
                        "is_published": True,                         # <-- si tu columna existe y es NOT NULL
                    }
                )

                for tst in l.get("tests", []):
                    test_rows.append(
                        {
                            "id": str(uuid.uuid4()),
                            "lesson_id": lesson_id,
                            "question": tst.get("question"),
                            "options": tst.get("options"),
                            "answer": tst.get("answer"),
                            "solution": tst.get("solution"),
                        }
                    )

    if lesson_rows:
        sb.table("lessons").insert(lesson_rows).execute()
    if test_rows:
        sb.table("tests").insert(test_rows).execute()

    # 5) Marcar draft como publicado
    sb.table("course_drafts").update({"status": "published"}).eq("id", body.draftId).execute()

    return {"ok": True, "courseId": course_id}

# ---------------------------
# Delete Course (deep delete)
# ---------------------------
@app.delete("/courses/{course_id}")
async def delete_course(
    course_id: str = Path(..., description="Course UUID"),
    user=Depends(get_current_user),
):
    sb = supabase_client()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    # 0) Verificar que el curso existe y pertenece al usuario
    resp = sb.table("courses").select("id,user_id").eq("id", course_id).single().execute()
    row = getattr(resp, "data", None) or getattr(resp, "json", None)
    if not row:
        raise HTTPException(status_code=404, detail="Course not found")
    if row.get("user_id") and row["user_id"] != user.get("id"):
        raise HTTPException(status_code=403, detail="You do not own this course")

    # Recolectar ids de módulos, topics y lessons para borrar en orden seguro
    # 1) módulos
    mods = sb.table("modules").select("id").eq("course_id", course_id).execute()
    mod_ids = [m["id"] for m in (mods.data or [])]

    # 2) topics (por módulo)
    topic_ids: List[str] = []
    if mod_ids:
        # borrar por lotes si hay muchos ids (PostgREST IN tiene límite ~ 2048 chars, pero aquí vale)
        topics = sb.table("topics").select("id").in_("module_id", mod_ids).execute()
        topic_ids = [t["id"] for t in (topics.data or [])]

    # 3) lessons (por topic)
    lesson_ids: List[str] = []
    if topic_ids:
        lessons = sb.table("lessons").select("id").in_("topic_id", topic_ids).execute()
        lesson_ids = [l["id"] for l in (lessons.data or [])]

    # 4) tests (por lesson) -> borrar primero los más profundos
    if lesson_ids:
        sb.table("tests").delete().in_("lesson_id", lesson_ids).execute()

    # 5) lessons
    if lesson_ids:
        sb.table("lessons").delete().in_("id", lesson_ids).execute()

    # 6) topics
    if topic_ids:
        sb.table("topics").delete().in_("id", topic_ids).execute()

    # 7) modules
    if mod_ids:
        sb.table("modules").delete().in_("id", mod_ids).execute()

    # 8) course
    sb.table("courses").delete().eq("id", course_id).execute()

    # 9) (opcional) borrar draft si coincide el id del curso
    try:
        sb.table("course_drafts").delete().eq("id", course_id).execute()
    except Exception:
        pass

    return {"ok": True, "deletedCourseId": course_id}


# ---------------------------
# Public Course Cards (for unauthenticated users)
# ---------------------------
class PublicCourseCard(BaseModel):
    id: str
    title: str
    thumbnail_url: Optional[str]
    level: Optional[str]
    duration_weeks: Optional[int]
    rating_avg: float
    reviews_count: int
    created_at: str
    instructor_name: str
    instructor_avatar: Optional[str]
    is_featured: bool

@app.get("/courses/public/cards", response_model=List[PublicCourseCard])
async def get_public_course_cards(
    search: Optional[str] = None,
    level: Optional[str] = None,
    sort_by: str = "popular",  # popular, rating, newest
    page: int = 1,
    limit: int = 12
):
    """
    Get course cards for public display (unauthenticated users).
    Returns only basic information needed for course browsing.
    """
    sb = supabase_client()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    # Calculate offset for pagination
    offset = (page - 1) * limit
    
    # Build query for published courses only
    query = sb.table("courses").select(
        "id, title, thumbnail_url, level, duration_weeks, rating_avg, reviews_count, created_at, user_id"
    ).eq("is_published", True)
    
    # Apply search filter if provided
    if search and search.strip():
        search_term = search.strip()
        query = query.or_(f"title.ilike.%{search_term}%")
    
    # Apply level filter if provided
    if level and level != "all":
        query = query.eq("level", level)
    
    # Apply sorting
    if sort_by == "rating":
        query = query.order("rating_avg", desc=True)
    elif sort_by == "newest":
        query = query.order("created_at", desc=True)
    else:  # popular (default)
        query = query.order("reviews_count", desc=True)
    
    # Apply pagination
    query = query.range(offset, offset + limit - 1)
    
    try:
        response = query.execute()
        courses_data = response.data or []
        
        # Get unique user IDs to fetch profiles
        user_ids = list(set(course.get("user_id") for course in courses_data if course.get("user_id")))
        
        # Fetch profiles for all instructors
        profiles_data = {}
        if user_ids:
            profiles_response = sb.table("profiles").select("id, first_name, last_name, avatar_url").in_("id", user_ids).execute()
            profiles_list = profiles_response.data or []
            profiles_data = {profile["id"]: profile for profile in profiles_list}
        
        # Transform data to match our model
        public_cards = []
        for course in courses_data:
            # Get instructor name
            instructor_name = "Unknown Instructor"
            instructor_avatar = None
            
            user_id = course.get("user_id")
            if user_id and user_id in profiles_data:
                profile = profiles_data[user_id]
                first_name = profile.get("first_name", "")
                last_name = profile.get("last_name", "")
                full_name = f"{first_name} {last_name}".strip()
                if full_name:
                    instructor_name = full_name
                instructor_avatar = profile.get("avatar_url")
            
            # Determine if course is featured
            rating_avg = course.get("rating_avg") or 0
            reviews_count = course.get("reviews_count") or 0
            is_featured = rating_avg >= 4.5 and reviews_count >= 10
            
            public_card = PublicCourseCard(
                id=course["id"],
                title=course["title"],
                thumbnail_url=course.get("thumbnail_url"),
                level=course.get("level"),
                duration_weeks=course.get("duration_weeks"),
                rating_avg=rating_avg,
                reviews_count=reviews_count,
                created_at=course["created_at"],
                instructor_name=instructor_name,
                instructor_avatar=instructor_avatar,
                is_featured=is_featured
            )
            public_cards.append(public_card)
        
        return public_cards
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch courses: {str(e)}")

@app.get("/courses/public/featured", response_model=List[PublicCourseCard])
async def get_featured_course_cards(limit: int = 6):
    """
    Get featured course cards for public display.
    Returns only courses with high ratings and review counts.
    """
    sb = supabase_client()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    try:
        # Get courses with high ratings and review counts
        response = sb.table("courses").select(
            "id, title, thumbnail_url, level, duration_weeks, rating_avg, reviews_count, created_at, user_id"
        ).eq("is_published", True).order("rating_avg", desc=True).limit(limit).execute()
        
        courses_data = response.data or []
        
        # Get unique user IDs to fetch profiles
        user_ids = list(set(course.get("user_id") for course in courses_data if course.get("user_id")))
        
        # Fetch profiles for all instructors
        profiles_data = {}
        if user_ids:
            profiles_response = sb.table("profiles").select("id, first_name, last_name, avatar_url").in_("id", user_ids).execute()
            profiles_list = profiles_response.data or []
            profiles_data = {profile["id"]: profile for profile in profiles_list}
        
        # Transform data to match our model
        featured_cards = []
        for course in courses_data:
            # Get instructor name
            instructor_name = "Unknown Instructor"
            instructor_avatar = None
            
            user_id = course.get("user_id")
            if user_id and user_id in profiles_data:
                profile = profiles_data[user_id]
                first_name = profile.get("first_name", "")
                last_name = profile.get("last_name", "")
                full_name = f"{first_name} {last_name}".strip()
                if full_name:
                    instructor_name = full_name
                instructor_avatar = profile.get("avatar_url")
            
            rating_avg = course.get("rating_avg") or 0
            reviews_count = course.get("reviews_count") or 0
            is_featured = rating_avg >= 4.5 and reviews_count >= 10
            
            featured_card = PublicCourseCard(
                id=course["id"],
                title=course["title"],
                thumbnail_url=course.get("thumbnail_url"),
                level=course.get("level"),
                duration_weeks=course.get("duration_weeks"),
                rating_avg=rating_avg,
                reviews_count=reviews_count,
                created_at=course["created_at"],
                instructor_name=instructor_name,
                instructor_avatar=instructor_avatar,
                is_featured=is_featured
            )
            featured_cards.append(featured_card)
        
        return featured_cards
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch featured courses: {str(e)}")

@app.get("/courses/public/{course_id}/preview")
async def get_public_course_preview(course_id: str):
    """
    Get limited course preview for unauthenticated users.
    Returns basic course information without sensitive content.
    """
    sb = supabase_client()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    try:
        # Get basic course information
        response = sb.table("courses").select(
            "id, title, thumbnail_url, level, duration_weeks, rating_avg, reviews_count, created_at, language, user_id"
        ).eq("id", course_id).eq("is_published", True).single().execute()
        
        course_data = response.data
        if not course_data:
            raise HTTPException(status_code=404, detail="Course not found")
        
        # Get instructor name
        instructor_name = "Unknown Instructor"
        instructor_avatar = None
        
        user_id = course_data.get("user_id")
        if user_id:
            profile_response = sb.table("profiles").select("first_name, last_name, avatar_url").eq("id", user_id).single().execute()
            if profile_response.data:
                profile = profile_response.data
                first_name = profile.get("first_name", "")
                last_name = profile.get("last_name", "")
                full_name = f"{first_name} {last_name}".strip()
                if full_name:
                    instructor_name = full_name
                instructor_avatar = profile.get("avatar_url")
        
        # Determine if course is featured
        rating_avg = course_data.get("rating_avg") or 0
        reviews_count = course_data.get("reviews_count") or 0
        is_featured = rating_avg >= 4.5 and reviews_count >= 10
        
        preview_data = {
            "id": course_data["id"],
            "title": course_data["title"],
            "description": "Sign up to see the full course description and curriculum details.",
            "thumbnail_url": course_data.get("thumbnail_url"),
            "level": course_data.get("level"),
            "duration_weeks": course_data.get("duration_weeks"),
            "rating_avg": rating_avg,
            "reviews_count": reviews_count,
            "created_at": course_data["created_at"],
            "language": course_data.get("language", "English"),
            "is_featured": is_featured,
            "instructor": {
                "name": instructor_name,
                "avatar_url": instructor_avatar
            },
            "modules": []  # Empty for unauthenticated users
        }
        
        return preview_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch course preview: {str(e)}")

# ---------------------------
# Reviews API Endpoints
# ---------------------------

@app.get("/courses/{course_id}/reviews")
async def get_course_reviews(
    course_id: str,
    page: int = 1,
    limit: int = 10,
    sort: str = "newest"  # newest, oldest, highest, lowest, helpful
):
    """Get paginated reviews for a course"""
    sb = supabase_client()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    # Calculate offset
    offset = (page - 1) * limit
    
    # Build query
    query = sb.table("course_reviews").select(
        "id, rating, title, body, is_anonymous, created_at, updated_at, is_edited, helpful_count, "
        "profiles!course_reviews_user_id_fkey(id, first_name, last_name, avatar_url)"
    ).eq("course_id", course_id)
    
    # Apply sorting
    if sort == "oldest":
        query = query.order("created_at", desc=False)
    elif sort == "highest":
        query = query.order("rating", desc=True)
    elif sort == "lowest":
        query = query.order("rating", desc=False)
    elif sort == "helpful":
        query = query.order("helpful_count", desc=True)
    else:  # newest (default)
        query = query.order("created_at", desc=True)
    
    # Apply pagination
    query = query.range(offset, offset + limit - 1)
    
    try:
        response = query.execute()
        reviews = response.data or []
        
        # Get instructor responses for these reviews
        if reviews:
            review_ids = [review["id"] for review in reviews]
            responses_query = sb.table("review_responses").select(
                "review_id, response, created_at, "
                "profiles!review_responses_instructor_id_fkey(first_name, last_name, avatar_url)"
            ).in_("review_id", review_ids)
            
            responses_result = responses_query.execute()
            responses_data = responses_result.data or []
            
            # Create a map of review_id to response
            responses_map = {resp["review_id"]: resp for resp in responses_data}
            
            # Add responses to reviews
            for review in reviews:
                review["instructor_response"] = responses_map.get(review["id"])
        
        return {
            "reviews": reviews,
            "page": page,
            "limit": limit,
            "has_more": len(reviews) == limit
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch reviews: {str(e)}")

@app.post("/courses/{course_id}/reviews")
async def submit_review(
    course_id: str,
    review: ReviewSubmission,
    user=Depends(get_current_user)
):
    """Submit a new review for a course"""
    sb = supabase_client()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    user_id = user.get("id")
    
    try:
        # Check if user is enrolled in the course (you might need to implement this check)
        # For now, we'll skip this check but you should add enrollment verification
        
        # Check if user already reviewed this course
        existing_review = sb.table("course_reviews").select("id").eq("course_id", course_id).eq("user_id", user_id).execute()
        if existing_review.data:
            raise HTTPException(status_code=400, detail="You have already reviewed this course")
        
        # Check if user is the course instructor
        course_check = sb.table("courses").select("user_id").eq("id", course_id).single().execute()
        if course_check.data and course_check.data["user_id"] == user_id:
            raise HTTPException(status_code=400, detail="You cannot review your own course")
        
        # Insert the review
        review_data = {
            "course_id": course_id,
            "user_id": user_id,
            "rating": review.rating,
            "title": review.title,
            "body": review.body,
            "is_anonymous": review.is_anonymous
        }
        
        result = sb.table("course_reviews").insert(review_data).execute()
        
        if result.data:
            return {"message": "Review submitted successfully", "review_id": result.data[0]["id"]}
        else:
            raise HTTPException(status_code=500, detail="Failed to submit review")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit review: {str(e)}")

@app.put("/reviews/{review_id}")
async def update_review(
    review_id: str,
    review_update: ReviewUpdate,
    user=Depends(get_current_user)
):
    """Update an existing review"""
    sb = supabase_client()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    user_id = user.get("id")
    
    try:
        # Check if review exists and belongs to user
        existing_review = sb.table("course_reviews").select("*").eq("id", review_id).eq("user_id", user_id).single().execute()
        if not existing_review.data:
            raise HTTPException(status_code=404, detail="Review not found or you don't have permission to edit it")
        
        # Build update data
        update_data = {"updated_at": datetime.now(timezone.utc).isoformat(), "is_edited": True}
        
        if review_update.rating is not None:
            update_data["rating"] = review_update.rating
        if review_update.title is not None:
            update_data["title"] = review_update.title
        if review_update.body is not None:
            update_data["body"] = review_update.body
        if review_update.is_anonymous is not None:
            update_data["is_anonymous"] = review_update.is_anonymous
        
        # Update the review
        result = sb.table("course_reviews").update(update_data).eq("id", review_id).execute()
        
        return {"message": "Review updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update review: {str(e)}")

@app.delete("/reviews/{review_id}")
async def delete_review(
    review_id: str,
    user=Depends(get_current_user)
):
    """Delete a review"""
    sb = supabase_client()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    user_id = user.get("id")
    
    try:
        # Check if review exists and belongs to user
        existing_review = sb.table("course_reviews").select("*").eq("id", review_id).eq("user_id", user_id).single().execute()
        if not existing_review.data:
            raise HTTPException(status_code=404, detail="Review not found or you don't have permission to delete it")
        
        # Delete the review (cascade will handle responses and helpfulness votes)
        sb.table("course_reviews").delete().eq("id", review_id).execute()
        
        return {"message": "Review deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete review: {str(e)}")

# Health
@app.get("/")
async def health():
    return {"ok": True, "name": APP_NAME}

# Test endpoint for video search (development only)
@app.get("/test/video-search")
async def test_video_search(
    lesson_title: str = "Welcome and Global Vision to this Course", 
    course_title: str = "Professional Bakery Course",
    lang: str = "en"
):
    """
    Test endpoint to verify video search functionality.
    Usage: GET /test/video-search?lesson_title=lesson&course_title=course&lang=en
    """
    if os.getenv("DEBUG", "false").lower() != "true":
        raise HTTPException(status_code=404, detail="Endpoint not available in production")
    
    should_include = should_include_video(lesson_title, course_title, lesson_title)
    
    video_url = None
    search_query = None
    if should_include:
        search_query = create_contextual_search_query(lesson_title, course_title, lesson_title)
        video_url = await search_video_alternative_sources(search_query, lang)
    
    return {
        "lesson_title": lesson_title,
        "course_title": course_title,
        "language": lang,
        "should_include_video": should_include,
        "search_query": search_query,
        "video_found": video_url,
        "search_enabled": VIDEO_SEARCH_ENABLED
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    # Disable reload in production (when PORT is set by platform)
    reload = os.getenv("PORT") is None
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload)
