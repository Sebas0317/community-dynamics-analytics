"""Central configuration, group archetypes, exclusion rules, and local NLP lexicons."""

from __future__ import annotations

import re
from pathlib import Path

# === PATHS ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

DEFAULT_BASELINE_DB = WORKSPACE_ROOT / "Analisis_Ran_9_Nov" / "data" / "analytics.db"
DEFAULT_RAW_DIR = WORKSPACE_ROOT / "Analisis_Ran_9_Nov" / "exportaciones_raw"

PRIVATE_DIR = PROJECT_ROOT / "data_private"
STAGING_DB_PATH = PRIVATE_DIR / "staging_private.db"

PUBLIC_DIR = PROJECT_ROOT / "data_public"
PARQUET_DIR = PUBLIC_DIR / "parquet"
OUTPUTS_DIR = PUBLIC_DIR / "outputs"
DUCKDB_PATH = PUBLIC_DIR / "community_analytics.duckdb"
PRIVACY_MANIFEST_PATH = PUBLIC_DIR / "privacy_audit_manifest.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# === PRIVACY EXCLUSIONS (Chats 1-a-1 o privados jamás procesados) ===
# Generic exclusion tokens kept in public code; local personal exclusion names can be
# added privately in `data_private/local_exclusions.txt` (gitignored, one pattern per line).
_BASE_EXCLUDED_PATTERNS = [
    "chatadmin",
    "(admin)",
    "moderacion",
    "incidente",
    "privado",
    "olimpo",
    "funa",
    "arrebol",
    "elitistas",
]


def _load_excluded_patterns() -> list[str]:
    patterns = list(_BASE_EXCLUDED_PATTERNS)
    local_file = PRIVATE_DIR / "local_exclusions.txt"
    if local_file.exists():
        for line in local_file.read_text(encoding="utf-8").splitlines():
            tok = line.strip().lower()
            if tok and not tok.startswith("#"):
                patterns.append(tok)
    return patterns


EXCLUDED_CHAT_PATTERNS = _load_excluded_patterns()

# === ANONYMIZED GROUP ARCHETYPES ===
# Maps internal RP codes -> (group_id, display_label, macro_category, sub_archetype, bogota_zone)
GROUP_ARCHETYPES: dict[str, tuple[str, str, str, str, str]] = {
    "RP-001": ("G01_CORE_GENERAL", "Núcleo General", "Core", "Matriz Comunitaria", "Ciudad"),
    "RP-002": ("G02_SOCIAL_CHAT_LIBRE", "Chat Libre Social", "Social", "Conversación Abierta", "Ciudad"),
    "RP-003": ("G03_ONBOARDING_CENSO", "Canal Onboarding", "Onboarding", "Bienvenida y Presentación", "Ciudad"),
    "RP-004": ("G04_SOCIAL_NOCTURNA", "Vida Nocturna y Rumba", "Social", "Ocio Nocturno", "Ciudad"),
    "RP-005": ("G05_TEMATICO_GAMING", "Gaming y Juegos de Mesa", "Temático", "Juegos y Lúdica", "Ciudad"),
    "RP-006": ("G06_SOCIAL_HUMOR", "Humor y Desahogo", "Social", "Sátira y Catarsis", "Ciudad"),
    "RP-007": ("G07_GEO_NORTE", "Zona Norte Bogotá", "Geográfico", "Suba / Usaquén", "Norte"),
    "RP-008": ("G08_GEO_SUR", "Zona Sur Bogotá", "Geográfico", "Bosa / Kennedy", "Sur"),
    "RP-009": ("G09_GEO_CENTRO", "Zona Centro Bogotá", "Geográfico", "Centro / Chapinero", "Centro"),
    "RP-010": ("G10_TEMATICO_ARTE", "Arte y Fotografía", "Temático", "Cultura y Creatividad", "Ciudad"),
    "RP-011": ("G11_TEMATICO_CONCIERTOS", "Conciertos y Festivales", "Temático", "Música en Vivo", "Ciudad"),
    "RP-012": ("G12_TEMATICO_FITNESS", "Fitness y Deporte", "Temático", "Entrenamiento y Gym", "Ciudad"),
    "RP-013": ("G13_TEMATICO_ROCK", "Rock y Metal", "Temático", "Subcultura Musical", "Ciudad"),
    "RP-014": ("G14_TEMATICO_MOTOS", "Rodadas en Moto", "Temático", "Movilidad y Rutas", "Ciudad"),
    "RP-015": ("G15_TEMATICO_BAILE", "Baile y Salsa", "Temático", "Danza Social", "Ciudad"),
    "RP-016": ("G16_TEMATICO_GEEK", "Cultura Geek y Anime", "Temático", "Anime y Cómics", "Ciudad"),
    "RP-017": ("G17_TEMATICO_IDIOMAS", "Intercambio de Idiomas", "Temático", "Aprendizaje Social", "Ciudad"),
    "RP-018": ("G18_EFIMERO_FUTBOL", "Convocatoria Fútbol", "Efímero", "Evento Deportivo Puntual", "Ciudad"),
    "RP-019": ("G19_GEO_OCCIDENTE", "Zona Occidente Bogotá", "Geográfico", "Engativá / Occidente", "Occidente"),
    "RP-025": ("G20_EFIMERO_NAVIDAD", "Evento Festivo Temático", "Efímero", "Encuentro Masivo Puntual", "Ciudad"),
}


def resolve_group_archetype(code: str, raw_name: str = "") -> tuple[str, str, str, str, str]:
    """Resolve known RP codes or assign deterministic anonymized archetype for new groups."""
    if code in GROUP_ARCHETYPES:
        return GROUP_ARCHETYPES[code]

    name_l = raw_name.lower()
    m = re.search(r"(\d+)", code)
    num = int(m.group(1)) if m else 99
    prefix = f"G{num:02d}"

    if any(k in name_l for k in ["norte", "suba", "usaqu"]):
        return (f"{prefix}_GEO_NORTE_EXT", "Zona Norte (Ext)", "Geográfico", "Zonal Norte", "Norte")
    if any(k in name_l for k in ["sur", "kennedy", "bosa"]):
        return (f"{prefix}_GEO_SUR_EXT", "Zona Sur (Ext)", "Geográfico", "Zonal Sur", "Sur")
    if any(k in name_l for k in ["engativ", "occidente", "fontib"]):
        return (f"{prefix}_GEO_OCCIDENTE_EXT", "Zona Occidente (Ext)", "Geográfico", "Zonal Occidente", "Occidente")
    if any(k in name_l for k in ["centro", "chapinero", "teusaquillo"]):
        return (f"{prefix}_GEO_CENTRO_EXT", "Zona Centro (Ext)", "Geográfico", "Zonal Centro", "Centro")
    if any(k in name_l for k in ["presenta", "bienvenid", "onboarding"]):
        return (f"{prefix}_ONBOARDING_EXT", "Canal Onboarding (Ext)", "Onboarding", "Bienvenida", "Ciudad")
    if any(k in name_l for k in ["fiesta", "rumba", "simulacro", "halloween", "navid", "evento"]):
        return (f"{prefix}_EFIMERO_EVENTO", f"Evento Comunitario #{num}", "Efímero", "Evento Puntual", "Ciudad")
    return (f"{prefix}_TEMATICO_EXTRA", f"Subgrupo Temático #{num}", "Temático", "Nicho Comunitario", "Ciudad")


# === MEDIA EXTENSIONS ===
MEDIA_EXT_MAP = {
    ".webp": "sticker",
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".gif": "image",
    ".opus": "voice_note",
    ".ogg": "voice_note",
    ".m4a": "audio",
    ".mp3": "audio",
    ".wav": "audio",
    ".mp4": "video",
    ".mov": "video",
    ".mkv": "video",
    ".pdf": "document",
    ".docx": "document",
    ".doc": "document",
    ".xlsx": "document",
    ".pptx": "document",
    ".vcf": "contact",
}

# === LOCAL SPANISH / COLOMBIAN SENTIMENT LEXICON (100% offline) ===
POSITIVE_WORDS = {
    "bien", "bueno", "buena", "buenisimo", "buenísimo", "excelente", "genial", "chimba",
    "bacano", "bacana", "chévere", "chevere", "brutal", "increible", "increíble", "feliz",
    "gracias", "graciass", "mil", "amo", "encanta", "encantó", "encanto", "divertido",
    "divertida", "alegre", "super", "súper", "top", "mejor", "lindos", "linda", "lindo",
    "hermoso", "hermosa", "crack", "duro", "firme", "parche", "parchar", "vamos", "listo",
    "deuna", "hagale", "hágale", "bienvenido", "bienvenida", "bienvenidos", "bienvenidas",
    "un gusto", "gusto", "saludos", "éxito", "exito", "maravilloso", "perfecto", "perfecta",
    "felicitaciones", "felicidades", "abrazo", "rico", "rica", "delicioso", "deli", "espectacular",
    "apoyo", "recomiendo", "recomendado", "contento", "contenta", "emocionado", "emocionada",
    "motivado", "motivada", "sano", "tranqui", "elegante", "pulcro", "severo", "severa",
}

NEGATIVE_WORDS = {
    "mal", "malo", "mala", "horrible", "terrible", "pésimo", "pesimo", "triste", "aburrido",
    "aburrida", "odio", "rabia", "enojo", "enojado", "molesto", "molesta", "queja", "quejas",
    "fastidio", "pereza", "mamera", "cansado", "cansada", "agotado", "estresado", "estrés",
    "estres", "feo", "fea", "basura", "mierda", "puta", "gonorrea", "hp", "malparido",
    "grosero", "grosera", "tóxico", "toxico", "tóxica", "toxica", "chisme", "pelea", "drama",
    "cancelado", "cancelar", "falló", "fallo", "error", "problema", "problemas", "lástima",
    "lastima", "decepcion", "decepción", "decepcionado", "llorando", "dolor", "enfermo",
    "enferma", "paila", "pailas", "grave", "duro", "jodido", "jodida", "nunca", "nadie",
    "ignoran", "ignorado", "aburridos", "muerto", "murió", "murio", "peor", "fatal",
}

SPANISH_STOPWORDS = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para",
    "con", "no", "una", "su", "al", "lo", "como", "mas", "más", "pero", "sus", "le", "ya",
    "o", "este", "si", "sí", "porque", "esta", "está", "entre", "cuando", "muy", "sin",
    "sobre", "también", "tambien", "me", "hasta", "hay", "donde", "dónde", "quien", "quién",
    "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra", "otros", "ese",
    "eso", "ante", "ellos", "e", "esto", "mí", "mi", "antes", "algunos", "qué", "unos", "yo",
    "otro", "otras", "otra", "él", "tanto", "esa", "estos", "mucho", "quienes", "nada",
    "muchos", "cual", "poco", "ella", "estar", "estas", "algunas", "algo", "nosotros", "te",
    "ti", "tu", "tú", "tus", "ellas", "nosotras", "vosotros", "os", "mio", "tuyo", "suyo",
    "es", "son", "fue", "ser", "era", "soy", "eres", "somos", "va", "van", "voy", "vas",
    "vamos", "han", "he", "has", "ha", "hemos", "tengo", "tiene", "tienen", "tenemos",
    "hacer", "hace", "hago", "asi", "así", "aqui", "aquí", "ahi", "ahí", "alla", "allá",
    "pues", "entonces", "solo", "sólo", "bien", "mal", "siempre", "nunca", "ahora", "hoy",
    "ayer", "mañana", "vez", "veces", "puede", "pueden", "puedo", "creo", "ver", "voy",
    "jaja", "jajaja", "jajajaja", "jajajajaja", "jsjs", "jsjsjs", "jaj", "xd", "ok", "okey",
    "bueno", "hola", "holi", "holaa", "buenas", "dias", "días", "tardes", "noches", "gracias",
    "porfa", "favor", "igual", "tipo", "como", "osea", "sea", "aja", "ajá", "sip", "nop",
    "multimedia", "omitido", "archivo", "adjunto", "mensaje", "editado", "eliminó", "elimino",
    "https", "http", "www", "com", "co", "wa", "webp", "jpg", "opus", "mp4", "stk", "img",
    "ptt", "vid", "aud", "doc", "que", "q", "xq", "pq", "pa", "pal", "ps", "mk", "uy", "ay",
    "uh", "eh", "mm", "mmm", "ja", "je", "jeje", "ji", "jiji", "bro", "man", "wey", "parcero",
}

POSITIVE_EMOJIS = set("😀😃😄😁😆😅😂🤣😊😇🙂😉😌😍🥰😘😗😙😚😋😛😝😜🤪🥳🤩😎🙌👏🎉🎊🔥❤️🧡💛💚💙💜💖✨🌟💯👍👌💪🥂🍻🎶🎵🎸🕺💃")
NEGATIVE_EMOJIS = set("😞😔😟😕🙁☹️😣😖😫😩🥺😢😭😤😠😡🤬🤯😳😱😨😰😥😓💔👎😒🙄😑😐😴🤮🤢🤕🤒")

# === ANALYSIS CONSTANTS ===
REPLY_WINDOW_SECONDS = 180          # 3 min window for inferred SNA interaction edge
ONBOARDING_GROUP_ID = "G03_ONBOARDING_CENSO"
CORE_GROUP_ID = "G01_CORE_GENERAL"
PURGE_DATE_ISO = "2026-10-31"       # Halloween inactive purge event
PLAN_TRACTION_SHORT_HOURS = 2
PLAN_TRACTION_LONG_HOURS = 24
