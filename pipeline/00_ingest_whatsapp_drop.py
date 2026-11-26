"""Step 00: Private WhatsApp Drop Ingestion & Multimodal Metadata Scanner.

Supports two execution modes:
1. Baseline mode (default): Reads existing `analytics.db` + scans `exportaciones_raw`
   for physical media files (.webp, .jpg, .opus, .mp4, .pdf, etc.) and writes a clean
   private staging SQLite DB (`data_private/staging_private.db`).
2. Fresh Friday Drop mode (`--raw-dir <path>`): Unzips any `.zip` exports in `--raw-dir`,
   filters out private/excluded chats, parses all WhatsApp `.txt` files incrementally
   (with optional `--merge-baseline` deduplication), scans all media files, and builds
   `data_private/staging_private.db`.

Everything written by this script stays strictly inside `data_private/` (gitignored).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sqlite3
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from mutagen import File as MutagenFile
from PIL import Image

from pipeline.config import (
    DEFAULT_BASELINE_DB,
    DEFAULT_RAW_DIR,
    EXCLUDED_CHAT_PATTERNS,
    MEDIA_EXT_MAP,
    PRIVATE_DIR,
    STAGING_DB_PATH,
)

# WhatsApp timestamp regex covering Android & iOS (ES/EN, 12h/24h, brackets or dash)
TS_DASH_RE = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4}),?\s+"
    r"(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[\u202f\s]*(?:a\.\s*m\.|p\.\s*m\.|AM|PM))?)"
    r"\s*-\s*(.*)$",
    re.IGNORECASE,
)
TS_BRACKET_RE = re.compile(
    r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s+"
    r"(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[\u202f\s]*(?:a\.\s*m\.|p\.\s*m\.|AM|PM))?)\]\s*(.*)$",
    re.IGNORECASE,
)
ATTACHMENT_RE = re.compile(
    r"([\w\-.]+\.(?:webp|jpg|jpeg|png|gif|opus|ogg|mp4|mov|mp3|m4a|wav|pdf|docx?|xlsx?|pptx?|vcf))\s*"
    r"(?:\(archivo adjunto\)|\(file attached\)|<adjunto:)?",
    re.IGNORECASE,
)
DAY_NAMES_ES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


def is_excluded_chat(name: str) -> bool:
    low = name.lower()
    return any(pat in low for pat in EXCLUDED_CHAT_PATTERNS)


def parse_wa_datetime(d_str: str, t_str: str) -> datetime | None:
    try:
        t_clean = t_str.replace("\u202f", " ").replace("\xa0", " ").strip().lower()
        t_clean = re.sub(r"\s+", " ", t_clean)
        is_pm = "p. m." in t_clean or "pm" in t_clean or "p.m." in t_clean
        is_am = "a. m." in t_clean or "am" in t_clean or "a.m." in t_clean
        t_digits = re.sub(r"[^\d:]", "", t_clean)
        t_parts = [int(p) for p in t_digits.split(":")]
        h = t_parts[0]
        m = t_parts[1]
        s = t_parts[2] if len(t_parts) > 2 else 0
        if is_pm and h < 12:
            h += 12
        elif is_am and h == 12:
            h = 0
        d_parts = [int(p) for p in d_str.split("/")]
        day, month, year = d_parts[0], d_parts[1], d_parts[2]
        if month > 12 and day <= 12:
            day, month = month, day
        if year < 100:
            year += 2000
        return datetime(year, month, day, h, m, s)
    except Exception:
        return None


def normalize_sender(raw_sender: str) -> str:
    if not raw_sender:
        return ""
    s = raw_sender.replace("\u200e", "").replace("\u202f", " ").replace("\xa0", " ").strip()
    if re.match(r"^\+?\d[\d\s-]{7,}\d$", s):
        clean_ph = re.sub(r"[\s-]", "", s)
        if not clean_ph.startswith("+"):
            clean_ph = "+" + clean_ph
        return clean_ph
    return s


def classify_message_category(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["pola", "cerveza", "bar", "rumba", "fiesta", "trago", "chiva", "licor", "shots"]):
        return "Fiesta y Vida Nocturna"
    if any(w in t for w in ["comida", "almuerzo", "cena", "hamburguesa", "pizza", "restaurante", "café", "cafe", "sushi"]):
        return "Gastronomía y Salidas"
    if any(w in t for w in ["fútbol", "futbol", "gym", "calistenia", "entrenamiento", "partido", "cancha", "trotar", "senderismo", "bici", "rodada", "moto"]):
        return "Deportes y Actividad Física"
    if any(w in t for w in ["juegos", "bolirana", "billar", "cartas", "uno", "rol", "videojuegos", "play", "xbox", "switch", "boardgames", "lúdica"]):
        return "Juegos y Entretenimiento"
    if any(w in t for w in ["cine", "pelicula", "película", "museo", "expo", "arte", "pintura", "teatro", "salsa", "bachata", "baile", "anime", "manga", "rock", "metal", "concierto", "festival"]):
        return "Cultura, Arte y Música"
    if any(w in t for w in ["english", "ingles", "inglés", "spanglish", "language", "practice"]):
        return "Idiomas"
    if any(w in t for w in ["plan", "planes", "parche", "parchar", "caigan", "quién se anima", "quien se anima", "nos vemos", "punto de encuentro"]):
        return "Convocatoria y Planes"
    return "General / Charla"


def is_plan_proposal(text: str) -> bool:
    t = text.lower()
    triggers = [
        "plan ", "planes", "parche", "parchar", "quién se anima", "quien se anima",
        "caigan", "armemos", "hagamos plan", "propongo", "punto de encuentro",
        "nos vemos en", "quien cae", "quién cae", "salida ", "rodada", "partido ",
    ]
    return len(t.split()) >= 4 and any(tr in t for tr in triggers)


def extract_media_metadata(filepath: Path) -> dict:
    """Extract purely technical metadata from a local media file without storing content."""
    ext = filepath.suffix.lower()
    subtype = MEDIA_EXT_MAP.get(ext, "other")
    size_bytes = filepath.stat().st_size

    # SHA-256 content digest (used later with HMAC for anonymized viral sticker matching)
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    raw_sha256 = h.hexdigest()

    width: int | None = None
    height: int | None = None
    is_animated: int = 0
    duration_sec: float | None = None

    if subtype in ("sticker", "image"):
        try:
            with Image.open(filepath) as img:
                width, height = img.size
                is_animated = 1 if getattr(img, "is_animated", False) else 0
        except Exception:
            pass
    elif subtype in ("voice_note", "audio", "video"):
        try:
            mf = MutagenFile(filepath)
            if mf is not None and mf.info is not None and hasattr(mf.info, "length"):
                duration_sec = round(float(mf.info.length), 2)
        except Exception:
            pass

    aspect_ratio = round(width / height, 3) if (width and height and height > 0) else None

    return {
        "filename": filepath.name,
        "ext": ext,
        "media_subtype": subtype,
        "file_size_bytes": size_bytes,
        "raw_sha256": raw_sha256,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "is_animated": is_animated,
        "duration_sec": duration_sec,
    }


def scan_media_directory(raw_dir: Path, folder_to_code: dict[str, str]) -> list[dict]:
    """Walk raw_dir and extract technical metadata for every physical media file."""
    records: list[dict] = []
    if not raw_dir.exists():
        print(f"[WARN] Raw media directory not found: {raw_dir}")
        return records

    norm_folder_map = {k.lower().strip(): v for k, v in folder_to_code.items()}

    for root, _, files in os.walk(raw_dir):
        root_path = Path(root)
        # Resolve group_code from relative top folder
        try:
            rel_parts = root_path.relative_to(raw_dir).parts
            top_folder = rel_parts[0] if rel_parts else ""
        except ValueError:
            top_folder = root_path.name

        if is_excluded_chat(top_folder):
            continue

        group_code = norm_folder_map.get(top_folder.lower().strip(), "")
        if not group_code:
            # Fuzzy match by substring if folder name slightly changed
            tf_low = top_folder.lower()
            for fk, code in norm_folder_map.items():
                if fk and (fk in tf_low or tf_low in fk):
                    group_code = code
                    break

        for fname in files:
            ext = Path(fname).suffix.lower()
            if ext not in MEDIA_EXT_MAP:
                continue
            fpath = root_path / fname
            try:
                meta = extract_media_metadata(fpath)
                meta["group_code"] = group_code
                meta["source_folder"] = top_folder
                records.append(meta)
            except Exception:
                continue

    return records


def init_staging_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = NORMAL;")

    cur.executescript(
        """
        CREATE TABLE groups (
            code TEXT PRIMARY KEY,
            name TEXT,
            category TEXT,
            folder TEXT
        );

        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_code TEXT NOT NULL,
            msg_index INTEGER NOT NULL,
            dt_iso TEXT NOT NULL,
            date_str TEXT NOT NULL,
            hour INTEGER NOT NULL,
            day_of_week INTEGER NOT NULL,
            sender_norm TEXT,
            is_system INTEGER NOT NULL,
            is_media INTEGER NOT NULL,
            is_deleted INTEGER NOT NULL,
            message_type TEXT NOT NULL,
            content TEXT NOT NULL,
            word_count INTEGER NOT NULL,
            category TEXT NOT NULL,
            attachment_filename TEXT
        );

        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_code TEXT NOT NULL,
            dt_iso TEXT NOT NULL,
            event_type TEXT NOT NULL,
            actor TEXT,
            target TEXT,
            description TEXT,
            message_id INTEGER
        );

        CREATE TABLE plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_code TEXT NOT NULL,
            dt_iso TEXT NOT NULL,
            date_str TEXT NOT NULL,
            category TEXT NOT NULL,
            initiator TEXT,
            message_id INTEGER
        );

        CREATE TABLE media_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_code TEXT,
            source_folder TEXT,
            filename TEXT NOT NULL,
            ext TEXT NOT NULL,
            media_subtype TEXT NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            raw_sha256 TEXT NOT NULL,
            width INTEGER,
            height INTEGER,
            aspect_ratio REAL,
            is_animated INTEGER DEFAULT 0,
            duration_sec REAL
        );

        CREATE INDEX idx_stg_msg_grp_dt ON messages (group_code, dt_iso);
        CREATE INDEX idx_stg_msg_attach ON messages (group_code, attachment_filename);
        CREATE INDEX idx_stg_media_lookup ON media_files (group_code, filename);
        CREATE INDEX idx_stg_media_fname ON media_files (filename);
        """
    )
    conn.commit()
    return conn


def extract_attachment_filename(content: str) -> str | None:
    if not content:
        return None
    m = ATTACHMENT_RE.search(content)
    return m.group(1).strip() if m else None


def load_from_baseline_db(staging_conn: sqlite3.Connection, baseline_db: Path) -> dict[str, str]:
    """Copy community groups, messages, events, and plans from baseline analytics.db."""
    src = sqlite3.connect(baseline_db)
    scur = src.cursor()
    dcur = staging_conn.cursor()

    folder_to_code: dict[str, str] = {}
    groups = scur.execute("SELECT code, name, category, folder FROM groups").fetchall()
    valid_codes: set[str] = set()
    for code, name, cat, folder in groups:
        if is_excluded_chat(name or "") or is_excluded_chat(folder or ""):
            continue
        valid_codes.add(code)
        if folder:
            folder_to_code[folder] = code
        dcur.execute(
            "INSERT INTO groups (code, name, category, folder) VALUES (?, ?, ?, ?)",
            (code, name, cat, folder),
        )

    # Messages (preserve original id so plans.message_id / events.message_id stay aligned)
    rows = scur.execute(
        """
        SELECT id, group_code, msg_index, dt_iso, date_str, hour, day_of_week,
               sender_norm, is_system, is_media, is_deleted, message_type,
               content, word_count, category
        FROM messages
        """
    ).fetchall()
    msg_batch = []
    for r in rows:
        if r[1] not in valid_codes:
            continue
        attach = extract_attachment_filename(r[12]) if r[9] == 1 else None
        msg_batch.append((*r, attach))

    dcur.executemany(
        """
        INSERT INTO messages (
            id, group_code, msg_index, dt_iso, date_str, hour, day_of_week,
            sender_norm, is_system, is_media, is_deleted, message_type,
            content, word_count, category, attachment_filename
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        msg_batch,
    )

    ev_rows = scur.execute(
        "SELECT id, group_code, dt_iso, event_type, actor, target, description, message_id FROM events"
    ).fetchall()
    dcur.executemany(
        "INSERT INTO events (id, group_code, dt_iso, event_type, actor, target, description, message_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [r for r in ev_rows if r[1] in valid_codes],
    )

    pl_rows = scur.execute(
        "SELECT id, group_code, dt_iso, date_str, category, initiator, message_id FROM plans"
    ).fetchall()
    dcur.executemany(
        "INSERT INTO plans (id, group_code, dt_iso, date_str, category, initiator, message_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [r for r in pl_rows if r[1] in valid_codes],
    )

    staging_conn.commit()
    src.close()
    print(f"[INGEST] Baseline loaded: {len(valid_codes)} groups, {len(msg_batch):,} messages, {len(ev_rows):,} events, {len(pl_rows):,} plans.")
    return folder_to_code


def classify_system_event(body: str) -> str:
    b = body.lower()
    if any(k in b for k in ["se unió", "se unio", "añadió", "anadio", "joined", "added"]):
        return "join"
    if any(k in b for k in ["salió", "salio", "left"]):
        return "leave"
    if any(k in b for k in ["eliminó a", "elimino a", "removed"]):
        return "kick"
    if any(k in b for k in ["fijó un mensaje", "fijo un mensaje", "pinned a message"]):
        return "pin"
    if any(k in b for k in ["ícono del grupo", "icono del grupo", "group icon"]):
        return "icon"
    if any(k in b for k in ["descripción del grupo", "descripcion del grupo", "group description"]):
        return "desc"
    return "other"


def parse_txt_export(txt_path: Path, group_code: str) -> list[dict]:
    """Parse a standard WhatsApp .txt export into structured message dicts."""
    messages: list[dict] = []
    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    current: dict | None = None
    for raw_line in lines:
        line = raw_line.rstrip("\r\n").lstrip("\ufeff\u200e")
        m = TS_DASH_RE.match(line) or TS_BRACKET_RE.match(line)
        if m:
            if current is not None:
                messages.append(current)
            dt = parse_wa_datetime(m.group(1), m.group(2))
            if dt is None:
                current = None
                continue
            rest = m.group(3)
            if ": " in rest:
                sender_raw, content = rest.split(": ", 1)
                is_system = 0
                sender_norm = normalize_sender(sender_raw)
            else:
                sender_norm = ""
                content = rest
                is_system = 1

            current = {
                "group_code": group_code,
                "dt": dt,
                "dt_iso": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "date_str": dt.strftime("%Y-%m-%d"),
                "hour": dt.hour,
                "day_of_week": dt.weekday(),
                "sender_norm": sender_norm,
                "is_system": is_system,
                "content": content,
            }
        elif current is not None:
            current["content"] += "\n" + line

    if current is not None:
        messages.append(current)
    return messages


def ingest_fresh_drop(
    staging_conn: sqlite3.Connection,
    drop_dir: Path,
    existing_folder_map: dict[str, str],
) -> dict[str, str]:
    """Unzip and incrementally merge a fresh Friday WhatsApp drop into staging_private.db."""
    cur = staging_conn.cursor()
    temp_extract = Path(tempfile.mkdtemp(prefix="wa_drop_"))
    try:
        # Copy / unzip everything into working view
        for item in drop_dir.iterdir():
            if is_excluded_chat(item.name):
                continue
            if item.suffix.lower() == ".zip":
                dest = temp_extract / item.stem
                dest.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(item, "r") as zf:
                    zf.extractall(dest)
            elif item.is_dir():
                shutil.copytree(item, temp_extract / item.name, dirs_exist_ok=True)

        # Build existing dedup signature set: (group_code, dt_iso, sender_norm, content[:60])
        existing_sigs = set(
            cur.execute(
                "SELECT group_code, dt_iso, COALESCE(sender_norm, ''), substr(content, 1, 60) FROM messages"
            ).fetchall()
        )
        max_idx_by_group = dict(
            cur.execute("SELECT group_code, COALESCE(MAX(msg_index), 0) FROM messages GROUP BY group_code").fetchall()
        )
        existing_groups = {
            r[0]: (r[1] or "").lower() for r in cur.execute("SELECT code, name FROM groups").fetchall()
        }
        next_rp_num = max(
            [int(re.search(r"(\d+)", c).group(1)) for c in existing_groups if re.search(r"(\d+)", c)] + [25]
        ) + 1

        folder_to_code = dict(existing_folder_map)
        added_msgs = 0
        added_events = 0
        added_plans = 0

        for txt_path in temp_extract.rglob("*.txt"):
            folder_name = txt_path.parent.name
            clean_chat_name = re.sub(r"^Chat de WhatsApp con\s+", "", txt_path.stem, flags=re.IGNORECASE).strip()
            if is_excluded_chat(folder_name) or is_excluded_chat(clean_chat_name):
                continue

            # Match to existing group code or create a new one
            matched_code = None
            cn_low = clean_chat_name.lower()
            for code, gname in existing_groups.items():
                if gname and (gname == cn_low or gname[:14] in cn_low or cn_low[:14] in gname):
                    matched_code = code
                    break
            if not matched_code:
                matched_code = f"RP-{next_rp_num:03d}"
                next_rp_num += 1
                existing_groups[matched_code] = cn_low
                cur.execute(
                    "INSERT OR IGNORE INTO groups (code, name, category, folder) VALUES (?, ?, ?, ?)",
                    (matched_code, clean_chat_name, "Comunitario", folder_name),
                )
            folder_to_code[folder_name] = matched_code

            parsed = parse_txt_export(txt_path, matched_code)
            idx = max_idx_by_group.get(matched_code, 0)
            for msg in parsed:
                sig = (matched_code, msg["dt_iso"], msg["sender_norm"], msg["content"][:60])
                if sig in existing_sigs:
                    continue
                existing_sigs.add(sig)
                idx += 1

                content = msg["content"]
                c_low = content.lower()
                is_deleted = 1 if ("eliminó este mensaje" in c_low or "eliminaste este mensaje" in c_low or "deleted this message" in c_low) else 0
                attach = extract_attachment_filename(content)
                is_media = 1 if (attach is not None or "multimedia omitido" in c_low or "media omitted" in c_low) else 0

                if msg["is_system"]:
                    ev_type = classify_system_event(content)
                    mtype = f"system_{ev_type}"
                    cat = "Sistema y Moderación"
                    wc = 0
                elif is_deleted:
                    mtype = "deleted"
                    cat = "Sistema y Moderación"
                    wc = 0
                elif is_media:
                    mtype = "media"
                    cat = "General / Charla"
                    wc = 0
                else:
                    mtype = "text"
                    cat = classify_message_category(content)
                    wc = len(content.split())

                cur.execute(
                    """
                    INSERT INTO messages (
                        group_code, msg_index, dt_iso, date_str, hour, day_of_week,
                        sender_norm, is_system, is_media, is_deleted, message_type,
                        content, word_count, category, attachment_filename
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        matched_code, idx, msg["dt_iso"], msg["date_str"], msg["hour"],
                        msg["day_of_week"], msg["sender_norm"], msg["is_system"], is_media,
                        is_deleted, mtype, content, wc, cat, attach,
                    ),
                )
                new_msg_id = cur.lastrowid
                added_msgs += 1

                if msg["is_system"]:
                    ev_type = classify_system_event(content)
                    if ev_type != "other":
                        cur.execute(
                            "INSERT INTO events (group_code, dt_iso, event_type, actor, target, description, message_id) VALUES (?, ?, ?, '', '', ?, ?)",
                            (matched_code, msg["dt_iso"], ev_type, content, new_msg_id),
                        )
                        added_events += 1
                elif mtype == "text" and is_plan_proposal(content):
                    cur.execute(
                        "INSERT INTO plans (group_code, dt_iso, date_str, category, initiator, message_id) VALUES (?, ?, ?, ?, ?, ?)",
                        (matched_code, msg["dt_iso"], msg["date_str"], cat, msg["sender_norm"], new_msg_id),
                    )
                    added_plans += 1
            max_idx_by_group[matched_code] = idx

        # Scan media files from drop
        drop_media = scan_media_directory(temp_extract, folder_to_code)
        if drop_media:
            cur.executemany(
                """
                INSERT INTO media_files (
                    group_code, source_folder, filename, ext, media_subtype,
                    file_size_bytes, raw_sha256, width, height, aspect_ratio,
                    is_animated, duration_sec
                ) VALUES (
                    :group_code, :source_folder, :filename, :ext, :media_subtype,
                    :file_size_bytes, :raw_sha256, :width, :height, :aspect_ratio,
                    :is_animated, :duration_sec
                )
                """,
                drop_media,
            )
        staging_conn.commit()
        print(f"[INGEST] Friday drop merged: +{added_msgs:,} msgs, +{added_events:,} events, +{added_plans:,} plans, +{len(drop_media):,} media files.")
        return folder_to_code
    finally:
        shutil.rmtree(temp_extract, ignore_errors=True)


def run_ingestion(
    baseline_db: Path | None = DEFAULT_BASELINE_DB,
    raw_dir: Path | None = DEFAULT_RAW_DIR,
    drop_dir: Path | None = None,
) -> Path:
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    conn = init_staging_db(STAGING_DB_PATH)
    folder_to_code: dict[str, str] = {}

    if baseline_db and baseline_db.exists():
        folder_to_code = load_from_baseline_db(conn, baseline_db)

    if raw_dir and raw_dir.exists():
        print(f"[INGEST] Scanning media metadata in {raw_dir} ...")
        media_records = scan_media_directory(raw_dir, folder_to_code)
        conn.executemany(
            """
            INSERT INTO media_files (
                group_code, source_folder, filename, ext, media_subtype,
                file_size_bytes, raw_sha256, width, height, aspect_ratio,
                is_animated, duration_sec
            ) VALUES (
                :group_code, :source_folder, :filename, :ext, :media_subtype,
                :file_size_bytes, :raw_sha256, :width, :height, :aspect_ratio,
                :is_animated, :duration_sec
            )
            """,
            media_records,
        )
        conn.commit()
        print(f"[INGEST] Media scanned: {len(media_records):,} files indexed (metadata only).")

    if drop_dir and drop_dir.exists():
        print(f"[INGEST] Merging fresh WhatsApp drop from {drop_dir} ...")
        ingest_fresh_drop(conn, drop_dir, folder_to_code)

    conn.close()
    print(f"[INGEST] Private staging DB ready at: {STAGING_DB_PATH}")
    return STAGING_DB_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest WhatsApp baseline and/or fresh Friday drop into private staging DB.")
    parser.add_argument("--baseline-db", type=Path, default=DEFAULT_BASELINE_DB)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--drop-dir", type=Path, default=None, help="Path to new Friday export folder/zips")
    args = parser.parse_args()
    run_ingestion(baseline_db=args.baseline_db, raw_dir=args.raw_dir, drop_dir=args.drop_dir)
