"""
Ferramentas que a IA pode usar.

O modelo nunca executa comandos livres. Ele só pode escrever uma chamada no formato
`nome: argumento` para uma das ferramentas registradas aqui, e o código decide o que
fazer. Abrir app só abre o que está no menu Iniciar (ou numa lista conhecida); não
existe ferramenta de "rodar comando no terminal", de propósito.

Toda ferramenta devolve um texto começando com "ok: ..." ou "erro: ...". É esse texto
que volta para o modelo (<|result|>) e ele responde a partir dele.

    python tools.py                 # lista os apps encontrados no seu PC
    python tools.py abrir_app spotify --dry-run
"""

import ast
import datetime
import difflib
import json
import math
import operator
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.parse
import urllib.request
import uuid
import webbrowser
from dataclasses import dataclass
from pathlib import Path

WINDOWS = platform.system() == "Windows"
MEMORY_DIR = Path(__file__).resolve().parent / "memoria"
DRY_RUN = False  # True = não abre nada de verdade (testes)

# Quem quiser ser avisado quando um lembrete disparar (o chat do terminal, por exemplo).
REMINDER_LISTENERS = []
ACTIVE_REMINDERS = {}  # id -> (timer, texto, quando dispara): a interface lista e cancela
LAST_REMINDER_ID = None  # o último lembrete criado (para o "cancelar" no cartão)
_REPLACED_FACTS = {}  # fato novo -> fatos antigos que ele substituiu (para o "desfazer" devolver)


@dataclass
class Tool:
    name: str
    func: object
    confirm: bool  # pede confirmação antes de executar?
    desc: str


TOOLS = {}


def tool(name, desc, confirm=False):
    def deco(f):
        TOOLS[name] = Tool(name, f, confirm, desc)
        return f
    return deco


def run(name, arg=""):
    t = TOOLS.get(name)
    if t is None:
        return f"erro: ferramenta {name} não existe"
    try:
        return t.func((arg or "").strip())
    except Exception as e:  # a IA nunca deve derrubar o programa
        return f"erro: {e}"


def parse_call(text):
    """'abrir_app: spotify' -> ('abrir_app', 'spotify'). None se não for uma chamada válida."""
    name, _, arg = text.partition(":")
    name = name.strip()
    if name not in TOOLS:
        return None
    return name, arg.strip()


# ----------------------------------------------------------------- utilidades
def strip_accents(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def norm(s):
    """Minúsculas, sem acento, sem pontuação estranha, espaços simples."""
    s = re.sub(r"[^\w\s.+#-]", " ", strip_accents(s))
    return re.sub(r"\s+", " ", s).strip()


def _mem_path(name):
    MEMORY_DIR.mkdir(exist_ok=True)
    return MEMORY_DIR / name


def _load_json(name, default):
    try:
        return json.loads(_mem_path(name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _save_json(name, data):
    _mem_path(name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _open(target):
    """Abre um arquivo, pasta, URL ou URI (ms-settings:) com o programa padrão."""
    if DRY_RUN:
        return
    if WINDOWS:
        os.startfile(target)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", target])
    else:
        subprocess.Popen(["xdg-open", target])


def _open_url(url):
    if not DRY_RUN:
        webbrowser.open(url)


# ======================================================================= APPS
# Nomes do dia a dia -> nomes como aparecem no menu Iniciar (em pt e em en).
APP_ALIASES = {
    "bloco de notas": ["Bloco de Notas", "Notepad"], "notepad": ["Bloco de Notas", "Notepad"],
    "calculadora": ["Calculadora", "Calculator"], "calc": ["Calculadora", "Calculator"],
    "explorador de arquivos": ["Explorador de Arquivos", "File Explorer"],
    "explorer": ["Explorador de Arquivos", "File Explorer"], "meus arquivos": ["Explorador de Arquivos", "File Explorer"],
    "arquivos": ["Explorador de Arquivos", "File Explorer"],
    "configuracoes": ["Configurações", "Settings"], "config": ["Configurações", "Settings"],
    "ajustes": ["Configurações", "Settings"], "settings": ["Configurações", "Settings"],
    "navegador": ["Google Chrome", "Microsoft Edge", "Firefox", "Brave", "Opera GX Browser", "Opera Browser"],
    "chrome": ["Google Chrome"], "google chrome": ["Google Chrome"], "edge": ["Microsoft Edge"],
    "firefox": ["Firefox", "Mozilla Firefox"], "brave": ["Brave"], "opera": ["Opera Browser", "Opera GX Browser"],
    "opera gx": ["Opera GX Browser"],
    "vscode": ["Visual Studio Code"], "vs code": ["Visual Studio Code"], "code": ["Visual Studio Code"],
    "visual studio code": ["Visual Studio Code"],
    "word": ["Word"], "excel": ["Excel"], "powerpoint": ["PowerPoint"], "outlook": ["Outlook", "Outlook (new)"],
    "onenote": ["OneNote"], "teams": ["Microsoft Teams", "Teams"],
    "terminal": ["Terminal", "Windows Terminal", "Prompt de Comando", "Command Prompt"],
    "cmd": ["Prompt de Comando", "Command Prompt"], "prompt de comando": ["Prompt de Comando", "Command Prompt"],
    "powershell": ["Windows PowerShell", "PowerShell"],
    "paint": ["Paint"], "gerenciador de tarefas": ["Gerenciador de Tarefas", "Task Manager"],
    "zap": ["WhatsApp"], "whats": ["WhatsApp"], "whatsapp": ["WhatsApp"],
    "telegram": ["Telegram Desktop", "Telegram"], "obs": ["OBS Studio"], "camera": ["Câmera", "Camera"],
    "fotos": ["Fotos", "Photos"], "relogio": ["Relógio", "Clock", "Alarmes e Relógio"],
    "alarme": ["Relógio", "Clock", "Alarmes e Relógio"], "loja": ["Microsoft Store"],
    "microsoft store": ["Microsoft Store"], "ferramenta de captura": ["Ferramenta de Captura", "Snipping Tool"],
    "painel de controle": ["Painel de Controle", "Control Panel"], "xbox": ["Xbox"],
    "lol": ["League of Legends"], "cs": ["Counter-Strike 2"], "cs2": ["Counter-Strike 2"],
    "epic": ["Epic Games Launcher"], "epic games": ["Epic Games Launcher"],
    "battle.net": ["Battle.net"], "davinci": ["DaVinci Resolve"],
}
# Se o app não aparecer no menu Iniciar, esses comandos do Windows sempre existem.
APP_FALLBACK = {
    "bloco de notas": "notepad.exe", "notepad": "notepad.exe", "calculadora": "calc.exe", "calc": "calc.exe",
    "paint": "mspaint.exe", "explorador de arquivos": "explorer.exe", "explorer": "explorer.exe",
    "gerenciador de tarefas": "taskmgr.exe", "cmd": "cmd.exe", "prompt de comando": "cmd.exe",
    "terminal": "wt.exe", "powershell": "powershell.exe", "configuracoes": "ms-settings:",
    "config": "ms-settings:", "ajustes": "ms-settings:", "painel de controle": "control.exe",
}
_JUNK = ("uninstall", "desinstalar", "readme", "leia-me", "leiame", "manual", "help", "ajuda",
         "website", "documentation", "release notes", "license", "licenca", "changelog")
_FILLER = re.compile(r"^(o|a|os|as|um|uma|meu|minha|app|aplicativo|programa|jogo|do|da|de|no|na)\s+")


class AppIndex:
    """Lista de apps instalados (via Get-StartApps), com busca aproximada."""

    def __init__(self):
        self.apps = None

    def load(self, refresh=False):
        cache = _mem_path("apps.json")
        if not refresh and cache.exists() and time.time() - cache.stat().st_mtime < 86400:
            self.apps = _load_json("apps.json", [])
            if self.apps:
                return self.apps
        self.apps = []
        if WINDOWS:
            cmd = ("[Console]::OutputEncoding=[Text.Encoding]::UTF8; "
                   "Get-StartApps | Select-Object Name,AppID | ConvertTo-Json -Compress")
            try:
                out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True,
                                     timeout=60, encoding="utf-8", errors="replace").stdout
                data = json.loads(out) if out.strip() else []
                if isinstance(data, dict):
                    data = [data]
                self.apps = [{"name": d["Name"], "id": d["AppID"]} for d in data if d.get("Name")]
            except (OSError, ValueError, subprocess.SubprocessError):
                self.apps = []
        _save_json("apps.json", self.apps)
        return self.apps

    @staticmethod
    def _score(q, name):
        n = norm(name)
        if q == n:
            return 1.0
        s = difflib.SequenceMatcher(None, q, n).ratio()
        palavras = n.split()
        # Palavra solta só vale se for o começo do nome, se o nome for curto ou se cobrir boa parte dele:
        # "trabalho" não abre "Conexão de Área de Trabalho Remota", "ficar" não abre "Verificar Atualizações".
        forte = n.startswith(q + " ") or len(palavras) <= 2 or len(q) >= 0.5 * len(n)
        if (q in palavras or n.startswith(q + " ")) and forte:
            s = max(s, 0.9)
        elif len(q) >= 4 and re.search(r"(^| )" + re.escape(q), n) and len(q) >= 0.5 * len(n):
            s = max(s, 0.8)
        elif len(n) >= 4 and re.search(r"(^| )" + re.escape(n) + r"( |$)", q):
            s = max(s, 0.78)
        if len(q) <= 3 and not (q in palavras and len(palavras) <= 2):  # sigla curta ("la" não abre "Lia")
            s = min(s, 0.5)
        if any(j in n for j in _JUNK):
            s *= 0.5
        return s

    def find(self, query):
        if self.apps is None:
            self.load()
        q = norm(query)
        while _FILLER.match(q):
            q = _FILLER.sub("", q, count=1)
        by_name = {norm(a["name"]): a for a in self.apps}
        for alias in APP_ALIASES.get(q, []):
            if norm(alias) in by_name:
                return by_name[norm(alias)]
        best, best_score = None, 0.0
        for a in self.apps:
            s = self._score(q, a["name"])
            if s > best_score or (s == best_score and best and len(a["name"]) < len(best["name"])):
                best, best_score = a, s
        return best if best_score >= 0.75 else None


APPS = AppIndex()


def _launch_app(entry):
    if DRY_RUN:
        return
    subprocess.Popen(["explorer.exe", "shell:AppsFolder\\" + entry["id"]])


def _app_paths(q):
    """Procura executáveis registrados no Windows (App Paths) ou no PATH."""
    if not re.fullmatch(r"[a-z0-9_.+-]+", q):
        return None
    if WINDOWS:
        import winreg
        for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            try:
                key = rf"Software\Microsoft\Windows\CurrentVersion\App Paths\{q}.exe"
                with winreg.OpenKey(root, key) as k:
                    path = winreg.QueryValue(k, None)
                    if path:
                        return path.strip('"')
            except OSError:
                pass
    return shutil.which(q)


@tool("abrir_app", "abre um programa instalado", confirm=True)
def abrir_app(arg):
    if not arg:
        return "erro: qual app?"
    q = norm(arg)
    while _FILLER.match(q):
        q = _FILLER.sub("", q, count=1)
    entry = APPS.find(q)
    if entry:
        _launch_app(entry)
        return f"ok: abri {entry['name']}"
    if q in APP_FALLBACK:
        _open(APP_FALLBACK[q])
        return f"ok: abri {arg}"
    path = _app_paths(q)
    if path:
        _open(path)
        return f"ok: abri {Path(path).stem}"
    if _site_url(q):  # "abre o youtube" pedido como app: abre o site
        return abrir_site(arg)
    return f"erro: não achei o app {arg}"


# ====================================================================== SITES
SITES = {
    "youtube": "https://www.youtube.com", "google": "https://www.google.com", "gmail": "https://mail.google.com",
    "github": "https://github.com", "netflix": "https://www.netflix.com", "wikipedia": "https://pt.wikipedia.org",
    "whatsapp web": "https://web.whatsapp.com", "whatsapp": "https://web.whatsapp.com",
    "instagram": "https://www.instagram.com", "twitter": "https://x.com", "reddit": "https://www.reddit.com",
    "linkedin": "https://www.linkedin.com", "twitch": "https://www.twitch.tv", "amazon": "https://www.amazon.com.br",
    "mercado livre": "https://www.mercadolivre.com.br", "g1": "https://g1.globo.com", "globo": "https://www.globo.com",
    "google drive": "https://drive.google.com", "drive": "https://drive.google.com",
    "google maps": "https://maps.google.com", "maps": "https://maps.google.com",
    "tradutor": "https://translate.google.com", "google tradutor": "https://translate.google.com",
    "canva": "https://www.canva.com", "facebook": "https://www.facebook.com", "tiktok": "https://www.tiktok.com",
    "pinterest": "https://www.pinterest.com", "prime video": "https://www.primevideo.com",
    "disney plus": "https://www.disneyplus.com", "disney+": "https://www.disneyplus.com", "max": "https://www.max.com",
    "stack overflow": "https://stackoverflow.com", "duolingo": "https://www.duolingo.com",
    "claude": "https://claude.ai", "google agenda": "https://calendar.google.com",
    "agenda": "https://calendar.google.com", "shopee": "https://shopee.com.br", "ifood": "https://www.ifood.com.br",
    "spotify web": "https://open.spotify.com", "nubank": "https://nubank.com.br", "uol": "https://www.uol.com.br",
}
_DOMAIN = re.compile(r"^(https?://)?([\w-]+\.)+[a-z]{2,}(/\S*)?$")


def _site_url(q):
    q = re.sub(r"^(o |a )?(site |pagina )?(do |da |de )?", "", norm(q))
    if q in SITES:
        return SITES[q]
    if _DOMAIN.match(q):
        return q if q.startswith("http") else "https://" + q
    close = difflib.get_close_matches(q, list(SITES), n=1, cutoff=0.78)
    return SITES[close[0]] if close else None


@tool("abrir_site", "abre um site no navegador", confirm=True)
def abrir_site(arg):
    url = _site_url(arg)
    if not url:
        return f"erro: não conheço o site {arg}"
    _open_url(url)
    return f"ok: abri {urllib.parse.urlparse(url).netloc.removeprefix('www.')}"


@tool("pesquisar", "pesquisa no Google", confirm=True)
def pesquisar(arg):
    if not arg:
        return "erro: pesquisar o quê?"
    _open_url("https://www.google.com/search?q=" + urllib.parse.quote_plus(arg))
    return "ok: pesquisa aberta no navegador"


@tool("youtube", "procura vídeos/músicas no YouTube", confirm=True)
def youtube(arg):
    if not arg:
        return abrir_site("youtube")
    _open_url("https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(arg))
    return "ok: busca aberta no youtube"


# ============================================================== HORA E DATA
DIAS = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]
MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro",
         "outubro", "novembro", "dezembro"]


def formatar_data(d):
    return f"{DIAS[d.weekday()]}, {d.day} de {MESES[d.month - 1]} de {d.year}"


@tool("hora", "diz a hora atual")
def hora(arg):
    return "ok: " + datetime.datetime.now().strftime("%H:%M")


@tool("data", "diz a data de hoje")
def data(arg):
    return "ok: " + formatar_data(datetime.date.today())


# ================================================================ CALCULADORA
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log10,
          "ln": math.log, "abs": abs, "round": round}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node):
    """Avalia só aritmética. Nada de nomes, atributos, imports: é seguro com texto do modelo."""
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(b) > 100 or abs(a) > 1e6):
            raise ValueError("número grande demais")
        return _OPS[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("expressão não suportada")


def _to_python(expr):
    s = strip_accents(expr.replace("×", "*").replace("÷", "/").replace("^", "**").replace("√", " raiz de "))
    s = re.sub(r"\s+", " ", s.replace("?", " ").replace("=", " ")).strip()
    s = re.sub(r"^(quanto e|quanto da|calcula|calcule|resolve)\s+", "", s)
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)  # vírgula decimal
    s = re.sub(r"(\d+(?:\.\d+)?)\s*(%|por cento)\s*(de|do|da|sobre)\s*", r"\1/100*", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*(%|por cento)", r"(\1/100)", s)
    for a, b in [("raiz quadrada de", "sqrt "), ("raiz de", "sqrt "), ("raiz", "sqrt "),
                 ("ao quadrado", "**2"), ("ao cubo", "**3"), ("elevado ao", "**"), ("elevado a", "**"),
                 ("vezes", "*"), ("multiplicado por", "*"), ("dividido por", "/"), ("divido por", "/"),
                 ("sobre", "/"), ("mais", "+"), ("menos", "-"), ("seno de", "sin "), ("cosseno de", "cos "),
                 ("log de", "log ")]:
        s = s.replace(a, b)
    s = re.sub(r"(?<=[\d)\s])x(?=[\s\d(])", "*", s)  # 3 x 4
    s = re.sub(r"\b(sqrt|sin|cos|tan|log)\s+(\d+(?:\.\d+)?)", r"\1(\2)", s)
    return s.strip()


def calcular_valor(expr):
    value = _eval(ast.parse(_to_python(expr), mode="eval"))
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("resultado inválido")
        if value.is_integer() and abs(value) < 1e15:
            value = int(value)
    if isinstance(value, int):
        return f"{value:,}".replace(",", ".") if abs(value) >= 10000 else str(value)
    return f"{value:.6g}".replace(".", ",")


@tool("calcular", "faz contas")
def calcular(arg):
    try:
        return "ok: " + calcular_valor(arg)
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError, OverflowError):
        return "erro: não consegui calcular isso"


# ================================================================== LEMBRETES
_NUM_WORDS = {"um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3, "quatro": 4, "cinco": 5, "seis": 6,
              "sete": 7, "oito": 8, "nove": 9, "dez": 10, "quinze": 15, "vinte": 20, "trinta": 30,
              "quarenta": 40, "cinquenta": 50, "sessenta": 60}


def _horario(s):
    """'às 15h', 'às 7:30', 'amanhã', 'amanhã 9h', 'hoje às 20h', 'meio-dia' -> (amanhã?, hora, minuto).
    None se não for horário marcado. 'daqui a 10 minutos' e '1h30' (sem 'às') continuam sendo duração."""
    s = strip_accents(s)
    amanha, hoje = bool(re.search(r"\bamanha\b", s)), bool(re.search(r"\bhoje\b", s))
    h = mi = None
    if re.search(r"\bmeio[ -]?dia\b", s):
        h, mi = 12, 0
    elif re.search(r"\bmeia[ -]?noite\b", s):
        h, mi = 0, 0
    else:
        m = (re.search(r"\b(\d{1,2}):(\d{2})\b", s)
             or re.search(r"\b(?:as|ate as)\s+(\d{1,2})(?:\s*(?:h|horas?)\s*(\d{2})?)?\b", s)
             or ((amanha or hoje) and re.search(r"\b(\d{1,2})\s*h(?:oras?)?\s*(\d{2})?\b", s)))
        if m:
            h, mi = int(m.group(1)), int(m.group(2) or 0)
            if h < 12 and re.search(r"\bda (tarde|noite)\b", s):  # "às 3 da tarde"
                h += 12
    if h is None:
        return (True, 9, 0) if amanha else None  # "amanhã" sem hora: às 9h
    if not (0 <= h <= 23 and 0 <= mi <= 59):
        return None
    return amanha, h, mi


def tempo_absoluto(s, agora=None):
    """Segundos até o horário marcado em s (None se s não marca horário). Já passou hoje = amanhã."""
    hr = _horario(s)
    if hr is None:
        return None
    amanha, h, mi = hr
    agora = agora or datetime.datetime.now()
    alvo = agora.replace(hour=h, minute=mi, second=0, microsecond=0)
    if amanha or alvo <= agora:
        alvo += datetime.timedelta(days=1)
    return int((alvo - agora).total_seconds())


def ancorar_lembrete(arg, fala):
    """O modelo às vezes perde o 'às 15h' ou o 'amanhã' ao copiar ('me avisa às 15h' -> 'lembrete: 15h', que
    viraria 15 HORAS). Se a fala marca horário e o argumento não, o horário da fala vale."""
    hr = _horario(fala)
    if hr is None:
        return arg
    amanha, h, mi = hr
    expr = " ".join((["amanhã"] if amanha else []) + [f"às {h}h" + (f"{mi:02d}" if mi else "")])
    tempo, sep, texto = arg.partition("|")
    if sep:
        return arg if _horario(tempo) == hr else f"{expr} | {texto.strip()}"
    try:
        parse_duracao(arg)  # o argumento todo é um tempo ("15h"): troca pelo horário
        return expr
    except ValueError:  # o argumento todo é a tarefa ("pagar a conta")
        return f"{expr} | {arg.strip()}"


def parse_duracao(s, agora=None):
    """'10 minutos', '1h30', 'meia hora', '30 seg', 'duas horas', 'às 15h', 'amanhã' -> segundos."""
    t = tempo_absoluto(s, agora)
    if t is not None:
        if t <= 0 or t > 7 * 86400:
            raise ValueError("horário fora do limite")
        return t
    if re.search(r"\bas\s+\d", strip_accents(s)):  # "às 25h": horário que não existe
        raise ValueError("horário inválido")
    s = norm(s)
    if "meia hora" in s:
        return 1800
    s = re.sub(r"\b(" + "|".join(_NUM_WORDS) + r")\b", lambda m: str(_NUM_WORDS[m.group(1)]), s)
    m = re.search(r"(\d+)\s*h(?:oras?)?\s*e?\s*(\d+)", s)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60
    total, found = 0.0, False
    for num, unit in re.findall(r"(\d+(?:[.,]\d+)?)\s*(horas?|h|minutos?|mins?|m|segundos?|seg|s)\b", s):
        n = float(num.replace(",", "."))
        total += n * (3600 if unit.startswith("h") else 1 if unit.startswith("s") else 60)
        found = True
    if not found:
        m = re.search(r"\d+(?:[.,]\d+)?", s)
        if not m:
            raise ValueError("sem número")
        total = float(m.group().replace(",", ".")) * 60  # número solto = minutos
    if total <= 0 or total > 7 * 86400:
        raise ValueError("duração fora do limite")
    return int(total)


def formatar_duracao(sec):
    h, rest = divmod(sec, 3600)
    m, s = divmod(rest, 60)
    parts = []
    if h:
        parts.append(f"{h} hora" + ("s" if h > 1 else ""))
    if m:
        parts.append(f"{m} minuto" + ("s" if m > 1 else ""))
    if s and not h:
        parts.append(f"{s} segundo" + ("s" if s > 1 else ""))
    return " e ".join(parts)


TOAST_APP_ID = "Lia.Assistente"
LOGO_PNG = Path(__file__).resolve().parent / "web" / "logo.png"
LOGO_ICO = Path(__file__).resolve().parent / "web" / "logo.ico"

# O Windows só mostra notificação de programa de desktop que tenha um atalho no Menu Iniciar
# com o mesmo AppUserModelID. Este script cria o atalho "Lia" (que abre a interface).
# Tudo entra por variável de ambiente; o C# só existe porque o PowerShell não grava o ID no .lnk.
_SHORTCUT_PS = r"""
Add-Type -TypeDefinition @'
using System; using System.Text; using System.Runtime.InteropServices;
[ComImport, Guid("00021401-0000-0000-C000-000000000046")] class CShellLink {}
[ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("000214F9-0000-0000-C000-000000000046")]
interface IShellLinkW {
  void GetPath([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder f, int c, IntPtr fd, uint fl);
  void GetIDList(out IntPtr p); void SetIDList(IntPtr p);
  void GetDescription([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder s, int c);
  void SetDescription([MarshalAs(UnmanagedType.LPWStr)] string s);
  void GetWorkingDirectory([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder s, int c);
  void SetWorkingDirectory([MarshalAs(UnmanagedType.LPWStr)] string s);
  void GetArguments([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder s, int c);
  void SetArguments([MarshalAs(UnmanagedType.LPWStr)] string s);
  void GetHotkey(out short h); void SetHotkey(short h);
  void GetShowCmd(out int i); void SetShowCmd(int i);
  void GetIconLocation([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder s, int c, out int i);
  void SetIconLocation([MarshalAs(UnmanagedType.LPWStr)] string s, int i);
  void SetRelativePath([MarshalAs(UnmanagedType.LPWStr)] string s, uint r);
  void Resolve(IntPtr h, uint f);
  void SetPath([MarshalAs(UnmanagedType.LPWStr)] string s);
}
[StructLayout(LayoutKind.Sequential, Pack = 4)] public struct PropertyKey { public Guid fmtid; public uint pid; }
[StructLayout(LayoutKind.Explicit, Size = 24)] public struct PropVariant { [FieldOffset(0)] public ushort vt; [FieldOffset(8)] public IntPtr p; }
[ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99")]
interface IPropertyStore {
  void GetCount(out uint c); void GetAt(uint i, out PropertyKey k);
  void GetValue(ref PropertyKey k, out PropVariant v); void SetValue(ref PropertyKey k, ref PropVariant v); void Commit();
}
[ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("0000010b-0000-0000-C000-000000000046")]
interface IPersistFile {
  void GetClassID(out Guid g); [PreserveSig] int IsDirty();
  void Load([MarshalAs(UnmanagedType.LPWStr)] string f, uint m);
  void Save([MarshalAs(UnmanagedType.LPWStr)] string f, bool r);
}
public static class LiaShortcut {
  public static void Create(string lnk, string target, string args, string dir, string icon, string aumid) {
    var link = (IShellLinkW)new CShellLink();
    link.SetPath(target); link.SetArguments(args); link.SetWorkingDirectory(dir);
    link.SetIconLocation(icon, 0); link.SetDescription("Lia");
    var store = (IPropertyStore)link;
    var key = new PropertyKey { fmtid = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"), pid = 5 };  // System.AppUserModel.ID
    var pv = new PropVariant { vt = 31, p = Marshal.StringToCoTaskMemUni(aumid) };  // VT_LPWSTR
    try { store.SetValue(ref key, ref pv); store.Commit(); ((IPersistFile)link).Save(lnk, true); }
    finally { Marshal.FreeCoTaskMem(pv.p); }
  }
}
'@
[LiaShortcut]::Create($env:LIA_LNK, $env:LIA_TARGET, $env:LIA_ARGS, $env:LIA_DIR, $env:LIA_ICON, $env:LIA_TOAST_APP)
"""


def _ensure_shortcut():
    """Cria (uma vez) o atalho "Lia" no Menu Iniciar do usuário, necessário para as notificações."""
    lnk = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Lia.lnk"
    if lnk.exists():
        return
    root = Path(__file__).resolve().parent
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    env = dict(os.environ, LIA_LNK=str(lnk), LIA_TARGET=str(pythonw if pythonw.exists() else sys.executable),
               LIA_ARGS="web.py", LIA_DIR=str(root), LIA_ICON=str(LOGO_ICO), LIA_TOAST_APP=TOAST_APP_ID)
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _SHORTCUT_PS], env=env,
                   capture_output=True, timeout=60, creationflags=subprocess.CREATE_NO_WINDOW)

# Mostra a notificação pela API do Windows (WinRT) via PowerShell, sem instalar nada.
# O XML vai por variável de ambiente: o texto do lembrete nunca entra no comando.
_TOAST_PS = r"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType = WindowsRuntime] | Out-Null
$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml($env:LIA_TOAST_XML)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($env:LIA_TOAST_APP).Show(
    [Windows.UI.Notifications.ToastNotification]::new($xml))
"""


def _toast(titulo, texto):
    """
    Notificação nativa do Windows com nome "Lia" e logo. scenario="alarm" deixa ela na tela
    até você fechar e passa pelo "Não incomodar" (que só deixa passar alarmes); "Adiar" e "Ok"
    são tratados pelo próprio Windows.
    """
    import winreg
    from xml.sax.saxutils import escape, quoteattr

    # Identidade da notificação: atalho no Menu Iniciar + nome/ícone no registro do usuário (sem admin)
    _ensure_shortcut()
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"Software\Classes\AppUserModelId\{TOAST_APP_ID}") as k:
        winreg.SetValueEx(k, "DisplayName", 0, winreg.REG_SZ, "Lia")
        if LOGO_PNG.exists():
            winreg.SetValueEx(k, "IconUri", 0, winreg.REG_SZ, str(LOGO_PNG))
    logo = f"<image placement=\"appLogoOverride\" src={quoteattr(LOGO_PNG.as_uri())}/>" if LOGO_PNG.exists() else ""
    xml = ("<toast scenario=\"alarm\">"
           "<visual><binding template=\"ToastGeneric\">"
           f"<text>{escape(titulo)}</text><text>{escape(texto)}</text>{logo}"
           "</binding></visual>"
           "<actions><input id=\"adiar\" type=\"selection\" defaultInput=\"5\">"
           "<selection id=\"5\" content=\"5 minutos\"/><selection id=\"10\" content=\"10 minutos\"/>"
           "<selection id=\"30\" content=\"30 minutos\"/><selection id=\"60\" content=\"1 hora\"/></input>"
           "<action activationType=\"system\" arguments=\"snooze\" hint-inputId=\"adiar\" content=\"Adiar\"/>"
           "<action activationType=\"system\" arguments=\"dismiss\" content=\"Ok\"/></actions>"
           "<audio src=\"ms-winsoundevent:Notification.Reminder\" loop=\"false\"/></toast>")
    env = dict(os.environ, LIA_TOAST_XML=xml, LIA_TOAST_APP=TOAST_APP_ID)
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _TOAST_PS], env=env,
                       capture_output=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
    return r.returncode == 0 and not r.stderr.strip()


def _fire_reminder(texto, rid=None):
    ACTIVE_REMINDERS.pop(rid, None)
    for cb in list(REMINDER_LISTENERS):
        try:
            cb(texto)
        except Exception:
            pass
    if WINDOWS and not DRY_RUN:
        try:
            if _toast("Lembrete", texto):
                return
        except Exception:
            pass
        # Plano B: a caixinha simples de sempre
        import ctypes
        import winsound
        winsound.MessageBeep(winsound.MB_ICONASTERISK)
        # MB_ICONINFORMATION | MB_SYSTEMMODAL | MB_TOPMOST: aparece por cima de tudo
        ctypes.windll.user32.MessageBoxW(None, texto, "Lembrete", 0x40 | 0x1000 | 0x40000)


@tool("lembrete", "cria um lembrete/timer: 'tempo | texto'")
def lembrete(arg):
    tempo, _, texto = arg.partition("|")
    texto = texto.strip() or "o timer acabou"
    try:
        sec = parse_duracao(tempo)
    except ValueError:
        return "erro: não entendi o tempo"
    rid = uuid.uuid4().hex[:8]
    t = threading.Timer(sec, _fire_reminder, args=(texto, rid))
    t.daemon = True  # lembretes vivem enquanto o programa estiver aberto
    if not DRY_RUN:
        global LAST_REMINDER_ID
        ACTIVE_REMINDERS[rid] = (t, texto, time.time() + sec)
        LAST_REMINDER_ID = rid
        t.start()
    return f"ok: em {formatar_duracao(sec)}: {texto}"


def lembretes_ativos():
    """Lembretes que ainda vão disparar (para a interface)."""
    agora = time.time()
    return sorted(({"id": k, "texto": v[1], "faltam": max(0, int(v[2] - agora))} for k, v in ACTIVE_REMINDERS.items()),
                  key=lambda r: r["faltam"])


def cancelar_lembrete(rid):
    item = ACTIVE_REMINDERS.pop(rid, None)
    if item:
        item[0].cancel()
    return item is not None


# ===================================================================== NOTAS
@tool("anotar", "salva uma nota")
def anotar(arg):
    if not arg:
        return "erro: anotar o quê?"
    with open(_mem_path("notas.md"), "a", encoding="utf-8") as f:
        f.write(f"- [{datetime.datetime.now():%Y-%m-%d %H:%M}] {arg}\n")
    return "ok: anotado"


@tool("ver_notas", "lê as últimas notas")
def ver_notas(arg):
    try:
        lines = _mem_path("notas.md").read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    notes = [re.sub(r"^- \[[^\]]*\]\s*", "", l) for l in lines if l.startswith("- ")]
    if not notes:
        return "ok: nenhuma nota ainda"
    return "ok: " + "; ".join(notes[-5:])


def memoria():
    """Fatos guardados e notas, para a interface mostrar (e apagar, se a pessoa quiser)."""
    try:
        linhas = _mem_path("notas.md").read_text(encoding="utf-8").splitlines()
    except OSError:
        linhas = []
    notas = [re.sub(r"^- \[[^\]]*\]\s*", "", l) for l in linhas if l.startswith("- ")]
    return {"fatos": _load_json("fatos.json", []), "notas": notas}


def apagar_memoria(tipo, i=None):
    """Apaga o item i (ou tudo, com i=None) dos fatos ou das notas."""
    if tipo == "fatos":
        fatos = _load_json("fatos.json", [])
        _save_json("fatos.json", [] if i is None else [f for k, f in enumerate(fatos) if k != i])
    elif tipo == "notas":
        path = _mem_path("notas.md")
        try:
            linhas = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        idx = [k for k, l in enumerate(linhas) if l.startswith("- ")]
        tirar = set(idx) if i is None else ({idx[i]} if 0 <= i < len(idx) else set())
        path.write_text("".join(l + "\n" for k, l in enumerate(linhas) if k not in tirar), encoding="utf-8")


def desfazer(tipo, ref):
    """Desfaz uma ação pelo cartão da interface. Devolve uma frase curta para mostrar."""
    if tipo == "lembrete":
        return "lembrete cancelado" if cancelar_lembrete(ref) else "esse lembrete já tinha disparado"
    if tipo == "fato":
        fatos = _load_json("fatos.json", [])
        if ref not in fatos:
            return "isso já não estava guardado"
        fatos = [f for f in fatos if f != ref] + _REPLACED_FACTS.pop(ref, [])  # devolve o que tinha sido trocado
        _save_json("fatos.json", fatos)
        return "esqueci isso"
    if tipo == "nota":
        path = _mem_path("notas.md")
        try:
            linhas = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return "nota não encontrada"
        for k in range(len(linhas) - 1, -1, -1):  # a mais recente com esse texto
            if linhas[k].startswith("- ") and re.sub(r"^- \[[^\]]*\]\s*", "", linhas[k]) == ref:
                del linhas[k]
                path.write_text("".join(l + "\n" for l in linhas), encoding="utf-8")
                return "nota apagada"
        return "nota não encontrada"
    return "não sei desfazer isso"


# ============================================================ MEMÓRIA LONGA
_STOP = set("o a os as um uma de do da dos das e eu meu minha me que qual quais é eh sou em no na pra para com "
            "voce vc sobre mim lembra sabe".split())


@tool("lembrar", "guarda um fato sobre você")
def lembrar(arg):
    if not arg:
        return "erro: lembrar o quê?"
    fatos = _load_json("fatos.json", [])
    # Um fato novo sobre o mesmo assunto ("meu nome é ...") substitui o antigo.
    chave = " ".join(norm(arg).split()[:3])
    _REPLACED_FACTS[arg] = [f for f in fatos if " ".join(norm(f).split()[:3]) == chave and f != arg]
    fatos = [f for f in fatos if " ".join(norm(f).split()[:3]) != chave]
    fatos.append(arg)
    _save_json("fatos.json", fatos)
    m = re.search(r"moro em ([\w\s'-]+)", arg, re.IGNORECASE)
    if m:
        cfg = _load_json("config.json", {})
        cfg["cidade"] = m.group(1).strip()
        _save_json("config.json", cfg)
    return "ok: guardei"


@tool("recordar", "procura algo que você me contou")
def recordar(arg):
    fatos = _load_json("fatos.json", [])
    if not fatos:
        return "erro: não tenho nada sobre isso"
    words = {w for w in norm(arg).split() if w not in _STOP}
    if not words:
        return "ok: " + "; ".join(fatos[-5:])

    def score(f):
        fw = set(norm(f).split())
        return sum(1 for w in words if w in fw or any(x.startswith(w[:5]) for x in fw if len(w) >= 5))
    ranked = sorted(fatos, key=score, reverse=True)
    hits = [f for f in ranked if score(f) > 0][:2]
    return "ok: " + "; ".join(hits) if hits else "erro: não tenho nada sobre isso"


# ============================================================ VOLUME E MÍDIA
def _press(vk, times=1):
    if DRY_RUN or not WINDOWS:
        return
    import ctypes
    for _ in range(times):
        ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
        ctypes.windll.user32.keybd_event(vk, 0, 2, 0)  # KEYEVENTF_KEYUP


@tool("volume", "mais / menos / mudo")
def volume(arg):
    a = norm(arg)
    if a in ("mais", "aumentar", "sobe", "alto", "+"):
        _press(0xAF, 5)  # VK_VOLUME_UP (cada toque = 2%)
        return "ok: volume aumentado"
    if a in ("menos", "diminuir", "abaixa", "baixo", "-"):
        _press(0xAE, 5)  # VK_VOLUME_DOWN
        return "ok: volume diminuído"
    if a in ("mudo", "mute", "silencio", "desmutar"):
        _press(0xAD)  # VK_VOLUME_MUTE
        return "ok: mudo alternado"
    return "erro: use mais, menos ou mudo"


@tool("midia", "pausar / proxima / anterior")
def midia(arg):
    a = norm(arg)
    if a in ("pausar", "tocar", "play", "pause", "continuar"):
        _press(0xB3)  # VK_MEDIA_PLAY_PAUSE
        return "ok: play/pause"
    if a in ("proxima", "pular", "next"):
        _press(0xB0)
        return "ok: próxima faixa"
    if a in ("anterior", "voltar", "previous"):
        _press(0xB1)
        return "ok: faixa anterior"
    return "erro: use pausar, proxima ou anterior"


# ===================================================================== PASTAS
_KNOWN_FOLDERS = {
    "downloads": ("Downloads", "{374DE290-123F-4565-9164-39C4925E467B}"),
    "documentos": ("Documentos", "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}"),
    "area de trabalho": ("Área de Trabalho", "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}"),
    "imagens": ("Imagens", "{33E28130-4E1E-4676-835A-98395C3BC3BB}"),
    "musicas": ("Músicas", "{4BD8D571-6D19-48D3-BE97-422220080E43}"),
    "videos": ("Vídeos", "{18989B1D-99B5-455B-841C-AB7C74E4DDFC}"),
}
_FOLDER_ALIASES = {"desktop": "area de trabalho", "documents": "documentos", "fotos": "imagens",
                   "pictures": "imagens", "music": "musicas", "musica": "musicas", "video": "videos",
                   "download": "downloads", "baixados": "downloads", "documento": "documentos"}


def _known_folder(guid):
    import ctypes
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                    ("Data4", ctypes.c_ubyte * 8)]
    u = uuid.UUID(guid)
    g = GUID(u.fields[0], u.fields[1], u.fields[2], (ctypes.c_ubyte * 8)(*u.bytes[8:]))
    ptr = ctypes.c_wchar_p()
    if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(g), 0, None, ctypes.byref(ptr)) != 0:
        raise OSError("pasta não encontrada")
    path = ptr.value
    ctypes.windll.ole32.CoTaskMemFree(ptr)
    return path


@tool("abrir_pasta", "abre downloads, documentos, imagens...", confirm=True)
def abrir_pasta(arg):
    q = re.sub(r"^(a |as |o |os |minha |minhas |meus |meu )?(pasta )?(de |do |da |dos |das )?", "", norm(arg))
    q = _FOLDER_ALIASES.get(q, q)
    if q in ("usuario", "home", "pessoal", "inicio"):
        _open(str(Path.home()))
        return "ok: abri a pasta do usuário"
    if q not in _KNOWN_FOLDERS:
        return f"erro: não conheço a pasta {arg}"
    nome, guid = _KNOWN_FOLDERS[q]
    path = _known_folder(guid) if WINDOWS else str(Path.home() / nome)
    _open(path)
    return f"ok: abri a pasta {nome}"


# ====================================================================== CLIMA
_WMO = {0: "céu limpo", 1: "quase limpo", 2: "parcialmente nublado", 3: "nublado", 45: "neblina", 48: "neblina",
        51: "garoa fraca", 53: "garoa", 55: "garoa forte", 61: "chuva fraca", 63: "chuva", 65: "chuva forte",
        66: "chuva congelante", 67: "chuva congelante", 71: "neve fraca", 73: "neve", 75: "neve forte",
        80: "pancadas de chuva", 81: "pancadas de chuva", 82: "pancadas fortes", 95: "tempestade",
        96: "tempestade com granizo", 99: "tempestade com granizo"}


def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "mini-gpt-assistente"})
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.loads(r.read().decode("utf-8"))


@tool("clima", "previsão do tempo (Open-Meteo)")
def clima(arg):
    cidade = re.sub(r"^(em |no |na |de |pra |para )", "", arg.strip(" ?"), flags=re.IGNORECASE)
    if not cidade:
        cidade = _load_json("config.json", {}).get("cidade", "")
    if not cidade:
        return "erro: não sei sua cidade"
    try:
        geo = _get_json("https://geocoding-api.open-meteo.com/v1/search?count=1&language=pt&name="
                        + urllib.parse.quote(cidade))
        if not geo.get("results"):
            return f"erro: não achei a cidade {cidade}"
        g = geo["results"][0]
        w = _get_json(f"https://api.open-meteo.com/v1/forecast?latitude={g['latitude']}&longitude={g['longitude']}"
                      "&current=temperature_2m,weather_code&daily=temperature_2m_max,temperature_2m_min,"
                      "precipitation_probability_max&timezone=auto&forecast_days=1")
    except OSError:
        return "erro: sem internet"
    cur, day = w["current"], w["daily"]
    desc = _WMO.get(cur["weather_code"], "tempo instável")
    chuva = day.get("precipitation_probability_max", [None])[0]
    texto = (f"{g['name']}: {cur['temperature_2m']:.0f}°C, {desc}, máx {day['temperature_2m_max'][0]:.0f}°C, "
             f"mín {day['temperature_2m_min'][0]:.0f}°C")
    if chuva is not None:
        texto += f", chuva {chuva}%"
    return "ok: " + texto


# =================================================================== WIKIPÉDIA
# A Lia não sabe fatos de cabeça (e inventaria): ela lê o começo do artigo e repete.
WIKI_API = "https://pt.wikipedia.org/w/api.php"
WIKI_UA = "Lia/1.0 (assistente pessoal em portugues; https://lia-ai.pages.dev)"  # a Wikimedia pede identificação
LAST_WIKI = None  # {"titulo", "url"} da última consulta: o cartão mostra o link do artigo
_ABREV = {"dr", "dra", "sr", "sra", "srta", "st", "jr", "prof", "profa", "a.c", "d.c", "séc", "vol", "cap", "n.º", "nº"}
_ARTIGO = re.compile(r"^(o|a|os|as|um|uma)\s+", re.IGNORECASE)


def resumo_wiki(texto, limite=260):
    """Primeiro parágrafo do artigo -> uma ou duas frases limpas, curtas o bastante para a Lia repetir.
    O mesmo que resumoWiki() do site (webapp/ferramentas.js): mexeu num, mexe no outro."""
    t = texto.strip().split("\n")[0]
    t = re.sub(r"\s*\[[^\[\]]*\]", "", t)  # pronúncia: [kɐ̃ˈbɛʁɐ]
    antes = None
    while antes != t:  # parênteses sem número (variante, nome em inglês) saem; com data ficam
        antes = t
        t = re.sub(r"\s*\(([^()]*)\)", lambda m: m.group(0) if re.search(r"[0-9]", m.group(1)) else "", t)
    t = re.sub(r"\s+([,.;:])", r"\1", re.sub(r"\s+", " ", t)).strip()
    frases, ini = [], 0
    for m in re.finditer(r"[.!?](?=\s+[A-ZÀ-Ý0-9\"“(])", t):
        w = t[ini:m.start()].rsplit(" ", 1)[-1]
        if w.lower() in _ABREV or (len(w) == 1 and w.isalpha()):  # "Dr.", "a.C.", iniciais ("J. R.")
            continue
        frases.append(t[ini:m.end()].strip())
        ini = m.end()
    if t[ini:].strip():
        frases.append(t[ini:].strip())
    out = frases[0] if frases else ""
    if len(frases) > 1 and len(out) < 80 and len(out) + 1 + len(frases[1]) <= limite:
        out += " " + frases[1]  # frase curta demais sozinha ("Camberra é a capital da Austrália."): junta a próxima
    if len(out) > limite:
        corte = out[:limite]
        k = corte.rfind(", ")
        corte = corte[:k] if k > limite * 0.6 else corte[:corte.rfind(" ")]
        out = corte.rstrip(" ,;:") + "…"
    return out


def assunto_wiki(arg):
    q = arg.strip(" ?!.\t")
    return _ARTIGO.sub("", q).strip()


@tool("consultar", "lê o resumo de um assunto na Wikipédia")
def consultar(arg):
    global LAST_WIKI
    LAST_WIKI = None
    q = assunto_wiki(arg)
    if not q:
        return "erro: consultar o quê?"
    p = {"action": "query", "format": "json", "formatversion": 2, "generator": "search", "gsrsearch": q,
         "gsrlimit": 3, "prop": "extracts|pageprops", "exintro": 1, "explaintext": 1, "exlimit": 3,
         "ppprop": "disambiguation", "redirects": 1}
    try:
        req = urllib.request.Request(WIKI_API + "?" + urllib.parse.urlencode(p), headers={"User-Agent": WIKI_UA})
        with urllib.request.urlopen(req, timeout=8) as r:
            d = json.loads(r.read().decode("utf-8"))
    except OSError:
        return "erro: sem internet"
    pages = [pg for pg in sorted(d.get("query", {}).get("pages", []), key=lambda pg: pg.get("index", 99))
             if "pageprops" not in pg and (pg.get("extract") or "").strip()]  # pula desambiguação
    if not pages:
        return f"erro: não achei {q} na Wikipédia"
    alvo = next((pg for pg in pages if norm(pg["title"]) == norm(q)), pages[0])  # título exato ganha
    resumo = resumo_wiki(alvo["extract"])
    if not resumo:
        return f"erro: não achei {q} na Wikipédia"
    LAST_WIKI = {"titulo": alvo["title"],
                 "url": "https://pt.wikipedia.org/wiki/" + urllib.parse.quote(alvo["title"].replace(" ", "_"))}
    return "ok: " + resumo


# ===================================================================== SISTEMA
@tool("bloquear", "bloqueia a tela do PC", confirm=True)
def bloquear(arg):
    if WINDOWS and not DRY_RUN:
        import ctypes
        ctypes.windll.user32.LockWorkStation()
    return "ok: tela bloqueada"


def describe():
    return "\n".join(f"  {t.name:<12} {t.desc}" + ("  (pede confirmação)" if t.confirm else "")
                     for t in TOOLS.values())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    DRY_RUN = "--dry-run" in sys.argv
    if args:
        print(run(args[0], " ".join(args[1:])))
    else:
        apps = APPS.load(refresh=True)
        print(f"{len(apps)} apps no menu Iniciar:")
        for a in sorted(apps, key=lambda a: a["name"].lower()):
            print("  " + a["name"])
        print("\nFerramentas:\n" + describe())
