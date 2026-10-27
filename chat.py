"""
Chat no terminal.

    python chat.py
    python chat.py --auto          # executa ferramentas sem pedir confirmação
    python chat.py --voz           # lê as respostas em voz alta (vozes do Windows)
    python chat.py --simular       # ferramentas não abrem nada de verdade
    python chat.py --sem-conselho  # responde direto, sem o debate entre papéis (conselho.py)
    python chat.py --debate        # mostra o debate do conselho antes de cada resposta

Comandos dentro do chat: /ajuda
"""

import argparse
import os
import platform
import subprocess
import sys

import tools
from agent import Assistant

# Cores ANSI (o "os.system('')" liga o suporte a cores no console do Windows).
if platform.system() == "Windows":
    os.system("")
DIM, BOLD, CYAN, GREEN, YELLOW, RED, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"

HELP = f"""{DIM}Comandos:
  /reset         começa uma conversa nova
  /auto          liga/desliga executar ferramentas sem perguntar
  /temp 0.7      criatividade (0 = sempre a mesma resposta, 1 = solta)
  /conselho      liga/desliga o debate entre papéis antes de responder
  /debate        liga/desliga mostrar o debate
  /voz           liga/desliga leitura em voz alta
  /ferramentas   lista o que eu sei fazer
  /apps          lista os apps que encontrei no seu PC
  /memoria       mostra o que guardei sobre você
  /sair          tchau{RESET}"""


def speak(text):
    """Fala usando o sintetizador do Windows, sem travar o chat."""
    if platform.system() != "Windows" or not text:
        return
    env = dict(os.environ, FALA_TEXTO=text)  # texto via variável: nada de injeção no comando
    cmd = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
           "$v = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'pt*' } | Select-Object -First 1; "
           "if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }; $s.Speak($env:FALA_TEXTO)")
    subprocess.Popen(["powershell", "-NoProfile", "-Command", cmd], env=env,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/chat.pt")
    ap.add_argument("--auto", action="store_true", help="não pede confirmação antes de abrir coisas")
    ap.add_argument("--voz", action="store_true")
    ap.add_argument("--simular", action="store_true", help="ferramentas em modo simulado")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--sem-conselho", action="store_true", help="responde direto, sem o debate entre papéis")
    ap.add_argument("--debate", action="store_true", help="mostra o debate do conselho")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        sys.stdin.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    tools.DRY_RUN = args.simular

    print(f"{DIM}Carregando...{RESET}", end="\r")
    bot = Assistant(args.ckpt, temperature=args.temperature, conselho=not args.sem_conselho)
    tools.APPS.load()
    tools.REMINDER_LISTENERS.append(lambda t: print(f"\n{YELLOW}{BOLD}⏰ Lembrete: {t}{RESET}\n> ", end="", flush=True))
    i = bot.info
    print(f"{BOLD}LIA{RESET} {DIM}· {i['params'] / 1e6:.1f}M parâmetros · treino {i['iter']} passos · "
          f"{len(tools.APPS.apps or [])} apps encontrados{RESET}")
    print(f"{DIM}Confirmação antes de abrir coisas: {'desligada' if args.auto else 'ligada'}. "
          f"Conselho: {'ligado' if bot.usar_conselho else 'desligado'}. /ajuda para comandos.{RESET}\n")
    auto, voz, debate = args.auto, args.voz, args.debate

    def confirm(name, arg):
        ans = input(f"  {YELLOW}↳ {name}: {arg}{RESET}  executar? [S/n] ").strip().lower()
        return ans in ("", "s", "sim", "y", "yes")

    while True:
        try:
            text = input(f"{BOLD}> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.startswith("/"):
            cmd, _, rest = text.partition(" ")
            if cmd in ("/sair", "/exit", "/quit"):
                break
            elif cmd == "/reset":
                bot.reset()
                print(f"{DIM}Conversa nova.{RESET}\n")
            elif cmd == "/auto":
                auto = not auto
                print(f"{DIM}Confirmação {'desligada' if auto else 'ligada'}.{RESET}\n")
            elif cmd == "/temp":
                try:
                    bot.temperature = float(rest)
                    print(f"{DIM}Temperatura {bot.temperature}.{RESET}\n")
                except ValueError:
                    print(f"{DIM}Use: /temp 0.7{RESET}\n")
            elif cmd == "/conselho":
                bot.usar_conselho = not bot.usar_conselho
                print(f"{DIM}Conselho {'ligado' if bot.usar_conselho else 'desligado'}.{RESET}\n")
            elif cmd == "/debate":
                debate = not debate
                print(f"{DIM}Debate {'visível' if debate else 'escondido'}.{RESET}\n")
            elif cmd == "/voz":
                voz = not voz
                print(f"{DIM}Voz {'ligada' if voz else 'desligada'}.{RESET}\n")
            elif cmd == "/ferramentas":
                print(DIM + tools.describe() + RESET + "\n")
            elif cmd == "/apps":
                names = sorted(a["name"] for a in tools.APPS.load(refresh=True))
                print(DIM + ", ".join(names) + RESET + "\n")
            elif cmd == "/memoria":
                fatos = tools._load_json("fatos.json", [])
                print(DIM + ("\n".join("  • " + f for f in fatos) or "  (nada guardado ainda)") + RESET + "\n")
            else:
                print(HELP + "\n")
            continue

        print(f"{CYAN}", end="", flush=True)
        reply = ""
        for ev in bot.chat(text, confirm=confirm, auto=auto):
            if ev["type"] == "conselho" and debate and ev["ev"] == "msg" and not ev["privado"]:
                color = YELLOW if ev["de"] == "Revisora" else DIM
                print(f"{RESET}{color}  {ev['de']}: {ev['texto']}{RESET}{CYAN}")
            elif ev["type"] == "token":
                print(ev["text"], end="", flush=True)
            elif ev["type"] == "text":
                reply = ev["text"]
                print(RESET)
            elif ev["type"] == "call" and (auto or not ev["confirm"]):
                print(f"{RESET}  {DIM}↳ {ev['tool']}: {ev['arg']}{RESET}")
            elif ev["type"] == "result":
                color = GREEN if ev["text"].startswith("ok") else RED
                print(f"{RESET}  {color}{DIM}{ev['text']}{RESET}{CYAN}", end="", flush=True)
                print()
        print(RESET)
        if voz:
            speak(reply)


if __name__ == "__main__":
    main()
