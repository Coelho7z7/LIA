"""
Trava de confiança (desde o v25 também no agent.py e no lia.js, TRAVA = 0.9; aqui serve para testar outros limites): no começo da resposta a uma fala do usuário, se a
probabilidade do token <|call|> for menor que o limite, a Lia conversa em vez de chamar ferramenta.

Motivo (medido no v22, bateria + conversas livres + inéditas nº 1): chamadas certas têm p(call) ~1,00; as à toa
ficam espalhadas (mediana 0,97, um quarto abaixo de 0,58). Com limite 0,9: 20 de 51 erros somem, 1 acerto vira erro.
"""

import torch


def aplicar(bot, limite):
    """Embrulha bot._prefill: só mexe no primeiro token da resposta a uma fala do usuário."""
    if not limite:
        return bot
    original = bot._prefill
    call = bot.tok.call

    def _prefill(prompt):
        logits = original(prompt)
        if prompt and prompt[-1] == bot.tok.assistant and bot.turns and bot.turns[-1][0] == "user":
            p = torch.softmax(logits.float(), -1)[call].item()
            if p < limite:
                logits = logits.clone()
                logits[call] = float("-inf")
        return logits

    bot._prefill = _prefill
    return bot
