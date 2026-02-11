"""Deterministic mock payloads for future LLM-based nodes and prompt experiments."""

ANALYZER_RESPONSE = {
    "domain": "calculo_i",
    "constraints": ["x>0"],
    "complexity_score": 0.35,
    "plan": [
        "Validar domínio e restrições",
        "Selecionar estratégia simbólica",
        "Executar validação cruzada",
    ],
}

SOLVER_RESPONSE = {
    "strategy": "symbolic",
    "symbolic_result": "2*x",
    "numeric_result": None,
}

VERIFIER_RESPONSE = {
    "success": True,
    "checks": {"plausible_symbolic": True, "numeric_finite": True},
    "notes": [],
    "confidence": 0.95,
}
