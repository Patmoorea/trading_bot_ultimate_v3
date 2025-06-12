def verify_core_methods():
    modules = {
        "TechnicalAnalyzer": ["analyze", "_calculate_rsi"],
        "ArbitrageEngine": ["find_usdc_arbitrage", "realtime_update"],
    }
    missing = []
    for mod, methods in modules.items():
        obj = globals().get(mod, None)
        if not obj:
            missing.append(f"{mod} (module non importé)")
            continue
        for method in methods:
            if not hasattr(obj, method):
                missing.append(f"{mod}.{method}")

    if missing:
        raise RuntimeError(f"Méthodes manquantes: {', '.join(missing)}")
