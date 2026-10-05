"""Calendario nacional e eventos comerciais conhecidos."""

from datetime import date

import holidays


EVENTOS = {
    "2025-11-11": ("11_11", "promocional"),
    "2025-11-28": ("black_friday", "promocional"),
    "2025-12-01": ("cyber_monday", "promocional"),
    "2026-03-15": ("dia_consumidor", "promocional"),
    "2026-04-05": ("pascoa", "presente"),
    "2026-05-10": ("dia_maes", "presente"),
    "2026-06-12": ("dia_namorados", "presente"),
}

FERIADOS_EXTRAS = {
    date(2026, 2, 16),
    date(2026, 2, 17),
    date(2026, 6, 4),
}


def feriados_br(anos) -> set[date]:
    """Retorna os feriados nacionais brasileiros dos anos informados."""
    anos_lista = [anos] if isinstance(anos, int) else list(anos)
    extras = {feriado for feriado in FERIADOS_EXTRAS if feriado.year in anos_lista}
    return set(holidays.country_holidays("BR", years=anos_lista).keys()) | extras
