from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    cache_name: str


SOURCES = {
    "cbr_reserves": Source(
        "CBR required reserves",
        "https://www.cbr.ru/vfs/hd_base/RReserves/required_reserves_table.xlsx",
        "required_reserves_table.xlsx",
    ),
    "cbr_ruonia": Source(
        "CBR RUONIA",
        "https://www.cbr.ru/hd_base/ruonia/",
        "ruonia.html",
    ),
    "cbr_repo": Source(
        "CBR repo auctions",
        "https://www.cbr.ru/hd_base/repo/",
        "repo.html",
    ),
    "cbr_keyrate": Source(
        "CBR key rate",
        "https://www.cbr.ru/hd_base/keyrate/",
        "keyrate.html",
    ),
    "minfin_ofz": Source(
        "Minfin OFZ auction results",
        "https://minfin.gov.ru/ru/document?id_4=315131-rezultaty_provedennykh_auktsionov_po_razmeshcheniyu_gosudarstvennykh_tsennykh_bumag_v_2026_godu_na_26.02.2026",
        "minfin_ofz.html",
    ),
    "fns_calendar": Source(
        "FNS tax calendar",
        "https://www.nalog.gov.ru/rn77/calendar/",
        "fns_calendar.html",
    ),
    "cbr_bank_sector": Source(
        "CBR bank sector statistics",
        "https://www.cbr.ru/statistics/bank_sector/sors/",
        "bank_sector.html",
    ),
    "roskazna_deposits": Source(
        "Federal Treasury EKS deposits",
        "https://roskazna.gov.ru/finansovye-operacii/razmeshchenie-sredstv-edinogo-kaznachejskogo-scheta/razmeshchenie-sredstv-edinogo-kaznachejskogo-scheta-na-bankovskih-depozitah",
        "roskazna_deposits.html",
    ),
    "cbr_liquidity": Source(
        "CBR banking sector liquidity",
        "https://www.cbr.ru/hd_base/bliquidity/?UniDbQuery.Posted=True&UniDbQuery.From=01.02.2014&UniDbQuery.To=02.03.2026",
        "bliquidity.html",
    ),
}


MODULE_LABELS = {
    "m1": "М1 Усреднение резервов",
    "m2": "М2 Репо ЦБ",
    "m3": "М3 ОФЗ",
    "m4": "М4 Налоги и сезонность",
    "m5": "М5 Казначейство",
}
