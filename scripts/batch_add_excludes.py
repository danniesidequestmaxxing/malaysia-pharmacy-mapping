"""
batch_add_excludes.py
---------------------
Resolve a list of pharmacy names → pharmacy_ids in the Google Places
CSV, append the ids to data/pharmacies_google_johor_excluded.txt, and
rewrite the main CSV + audit CSV to drop them.

Usage: edit NAMES_TO_EXCLUDE below, then `python scripts/batch_add_excludes.py`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


NAMES_TO_EXCLUDE = [
    # Screenshot 1 (rows 258-271)
    "Pathology & Clinical Laboratory (M) Sdn. Bhd. (Pathlab)",
    "Penatian setia",
    "Penawar Pharmacy Tioman",
    "Penawar Resdung",
    "Pengedar Shaklee Bandar Putra Fatimah",
    "Pengedar Shaklee Tongkang Pechah",
    "Perniagaan Ban Heo",
    "Perniagaan Kang Yuen",
    "Perniagaan Pt Tunas Budi",
    "Perniagaan Ubat Heng Long",
    "Perniagaan Yuh Feng",
    "Perubatan Cina Lim Han Lin",
    "Pesaka",
    "Pet Paradise Clinic",
    # Screenshot 2 (rows 350-372)
    "万春济药行",
    "中和堂",
    "仁生堂",
    "保康参茸药行",
    "元和堂药材贸易",
    "北乾那那药店",
    "双许诊所",
    "吉打中药店",
    "同和参茸药行 Thong Hoe Medical Store",
    "天益壽中医药行",
    "廖艾琦中医诊所",
    "慧德安参茸药行(分行)",
    "我",
    "新裕丰蔘茸海味药行",
    "新邦慈慧精舍",
    "杏福堂中医诊所 HAPPINESS TRADITIONAL CHINESE MEDICINE CENTRE",
    "栢龄药材店凉茶铺 Kedai Ubat Berin - Taman Pelangi",
    "永和栈百货药行",
    "祖传白鹤拳术医药所",
    "罗伟忠推拿中医医务所 TABIB URUT DAN UBAT TRADISI CINA",
    "育虹草药园",
    "达安参茸药行",
    "颜朝基传统医药中心",
    "龙山中西药行",  # match prefix only — Chinese parens differ between sources
    # Long text list
    "Fo Heng Enterprise Sdn Bhd",
    "Foo Sang Trading",
    "Formula Pte Ltd",
    "Fu Sin Nourishing House",
    "GNA",
    "Ginseng Biotech Sdn Bhd",
    "HAIDAH BEAUTY SHOP",
    "Iwaki",
    "Jementah Healthcare",
    "Jim Fong",
    "KEDAI UBAT CHAY SENG",  # has chinese suffix in CSV; match prefix
    "KHAI ZAM BURGER",
    "KLINIK DR HUDA",
    "KPH Care",
    "Kedai Shaklee Batu Pahat - Ainun",
    "Klinik Haiwan Alby",
    "Klinik Rawatan Islam Nur Isyraq",
    "Klinik Scientex Dr Salmah",
    "Klinik desa",
    "Koh Xing Trading",  # has TABIB CINA suffix; match prefix
    "Kumpulan Perubatan Penawar",
    "LAM FONG ENTERPRISE SDN. BHD.",
    "LIT GEAP TRADING",
    "Letak dan Ambil Klinik Kesihatan Tenggaroh 02",
    "Lian Heng",
    "Long Zhen Enterprise Sdn Bhd",
    "Lotus herb center",
    "Lovy Dispensaries Segamat",
    "Low Scinentific Sdn. Bhd.",
    "Luen Fook Medicine Sdn. Bhd.",
    "MEIMEI 88 ENTERPRISE",
    "Makmal Loji Pandu Biodiesel, UTHM",
    "Malaysia",
    "Mersing",
    "Minyak lintah",
    "Muzalia enterprise",
    "Nani",
    "Nurha Eskayvie",
    "Poh Ming Tong Medical Store",
    "Procuci Pontian",
    "Pusat Rawatan Homeopathy Homeomed - Parit Raja",
    "Pusat Rawatan Seisi Alam",
    "Pusat Tabib Ubat Cina Gau Tiow Kee",
    "Rawatan Tabib Tian Chin",
    "Reban bintak kempas",
    "Rokenrol Airbrush Studio. Gondrong Keren",
    "SIFU URUT",
    "SIN CHEAP HONG TRADING",
    "SNT Agro Farm",
    "STF Medic Enterprise",
    "STOKIS C2JOY BATU PAHAT",
    "Seng Seng tranditional medicine",
    "Sengse Pohkian",
    "Shaklee Kahang Timur",
    "Simpang masuk pot dadah",
    "Sinar Permata Pontian",
    "Sri Tanjung",
    "Sri Tanjung Anggerik",
    "Tegal Trading",
    "Tol tiram",
    "Tong Huat",
    "Traditional Massage Center Top One Massage",
    "WENG HENG GINSENG MEDICAL STORE S/B",
    "Wansern Biotechnology",
    "Yong Lian Medical Hall",
    "Yuen Sen For Medicine Shop",
    "Zhang Choon",
    "home rahul",
    "kampung sungai punggur laut timur",
    "kedai ubat ksm ho",
    "sheng heng",
    "taman soga",
]

# Names too short to safely substring-match — require exact-equal match.
SHORT_NAMES_EXACT = {"GNA", "Iwaki", "Malaysia", "Mersing", "Nani",
                     "Pesaka", "Tong Huat", "我", "sheng heng",
                     "taman soga", "Lian Heng", "home rahul",
                     "Sri Tanjung"}

CSV = Path("data/pharmacies_google_johor.csv")
EXCLUDE = Path("data/pharmacies_google_johor_excluded.txt")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def main() -> None:
    df = pd.read_csv(CSV)
    df["_n"] = df["name"].map(_norm)

    matched: list[tuple[str, str, str]] = []  # (search_name, pharmacy_id, name)
    misses: list[str] = []
    seen_ids: set[str] = set()

    for needle in NAMES_TO_EXCLUDE:
        n = _norm(needle)
        if needle in SHORT_NAMES_EXACT:
            hits = df[df["_n"] == n]
        else:
            hits = df[df["_n"].str.startswith(n)]
            if hits.empty:
                hits = df[df["_n"].str.contains(re.escape(n), regex=True)]
        if hits.empty:
            misses.append(needle)
            continue
        for _, row in hits.iterrows():
            pid = row["pharmacy_id"]
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            matched.append((needle, pid, row["name"]))

    print(f"Matched {len(matched)} ids from {len(NAMES_TO_EXCLUDE)} names")
    if misses:
        print(f"\nNo match for {len(misses)} names — please check spelling:")
        for m in misses:
            print(f"  {m!r}")

    # Append to exclude file (preserving any prior entries).
    existing_ids: set[str] = set()
    existing_text = EXCLUDE.read_text(encoding="utf-8") if EXCLUDE.exists() else ""
    for line in existing_text.splitlines():
        s = line.split("#", 1)[0].strip()
        if s:
            existing_ids.add(s)

    new_lines: list[str] = []
    for needle, pid, name in matched:
        if pid in existing_ids:
            continue
        existing_ids.add(pid)
        # Strip newlines and over-long names from the comment.
        clean_name = re.sub(r"\s+", " ", name).strip()[:80]
        new_lines.append(f"{pid}  # {clean_name}")

    if new_lines:
        if existing_text and not existing_text.endswith("\n"):
            existing_text += "\n"
        EXCLUDE.write_text(existing_text + "\n".join(new_lines) + "\n",
                            encoding="utf-8")
        print(f"\nAppended {len(new_lines)} new ids to {EXCLUDE}")
    else:
        print("\nNo new ids to append — already in exclude file.")


if __name__ == "__main__":
    main()
