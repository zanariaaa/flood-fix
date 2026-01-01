from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

# dunia banjir flood
# ------------------------------------------

@dataclass
class Area:
    """
    cuma dunia palsu yang punya faktor faktor untuk terjadinya banjir.
    Semua int adalah 0..1 kecuali elevation_m soalnya meteran
    """
    name: str
    pembuangan_air: float # 0=jelek, 1=bagus
    tipe_alas: float # 0=tanah, 1=beton
    slope: float # 0=biasa, 1=miring banget
    sungai_dekat: float # 0=jauh, 1=dekat
    elevation_m: float # kalau lebih tinggi biasanya aman

@dataclass
class StatusHari:
    curah_hujan_mm: float
    prev_rain_mm: float
    soil_sat: float # 0..1
    level_sungai: float # 0..1
    pasang_surut: float # efek coastal flood, 0..1
    banjir: int 
def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)
    
def simulasi(area: Area, prev: StatusHari | None) -> StatusHari:
    """
    buat hari palsu, terus return status hari itu
    """
    prev_rain = prev.curah_hujan_mm if prev else random.uniform(0, 30)
    prev_tanah = prev.soil_sat if prev else random.uniform(0.1, 0.6)
    prev_sungai = prev.level_sungai if prev else random.uniform(0.1, 0.6)

    if random.random() < 0.12:
        curah_hujan = random.uniform(60, 220)
    
    else:
        curah_hujan = random.uniform(0, 60)

    pasang_surut = random.random()

    absorpsi = (1.0 - area.tipe_alas) * (0.6 + 0.4 * area.slope)
    tambahan_tanah = (curah_hujan / 200.0) * absorpsi
    kurang_tanah = 0.08 * (0.5 + area.pembuangan_air) * (0.5 + area.slope)
    soil_sat = clamp(prev_tanah + tambahan_tanah - kurang_tanah, 0.0, 1.0)

    absorpsi = (1.0 - area.tipe_alas) * (0.6 + 0.4 * area.slope)
    tambahan_tanah = (curah_hujan / 200.0) * absorpsi
    kurang_tanah = 0.08 * (0.5 + area.pembuangan_air) * (0.5 + area.slope)
    soil_sat = clamp(prev_tanah + tambahan_tanah - kurang_tanah, 0.0, 1.0)

    sungai_naik = (curah_hujan / 220.0) * (0.3 + 0.7 * area.sungai_dekat)
    sungai_naik += 0.12 * pasang_surut * area.sungai_dekat
    sungai_turun = 0.10 * (0.4 + area.slope) * (0.4 + area.pembuangan_air)
    level_sungai = clamp(prev_sungai + sungai_naik - sungai_turun, 0.0, 1.0)

    elev_factor = clamp(1.0 - (area.elevation_m / 200.0), 0.0, 1.0)  # 0 kalo >=200m, 1 kalo 0m
    score = 0.0
    score += 1.6 * (curah_hujan / 220.0)
    score += 1.2 * (prev_rain / 220.0)
    score += 1.4 * soil_sat
    score += 1.8 * level_sungai
    score += 1.4 * area.tipe_alas
    score += 1.1 * area.sungai_dekat
    score += 0.7 * (1.0 - area.slope)
    score += 0.9 * pasang_surut * area.sungai_dekat
    score += 1.0 * elev_factor
    score -= 1.6 * area.pembuangan_air

    # probabilitas banjir ama random
    p_banjir = sigmoid(score - 3.2) # threshold tweak
    p_banjir = clamp(p_banjir + random.uniform(-0.03, 0.03), 0.0, 1.0)
    banjir = 1 if random.random() < p_banjir else 0

    return StatusHari(
        curah_hujan_mm=curah_hujan,
        prev_rain_mm=prev_rain,
        soil_sat=soil_sat,
        level_sungai=level_sungai,
        pasang_surut=pasang_surut,
        banjir=banjir,
    )

# dataset buat semuaa

def build_dataset(area: Area, n_hari: int = 3000) -> Tuple[List[List[float]], List[int]]:
    """
    Returns (X, y)
    Features:
      [curah_hujan, prev_rain, soil_sat, level_sungai, pasang_surut, pembuangan_air, tipe_tanah, slope, sungai_dekat, elevasi_scaled]
    """
    X: List[List[float]] = []
    y: List[int] = []

    prev_state: StatusHari | None = None
    for _ in range(n_hari):
        s = simulasi(area, prev_state)
        elevasi_scaled = clamp(area.elevation_m / 200.0, 0.0, 1.0)

        X.append([
            s.curah_hujan_mm / 220.0,
            s.prev_rain_mm / 220.0,
            s.soil_sat,
            s.level_sungai,
            s.pasang_surut,
            area.pembuangan_air,
            area.tipe_alas,
            area.slope,
            area.sungai_dekat,
            elevasi_scaled
        ])
        y.append(s.banjir)
        prev_state = s

    return X, y

# --------------------------------------------------------------
# buat ke bagian paling stabil ama cli dan logreg
# --------------------------------------------------------------

def train_logreg(X: List[List[float]], y: List[int], lr: float = 0.25, epochs: int = 800) -> Tuple[List[float], float]:
    """
    descent gradien buat logreg yay
    """
    n = len(X)
    d = len(X[0])
    w = [0.0] * d
    b = 0.0

    for ep in range(1, epochs + 1):
        # gradien
        gw = [0.0] * d
        gb = 0.0
        loss = 0.0

        for i in range(n):
            z = sum(w[j] * X[i][j] for j in range(d)) + b
            p = sigmoid(z)
            yi = y[i]

            # log(0) NOOOOOO
            p_clip = clamp(p, 1e-8, 1.0 - 1e-8)
            loss += -(yi * math.log(p_clip) + (1 - yi) * math.log(1 - p_clip))

            # gradien
            dz = (p - yi)
            for j in range(d):
                gw[j] += dz * X[i][j]
            gb += dz

        # rata rata
        loss /= n
        for j in range(d):
            gw[j] /= n
        gb /= n

        for j in range(d):
            w[j] -= lr * gw[j]
        b -= lr * gb

        # print
        if ep in (1, 10, 50, 200, epochs):
            correct = 0
            for i in range(n):
                p = sigmoid(sum(w[j] * X[i][j] for j in range(d)) + b)
                pred = 1 if p >= 0.5 else 0
                correct += 1 if pred == y[i] else 0
            acc = correct / n
            print(f"epoch {ep:>4} | loss {loss:.4f} | acc {acc:.3f}")

    return w, b


def prediksi_bahaya(w: List[float], b: float, x: List[float]) -> float:
    z = sum(w[j] * x[j] for j in range(len(w))) + b
    return sigmoid(z)


def label_bahaya(p: float) -> str:
    if p < 0.20:
        return "rendah :)"
    if p < 0.45:
        return "hati hati :/"
    if p < 0.70:
        return "bahaya :()"
    return "super bahaya !!!"


# -----------------------------
# main cli
# -----------------------------


def generate_flood_grid(area: Area, rows: int, cols: int, seed: int | None = None) -> List[List[float]]:
    """
    generate grid 2D dari area, buat visualisasi peta banjir, tiap cell dapet nilai
    0..5 depth 
    """
    if seed is not None:
        random.seed(seed)

    grid: List[List[float]] = [[0.0 for _ in range(cols)] for _ in range(rows)]

    # biar di tengah suka banjir
    center_r, center_c = rows // 2, cols // 2

    for r in range(rows):
        for c in range(cols):
            a = Area(
                name=f"{area.name}_{r}_{c}",
                pembuangan_air=clamp(area.pembuangan_air + random.uniform(-0.12, 0.12), 0.0, 1.0),
                tipe_alas=clamp(area.tipe_alas + random.uniform(-0.15, 0.15), 0.0, 1.0),
                slope=clamp(area.slope + random.uniform(-0.18, 0.18), 0.0, 1.0),
                sungai_dekat=clamp(area.sungai_dekat + (1.0 - (abs(r - center_r) + abs(c - center_c)) / float(rows + cols)) * 0.5 + random.uniform(-0.15, 0.15), 0.0, 1.0),
                elevation_m=area.elevation_m + (r - center_r) * 0.6 + (c - center_c) * 0.4 + random.uniform(-8.0, 8.0),
            )

            s = simulasi(a, None)
            raw = (
                1.8 * s.level_sungai
                + 0.9 * s.soil_sat
                + 0.6 * s.pasang_surut
                + (1.2 if s.banjir else 0.0)
                + 0.8 * a.sungai_dekat
                - 0.9 * a.pembuangan_air
            )
            # faktor elevasi
            elev_factor = clamp(1.0 - (a.elevation_m / 200.0), 0.0, 1.0)
            depth = max(0.0, raw * elev_factor * 1.2)

            depth = math.pow(depth, 0.85) if depth > 0 else 0.0

            grid[r][c] = depth
    # smoothing
    smoothed = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            ssum = 0.0
            cnt = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        ssum += grid[rr][cc]
                        cnt += 1
            smoothed[r][c] = ssum / max(1, cnt)

    all_vals = sorted(v for row in smoothed for v in row)
    if not all_vals:
        return smoothed
    p95 = all_vals[int(len(all_vals) * 0.95)]
    if p95 <= 1e-6:
        p95 = max(all_vals[-1], 1.0)
    out = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            v = smoothed[r][c] / p95 * 4.0
            out[r][c] = max(0.0, min(5.0, v))

    def buat_flood_grid(area2: Area, rows2: int, cols2: int, seed2: int | None = None) -> List[List[float]]:
        return buat_flood_grid(area2, rows2, cols2, seed2)

    return out

# back-compat alias
def buat_flood_grid(area: Area, rows: int, cols: int, seed: int | None = None) -> List[List[float]]:
    return buat_flood_grid(area, rows, cols, seed)

    return grid

def main() -> None:
    random.seed(7)

    area = Area(
        name="perumahan atok",
        pembuangan_air=0.35,
        tipe_alas=0.70,
        slope=0.20,
        sungai_dekat=0.85,
        elevation_m=25.0
    )

    print(f"simulasi banjir: latihan dengan data sintetik untuk '{area.name}'")
    X, y = build_dataset(area, n_hari=3500)
    w, b = train_logreg(X, y, lr=0.35, epochs=700)

    print("\nsilahkan masukkan data hari ini untuk prediksi banjir:")
    print("contoh data:")
    print("  curah_hujan_mm kini (0..250)")
    print("  curah_hujan_mm kemarin (0..250)")
    print("  soil_sat (0..1)")
    print("  level_sungai (0..1)")
    print("  pasang_surut (0..1)")
    print("tulis 'q' to keluar.\n")

    while True:
        raw = input("hujan_kini: ").strip()
        if raw.lower() in ("q", "keluar", "exit"):
            break
        try:
            hujan_kini = float(raw)
            hujan_kemarin = float(input("hujan_kemarin: ").strip())
            soil_sat = float(input("soil_sat (0..1): ").strip())
            level_sungai = float(input("level_sungai (0..1): ").strip())
            pasang_surut = float(input("pasang_surut (0..1): ").strip())
        except ValueError:
            print("yang ditulis hanya nomer!!\n")
            continue

        x = [
            clamp(hujan_kini / 220.0, 0.0, 1.2),
            clamp(hujan_kemarin / 220.0, 0.0, 1.2),
            clamp(soil_sat, 0.0, 1.0),
            clamp(level_sungai, 0.0, 1.0),
            clamp(pasang_surut, 0.0, 1.0),
            area.pembuangan_air,
            area.tipe_alas,
            area.slope,
            area.sungai_dekat,
            clamp(area.elevation_m / 200.0, 0.0, 1.0)
        ]

        p = prediksi_bahaya(w, b, x)
        print(f"\n probabilitas banjir: {p*100:.1f}%  |  bahaya: {label_bahaya(p)}")

        # kata2 tambahan
        if p >= 0.70:
            print("lari sekarang\n")
        elif p >= 0.45:
            print("liat dulu, radak bahaya.\n")
        else:
            print("bobok :)\n")


if __name__ == "__main__":
    main()