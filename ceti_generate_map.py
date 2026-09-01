"""
ceti_generate_map.py — Générateur de carte de situation CETI
UNITe PV — AO CRE PPE2 Neutre Période 5

Expose generer_carte() appelée par app.py.
Peut aussi être lancé directement depuis un terminal.
"""

import os, re, io, math, zipfile, tempfile
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.ops import unary_union
from pyproj import Transformer

# ── Cache tuiles IGN persistant entre sessions ────────────────────────────────
# Par défaut contextily supprime le cache à la fin de chaque session Python.
# On le redirige vers un dossier permanent pour éviter de re-télécharger
# les mêmes tuiles à chaque génération.
try:
    import contextily as _ctx_init
    _TILE_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               ".tile_cache")
    os.makedirs(_TILE_CACHE, exist_ok=True)
    _ctx_init.set_cache_dir(_TILE_CACHE)
except Exception:
    pass  # contextily absent ou erreur non bloquante


# ════════════════════════════════════════════════════════════════
# COULEURS CONVENTIONNELLES PLU
# ════════════════════════════════════════════════════════════════
COULEURS_PLU = {
    "U":  {"fc": "#FF6B6B", "ec": "#CC2200", "label": "Zone U — Urbaine"},
    "AU": {"fc": "#FFB347", "ec": "#CC6600", "label": "Zone AU — À urbaniser"},
    "A":  {"fc": "#F9E04B", "ec": "#CC9900", "label": "Zone A — Agricole"},
    "N":  {"fc": "#74C476", "ec": "#2D7A2D", "label": "Zone N — Naturelle"},
}
COULEUR_PLU_DEFAUT = {"fc": "#CCCCCC", "ec": "#888888", "label": "Zone — Autre"}


def charger_zones_urbanisme(x0, y0, x1, y1):
    """
    Interroge le Geoportail de l'Urbanisme (GPU) via API Carto IGN.
    Endpoint : https://apicarto.ign.fr/api/gpu/zone-urba

    Parametres : emprise en Lambert-93 (x0,y0 = coin SO, x1,y1 = coin NE)
    Retourne
    --------
    GeoDataFrame (n > 0)   : zones PLU trouvees
    GeoDataFrame vide      : API OK mais aucune zone = commune sous RNU
    None                   : echec de l'API (reseau, timeout...)
    """
    import json, requests
    from pyproj import Transformer as _Tr

    tr = _Tr.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    lon0, lat0 = tr.transform(x0, y0)
    lon1, lat1 = tr.transform(x1, y1)

    geom_bbox = {
        "type": "Polygon",
        "coordinates": [[[lon0, lat0], [lon1, lat0],
                          [lon1, lat1], [lon0, lat1],
                          [lon0, lat0]]],
    }
    params = {
        "geom":   json.dumps(geom_bbox, separators=(",", ":")),
        "_limit": 1000,
    }
    try:
        r = requests.get(
            "https://apicarto.ign.fr/api/gpu/zone-urba",
            params=params, timeout=30,
            headers={"Accept": "application/json"},
        )
        r.raise_for_status()
        features = r.json().get("features", [])
        if not features:
            print("GPU : aucune zone PLU (commune sous RNU probable)")
            # GeoDataFrame vide = signal RNU
            return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"),
                                    crs="EPSG:4326")
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        gdf = gdf.to_crs(epsg=2154)
        print("GPU : {} zones PLU chargees".format(len(gdf)))
        return gdf
    except Exception as e:
        print("GPU : echec ({}) : {}".format(type(e).__name__, str(e)[:80]))
        return None   # None = erreur reseau


def type_zone(libelle):
    """Retourne la clé de couleur (U/AU/A/N) depuis le libellé de zone."""
    if libelle is None:
        return None
    lib = str(libelle).upper().strip()
    for key in ["AU", "U", "A", "N"]:   # AU avant U pour éviter les faux positifs
        if lib.startswith(key):
            return key
    return None


# ════════════════════════════════════════════════════════════════
# PARAMETRES — lancement direct terminal uniquement
# ════════════════════════════════════════════════════════════════
SHP_PATH       = ""   # chemin local uniquement
NOM_PROJET     = "EMO 21 — Gissey / Darcey"
RECUL_CAPTEURS = 10
BUFFER_CARTE   = 650
ECHELLE        = 5000
URBANISME      = ""
FOND_AERIEN    = True
OUTPUT_DIR     = ""   # chemin local uniquement
DPI            = 150
ZH_PATH        = None   # chemin couche zones humides (optionnel)
ELEMENTS_PATH  = None   # chemin couche éléments techniques KML (optionnel)


# ════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ════════════════════════════════════════════════════════════════

def to_dms(deg, is_lat):
    hemi = ("N" if deg >= 0 else "S") if is_lat else ("E" if deg >= 0 else "O")
    deg  = abs(deg)
    d = int(deg); m = int((deg - d) * 60); s = (deg - d - m / 60) * 3600
    return "{:02d}\u00b0{:02d}'{:05.2f}'' {}".format(d, m, s, hemi)


def calcul_extremaux(terrain, tr, echelle=5000):
    """
    Selectionne 4 points bien repartis sur le contour reel du terrain.

    Algorithme
    ----------
    0. Echantillonnage de N_ECH points candidats sur le contour (et non sur
       l'enveloppe convexe) : les points restent sur le perimetre meme si la
       forme est concave, allongee ou composee de plusieurs parcelles, et le
       resultat ne depend plus du nombre de sommets (un triangle fonctionne).
    1. A = candidat le plus au nord (max Y).
    2. B = candidat maximisant la distance a A.
    3. C = candidat maximisant la distance minimale a {A, B} (maximin).
    4. D = candidat maximisant la distance minimale a {A, B, C} (maximin).

    Chaque point recoit ensuite l'une des 4 directions cardinales d'annotation,
    attribuee sans doublon selon sa direction sortante depuis le centroide.
    """
    cx, cy   = terrain.centroid.x, terrain.centroid.y

    minx, miny, maxx, maxy = terrain.bounds
    M_PER_PT = echelle * 0.0254 / 72   # metres par point typographique
    BOX_W, BOX_H = 92, 52              # taille boite en pts (fixe — la fonte est fixe)

    # PUSH : depasse la demi-largeur du terrain + marge lisible, cap a 150 pts
    terrain_hw = max(maxx - minx, maxy - miny) / 2
    PUSH = max(min(int(terrain_hw / M_PER_PT) + 25, 150), 40)

    N_ECH = 400   # points candidats echantillonnes sur le contour
    # Candidats sur le contour REEL du terrain (et non le hull) : garantit que les
    # points sont sur le perimetre, y compris sur terrain concave ou multi-parcelles
    _bnd   = terrain.boundary
    _cand  = [np.array(_bnd.interpolate(i / N_ECH, normalized=True).coords)[0][:2]
              for i in range(N_ECH)]
    # A = point le plus au nord (convention conservee)
    _sel = [int(np.argmax([c[1] for c in _cand]))]
    # B, C, D = selection gloutonne du point le plus eloigne des deja retenus (maximin)
    for _ in range(3):
        _sel.append(max(
            (k for k in range(N_ECH) if k not in _sel),
            key=lambda k: min(math.dist(_cand[k], _cand[s]) for s in _sel)
        ))

    # Offsets d'annotation : 4 directions cardinales, une par point.
    _tmpl = [
        ((-BOX_W // 2,  PUSH),           (0.0,  1.0)),   # N
        ((PUSH, -BOX_H // 2),            (1.0,  0.0)),   # E
        ((-BOX_W // 2, -(BOX_H + PUSH)), (0.0, -1.0)),   # S
        ((-(BOX_W + PUSH), -BOX_H // 2), (-1.0, 0.0)),   # O
    ]

    # Position et direction sortante (depuis le centroide) des 4 points retenus
    _pxy, _dirs = [], []
    for k in _sel:
        px, py = float(_cand[k][0]), float(_cand[k][1])
        _vx, _vy = px - cx, py - cy
        _nrm = max(math.hypot(_vx, _vy), 1e-9)
        _pxy.append((px, py))
        _dirs.append((_vx / _nrm, _vy / _nrm))

    # Longueur de la ligne de rappel qui traverserait l'emprise
    from shapely.geometry import LineString as _LS
    def _coupe(pt, adx, ady):
        bx = pt[0] + (adx + BOX_W / 2) * M_PER_PT
        by = pt[1] + (ady + BOX_H / 2) * M_PER_PT
        try:
            return _LS([pt, (bx, by)]).intersection(terrain).length
        except Exception:
            return 0.0

    # Attribution optimale sur les 24 permutations : on minimise d'abord la
    # longueur totale de fleche traversant le terrain (une ligne de rappel ne
    # doit pas couper l'emprise), puis on maximise l'alignement sortant.
    # L'heuristique par centroide seule echoue sur les formes allongees en
    # diagonale : le dernier point recoit par elimination une direction qui
    # renvoie la fleche a travers la parcelle.
    import itertools as _it
    _best, _bkey = None, None
    for _perm in _it.permutations(range(len(_tmpl))):
        _c = _a = 0.0
        for _i, _t in enumerate(_perm):
            (adx, ady), (dux, duy) = _tmpl[_t]
            _c += _coupe(_pxy[_i], adx, ady)
            _a += _dirs[_i][0] * dux + _dirs[_i][1] * duy
        _key = (round(_c, 1), -_a)
        if _bkey is None or _key < _bkey:
            _bkey, _best = _key, _perm

    result = []
    for _i, lbl in enumerate("ABCD"[:len(_sel)]):
        px, py = _pxy[_i]
        (adx, ady), (dux, duy) = _tmpl[_best[_i]]
        lon, lat = tr.transform(px, py)
        result.append({
            "label":  "Pt {}".format(lbl),
            "x": px, "y": py,
            "lat":    to_dms(lat, True),
            "lon":    to_dms(lon, False),
            "ann_dx": adx, "ann_dy": ady,
            "_ux": dux, "_uy": duy,
        })

    # Anti-chevauchement AABB : push sur l'axe de chevauchement minimal
    THRESH_X = (BOX_W + 6) * M_PER_PT
    THRESH_Y = (BOX_H + 6) * M_PER_PT
    STEP     = 12

    for _ in range(60):
        moved = False
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                ri, rj = result[i], result[j]
                cxi = ri["x"] + (ri["ann_dx"] + BOX_W / 2) * M_PER_PT
                cyi = ri["y"] + (ri["ann_dy"] + BOX_H / 2) * M_PER_PT
                cxj = rj["x"] + (rj["ann_dx"] + BOX_W / 2) * M_PER_PT
                cyj = rj["y"] + (rj["ann_dy"] + BOX_H / 2) * M_PER_PT
                ovx = THRESH_X - abs(cxi - cxj)
                ovy = THRESH_Y - abs(cyi - cyj)
                if ovx > 0 and ovy > 0:
                    if ovx < ovy:
                        push = math.copysign(ovx / 2 / M_PER_PT + STEP, cxi - cxj) or float(STEP)
                        result[i]["ann_dx"] = int(ri["ann_dx"] + push)
                        result[j]["ann_dx"] = int(rj["ann_dx"] - push)
                    else:
                        push = math.copysign(ovy / 2 / M_PER_PT + STEP, cyi - cyj) or float(STEP)
                        result[i]["ann_dy"] = int(ri["ann_dy"] + push)
                        result[j]["ann_dy"] = int(rj["ann_dy"] - push)
                    moved = True
        if not moved:
            break

    for pt in result:
        pt.pop("_ux", None)
        pt.pop("_uy", None)

    return result


def charger_geodata(path):
    """
    Charge un fichier géographique quel que soit son format :
    - .zip  → shapefile extrait dans un dossier temporaire
    - .kml  → lu directement par geopandas
    - .geojson / .json → lu directement
    Retourne un GeoDataFrame en EPSG:2154 (Lambert-93).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".zip":
        tmpdir = tempfile.mkdtemp()
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(tmpdir)
        shp = None
        for root, _, files in os.walk(tmpdir):
            for f in files:
                if f.endswith(".shp"):
                    shp = os.path.join(root, f)
                    break
        if shp is None:
            raise ValueError("Aucun .shp trouvé dans le zip : {}".format(path))
        gdf = gpd.read_file(shp)

    elif ext == ".kml":
        import fiona
        # Activer le driver KML (désactivé par défaut dans fiona)
        fiona.drvsupport.supported_drivers["KML"]  = "rw"
        fiona.drvsupport.supported_drivers["LIBKML"] = "rw"
        gdf = gpd.read_file(path, driver="KML")

    elif ext in (".geojson", ".json"):
        gdf = gpd.read_file(path)

    else:
        # Tentative générique
        gdf = gpd.read_file(path)

    # Reprojection en Lambert-93 si nécessaire
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    if gdf.crs.to_epsg() != 2154:
        gdf = gdf.to_crs(epsg=2154)

    return gdf


def draw_geom(ax, geom, fc="none", ec="black", lw=1.5,
              alpha_fill=0.3, ls="-", zorder=2):
    """Trace une géométrie Shapely sur un axe matplotlib."""
    geoms = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
    for g in geoms:
        if g.geom_type == "Polygon":
            xs, ys = g.exterior.xy
            if fc != "none":
                ax.fill(xs, ys, color=fc, alpha=alpha_fill, zorder=zorder)
            ax.plot(xs, ys, color=ec, linewidth=lw, linestyle=ls, zorder=zorder + 1)
        elif g.geom_type == "LineString":
            xs, ys = g.xy
            ax.plot(xs, ys, color=ec, linewidth=lw, linestyle=ls, zorder=zorder)
        elif g.geom_type == "Point":
            ax.plot(g.x, g.y, "o", color=ec, markersize=4, zorder=zorder)


def draw_hatch(ax, geom, ec="#0077BB", fc="#AEE4FF", hatch="////",
               alpha_fill=0.25, lw=1.5, zorder=6):
    """
    Trace un polygone hachuré (zones humides).
    Le hachure est appliqué via matplotlib directement.
    """
    geoms = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
    for g in geoms:
        if g.geom_type != "Polygon":
            continue
        xs, ys = g.exterior.xy
        ax.fill(xs, ys, fc=fc, alpha=alpha_fill, zorder=zorder)
        ax.fill(xs, ys, fc="none", hatch=hatch, ec=ec,
                linewidth=0.3, alpha=0.6, zorder=zorder + 1)
        ax.plot(xs, ys, color=ec, linewidth=lw, zorder=zorder + 2)


# ════════════════════════════════════════════════════════════════
# FONCTION PRINCIPALE
# ════════════════════════════════════════════════════════════════

def generer_carte(shp_path, nom_projet, recul_capteurs=10, urbanisme="",
                  echelle=5000, fond_aerien=True, dpi=150, buffer_carte=800,
                  tick_deg=0.005, zh_path=None, elements_path=None,
                  kml_panneaux=None, kml_pistes=None,
                  urba_terrain=False, urba_buffer=True,
                  format="png", debug=False):
    """
    Génère la carte de situation CETI et retourne les bytes PNG ou PDF.

    Paramètres
    ----------
    shp_path        : chemin .shp terrain d'implantation
    nom_projet      : nom affiché dans le titre et le nom de fichier
    recul_capteurs  : recul zone capteurs en mètres (défaut 10)
    urbanisme       : texte libre encart urbanisme
    echelle         : dénominateur échelle (défaut 5000)
    fond_aerien     : True = IGN Géoportail
    dpi             : résolution image (défaut 150)
    buffer_carte    : rayon emprise carte en mètres (défaut 800)
    tick_deg        : intervalle ticks WGS84 en degrés (défaut 0.005)
    zh_path         : chemin couche zones humides (.zip, .kml, .geojson) — optionnel
    elements_path   : chemin couche éléments techniques (.kml) — fallback si un seul KML
    kml_panneaux    : chemin KML rangées de panneaux (LineStrings courtes + polygones)
    kml_pistes      : chemin KML pistes/postes, ou liste de chemins (1 ou 2 fichiers) — optionnel
    format          : "png" (défaut) ou "pdf"

    Retourne : bytes PNG
    """

    # ── Horodatage ────────────────────────────────────────────────────────────
    import time as _time
    _t0 = _time.time()
    def _ts(label):
        if debug:
            print("[{:6.1f}s] {}".format(_time.time() - _t0, label))

    _ts("DEBUT generer_carte")

    # ── Géométries terrain ────────────────────────────────────────────────────
    gdf      = gpd.read_file(shp_path)
    terrain  = unary_union(gdf.geometry)
    capteurs = terrain.buffer(-recul_capteurs)
    buf600   = terrain.buffer(600)
    _ts("Shapefile + buffers terrain")

    minx, miny, maxx, maxy = terrain.bounds
    pad = buffer_carte
    x0, x1 = minx - pad, maxx + pad
    y0, y1 = miny - pad, maxy + pad
    geo_w, geo_h = x1 - x0, y1 - y0

    # ── Chargement couches optionnelles ───────────────────────────────────────
    from shapely.ops import polygonize as _polygonize

    gdf_zh = charger_geodata(zh_path) if zh_path else None
    _ts("Chargement ZH")

    # Résolution des sources KML éléments techniques :
    # Priorité : kml_panneaux / kml_pistes (mode deux KML)
    # Fallback : elements_path (mode ancien, un seul KML)
    gdf_panneaux = charger_geodata(kml_panneaux) if kml_panneaux else None

    # kml_pistes accepte un chemin unique (str) ou une liste de chemins
    if kml_pistes:
        import pandas as _pd_p
        _paths_pistes = [kml_pistes] if isinstance(kml_pistes, str) else list(kml_pistes)
        _gdfs_pistes  = [charger_geodata(p) for p in _paths_pistes if p]
        _gdfs_pistes  = [g for g in _gdfs_pistes if g is not None and len(g) > 0]
        gdf_pistes = _pd_p.concat(_gdfs_pistes, ignore_index=True) if _gdfs_pistes else None
    else:
        gdf_pistes = None

    _ts("Chargement KML panneaux + pistes")
    # Mode un seul KML (fallback) : on l'utilise comme gdf_panneaux
    if gdf_panneaux is None and elements_path:
        gdf_panneaux = charger_geodata(elements_path)

    # gdf_elts = union pour compatibilité avec le bloc d'affichage existant
    if gdf_panneaux is not None and gdf_pistes is not None:
        import pandas as _pd
        gdf_elts = _pd.concat([gdf_panneaux, gdf_pistes], ignore_index=True)
    elif gdf_panneaux is not None:
        gdf_elts = gdf_panneaux
    elif gdf_pistes is not None:
        gdf_elts = gdf_pistes
    else:
        gdf_elts = None

    # ── Construction zone capteurs depuis KML panneaux ────────────────────────
    capteurs_depuis_kml = None   # MultiPolygon ou None (par cluster)
    capteurs_clusters   = []     # liste de Polygon/MultiPolygon individuels

    if gdf_panneaux is not None and len(gdf_panneaux) > 0:
        mask_poly_p  = gdf_panneaux.geometry.geom_type.isin(["Polygon","MultiPolygon"])
        mask_lines_p = gdf_panneaux.geometry.geom_type.isin(["LineString","MultiLineString"])

        # 6.5 m : fusionne les espaces entre tables jusqu'a 13 m
        BUF_CLOSE = 6.5
        base_parts = []
        if mask_lines_p.any():
            _u = unary_union(gdf_panneaux[mask_lines_p].geometry)
            _pg = [p for p in _polygonize(_u) if p.area >= 0.5]
            if _pg:
                base_parts.append(unary_union(_pg))
                print("Polygonize panneaux : {} polygones, {:.2f} ha".format(
                    len(_pg), sum(p.area for p in _pg) / 10000))
            else:
                base_parts.append(_u.buffer(BUF_CLOSE))
                print("Polygonize : 0 anneau -> fallback buffer sur lignes")
        if mask_poly_p.any():
            base_parts.append(unary_union(gdf_panneaux[mask_poly_p].geometry))
        if base_parts:
            _ts("Closing debut")
            _base = unary_union(base_parts)
            zones_merged = _base.buffer(BUF_CLOSE).buffer(-BUF_CLOSE)
            _ts("Closing termine")
            n_zones = len(list(zones_merged.geoms)) if hasattr(zones_merged, "geoms") else (0 if zones_merged.is_empty else 1)
            print("Closing +{}/-{}m : {} zones, {:.2f} ha".format(BUF_CLOSE, BUF_CLOSE, n_zones, zones_merged.area/10000))

        if base_parts:
            # 4. Soustraction corridors pistes si KML pistes fourni
            BUF_PISTE = 3
            if gdf_pistes is not None and len(gdf_pistes) > 0:
                # ATTENTION — ordre des operations volontairement inverse de
                # celui des panneaux. Sur des LIGNES qui s'entrecroisent,
                # unary_union produit une geometrie massivement auto-intersectee
                # dont le buffer fait exploser GEOS. Mesure sur Chamouilley
                # (366 lignes, 76 km, 15 595 sommets) :
                #   unary_union(pistes).buffer(3)  -> 95,1 s et 1946 Mo
                #   pistes.buffer(3).union_all()   ->  0,8 s et  185 Mo
                # resultat geometrique identique.
                # Ne pas "harmoniser" avec le traitement des panneaux, qui lui
                # exige unary_union AVANT le buffer parce qu'il porte sur des
                # polygones simples.
                _corridors = gdf_pistes.geometry.buffer(BUF_PISTE).union_all()
                zones_merged = zones_merged.difference(_corridors)
                n_ap = (len(list(zones_merged.geoms))
                        if hasattr(zones_merged, "geoms")
                        else (0 if zones_merged.is_empty else 1))
                print("Soustraction corridors pistes ({}m, {:.2f} ha) -> {} clusters".format(
                    BUF_PISTE, _corridors.area / 10000, n_ap))

            # 5. Éclater en clusters individuels, buffer +3m pour zone capteurs
            if hasattr(zones_merged, "geoms"):
                raw_clusters = [g for g in zones_merged.geoms if not g.is_empty]
            else:
                raw_clusters = [zones_merged] if not zones_merged.is_empty else []

            capteurs_clusters = [c.buffer(5) for c in raw_clusters]
            if capteurs_clusters:
                capteurs_depuis_kml = unary_union(capteurs_clusters)
                print("Zone capteurs : {} cluster(s), {:.2f} ha total".format(
                    len(capteurs_clusters), capteurs_depuis_kml.area / 10000))
        else:
            print("Avertissement : KML panneaux sans géométries utilisables")

    elif gdf_elts is not None and len(gdf_elts) > 0:
        # Ancien fallback : buffer 5m autour de tous les éléments
        capteurs_depuis_kml = unary_union(gdf_elts.geometry).buffer(5)
        capteurs_clusters   = [capteurs_depuis_kml]
        print("Zone capteurs (fallback buffer) : {:.2f} ha".format(
            capteurs_depuis_kml.area / 10000))

    _ts("Zone capteurs terminee")
    # ZH : on decoupe a l emprise du terrain d implantation
    # (la couche ZH peut etre tres etendue)
    if gdf_zh is not None:
        # Certains exports SIG livrent les ZH comme des anneaux FERMES stockes en
        # LineString et non en Polygon. unary_union donnait alors un
        # MultiLineString : l intersection n etait pas vide (donc la legende
        # s affichait) mais draw_hatch ne trace que des Polygon, si bien qu aucune
        # zone n apparaissait sur la carte. On polygonise donc les anneaux fermes.
        _zh_parts = [g for g in gdf_zh.geometry
                     if g is not None and not g.is_empty
                     and g.geom_type in ("Polygon", "MultiPolygon")]
        _zh_lines = [g for g in gdf_zh.geometry
                     if g is not None and not g.is_empty
                     and g.geom_type in ("LineString", "MultiLineString")]
        if _zh_lines:
            _zh_rings = [p for p in _polygonize(unary_union(_zh_lines)) if p.area > 0]
            if _zh_rings:
                _zh_parts.extend(_zh_rings)
                print("ZH : {} anneau(x) ferme(s) converti(s) en polygone(s)".format(
                    len(_zh_rings)))
            elif not _zh_parts:
                print("Avertissement : ZH fournies en lignes ouvertes, "
                      "non polygonisables — couche ignoree")

        zh_geom = unary_union(_zh_parts).intersection(terrain) if _zh_parts else None
        if zh_geom is None or zh_geom.is_empty:
            print("Avertissement : la couche ZH n intersecte pas le terrain — ignoree")
            zh_geom = None
        else:
            print("ZH decoupee au terrain : {:.2f} ha".format(zh_geom.area / 10000))
    else:
        zh_geom = None
    _ts("Intersection ZH terminee")

    # ── Taille figure à l'échelle exacte ──────────────────────────────────────
    MARGIN_L, MARGIN_R     = 0.90, 0.15
    MARGIN_TOP, MARGIN_BOT = 0.95, 0.65

    ax_w_in  = geo_w / echelle / 0.0254
    ax_h_in  = geo_h / echelle / 0.0254
    fig_w_in = ax_w_in + MARGIN_L + MARGIN_R
    fig_h_in = ax_h_in + MARGIN_TOP + MARGIN_BOT
    echelle_lbl = "1 / {:,}".format(echelle).replace(",", " ")

    print("Axe : {:.1f}\u00d7{:.1f} cm | Figure : {}\u00d7{} px".format(
        ax_w_in * 2.54, ax_h_in * 2.54,
        int(fig_w_in * dpi), int(fig_h_in * dpi)))

    # ── Points extrémaux WGS84 DMS ────────────────────────────────────────────
    tr       = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    extremal = calcul_extremaux(terrain, tr, echelle)

    _ts("Figure matplotlib creee (avant fond IGN)")
    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax  = fig.add_axes([
        MARGIN_L   / fig_w_in,
        MARGIN_BOT / fig_h_in,
        ax_w_in    / fig_w_in,
        ax_h_in    / fig_h_in,
    ])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_autoscale_on(False)

    # ── Fond IGN ──────────────────────────────────────────────────────────────
    fond_ok = False
    ign_url = (
        "https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile"
        "&VERSION=1.0.0&LAYER=ORTHOIMAGERY.ORTHOPHOTOS"
        "&STYLE=normal&TILEMATRIXSET=PM"
        "&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&FORMAT=image/jpeg"
    )
    if fond_aerien:
        _ts("IGN fetch debut (contextily bounds2img)")
        try:
            import contextily as ctx
            from rasterio.transform import from_bounds as _rio_bounds
            from rasterio.warp import (reproject as _rio_proj, Resampling,
                                       calculate_default_transform as _rio_cdt)
            from rasterio.crs import CRS as _CRS

            _to3857 = Transformer.from_crs("EPSG:2154", "EPSG:3857", always_xy=True)
            _bx0, _by0 = _to3857.transform(x0, y0)
            _bx1, _by1 = _to3857.transform(x1, y1)

            # n_connections=4 : téléchargement parallèle des tuiles (~3× plus rapide)
            _img_wm, _ext = ctx.bounds2img(_bx0, _by0, _bx1, _by1,
                                            zoom="auto", source=ign_url, ll=False,
                                            n_connections=4)
            _ts("IGN tuiles telechargees ({} x {} px)".format(_img_wm.shape[1], _img_wm.shape[0]))
            _H, _W, _nb = _img_wm.shape
            _src_crs = _CRS.from_epsg(3857)
            _dst_crs = _CRS.from_epsg(2154)
            _west, _east, _south, _north = _ext
            _src_tf  = _rio_bounds(_west, _south, _east, _north, _W, _H)
            _dst_tf, _dw, _dh = _rio_cdt(_src_crs, _dst_crs, _W, _H,
                                           left=_west, bottom=_south,
                                           right=_east, top=_north)
            _img_l93 = np.zeros((_dh, _dw, _nb), dtype=_img_wm.dtype)
            for _b in range(_nb):
                _rio_proj(_img_wm[:, :, _b], _img_l93[:, :, _b],
                          src_transform=_src_tf, src_crs=_src_crs,
                          dst_transform=_dst_tf, dst_crs=_dst_crs,
                          resampling=Resampling.bilinear)
            _el = _dst_tf.c; _et = _dst_tf.f
            _er = _el + _dst_tf.a * _dw; _eb = _et + _dst_tf.e * _dh
            ax.imshow(_img_l93, extent=[_el, _er, _eb, _et],
                      zorder=0, alpha=0.85, aspect="auto")
            fond_ok = True
            _ts("IGN reprojete et affiche")
            print("OK Fond IGN charge")
        except Exception as e:
            print("WARN Fond IGN indisponible ({}) - fond neutre".format(type(e).__name__))

    if not fond_ok:
        ax.set_facecolor("#f0ede8")
        ax.grid(True, linestyle=":", linewidth=0.5, color="#aaaaaa", alpha=0.6)

    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_autoscale_on(False)

    # ── Zones PLU (Géoportail de l'Urbanisme — API Carto IGN) ────────────────
    gdf_plu     = None
    legende_plu = {}      # clé tz -> couleur (pour légende, sans doublons)
    rnu_detecte = False

    if urba_terrain or urba_buffer:
        if urba_buffer:
            bx0, by0, bx1, by1 = buf600.bounds
        else:
            bx0, by0, bx1, by1 = terrain.bounds
        _ts("GPU/PLU fetch debut")
        gdf_plu = charger_zones_urbanisme(bx0, by0, bx1, by1)
        _ts("GPU/PLU fetch termine ({} zones)".format(len(gdf_plu) if gdf_plu is not None else 0))

    # Déterminer la géométrie de clipping une seule fois
    if urba_terrain or urba_buffer:
        if urba_terrain and urba_buffer:
            clip_geom = buf600
        elif urba_terrain:
            clip_geom = terrain
        else:
            clip_geom = buf600
    else:
        clip_geom = None

    if gdf_plu is None and clip_geom is not None:
        # ── API indisponible : pas de couleur, pas de hachure ───────────────
        # Conformément à la demande : "pas d'info = pas de couleur"
        # On laisse la zone sans remplissage, juste la note en légende
        pass   # rien à tracer, la note "Sans couleur = info non disponible" suffira

    if gdf_plu is not None and clip_geom is not None:
        if len(gdf_plu) == 0:
            # ── Commune sous RNU : hachure grise ────────────────────────────
            rnu_detecte = True
            draw_hatch(ax, clip_geom,
                       ec="#999999", fc="#CCCCCC", hatch="////",
                       alpha_fill=0.25, lw=0.8, zorder=1)
            rnu_pt = clip_geom.representative_point()
            ax.text(rnu_pt.x, rnu_pt.y, "RNU",
                    fontsize=13, fontweight="bold", color="#555555",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.45", fc="white",
                              alpha=0.90, ec="#999999", lw=1.2),
                    zorder=20)

        elif len(gdf_plu) > 0:
            # ── PLU numerise : dessiner chaque zone ─────────────────────────
            col_type = "typezone" if "typezone" in gdf_plu.columns else None
            col_lib  = "libelle"  if "libelle"  in gdf_plu.columns else None
            SEUIL_LABEL_M2 = 5000   # zone < 5 000 m² = pas de label

            for _, row in gdf_plu.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                try:
                    geom = geom.intersection(clip_geom)
                except Exception:
                    continue
                if geom is None or geom.is_empty:
                    continue

                # Determiner la categorie (U / AU / A / N)
                # type_zone() normalise AUs->AU, UB->U, etc.
                tz = None
                if col_type:
                    raw = str(row[col_type]).strip() if row[col_type] else ""
                    tz = type_zone(raw)
                if tz is None and col_lib:
                    raw = str(row[col_lib]).strip() if row[col_lib] else ""
                    tz = type_zone(raw)

                if tz is not None:
                    couleur = COULEURS_PLU[tz]
                    draw_geom(ax, geom, fc=couleur["fc"], ec=couleur["ec"],
                              lw=0.8, alpha_fill=0.30, ls="-", zorder=1)
                    if tz not in legende_plu:
                        legende_plu[tz] = couleur
                else:
                    # Zone sans categorie reconnue : bord gris, aucun remplissage
                    draw_geom(ax, geom, fc="none", ec="#AAAAAA",
                              lw=0.5, alpha_fill=0, ls="--", zorder=1)
                    legende_plu["?"] = None   # signale la presence de zones inconnues

                # ── Label libelle court (ex : Ns, AUx, A) ──────────────────
                lbl_txt = ""
                if col_lib:
                    v = str(row[col_lib]).strip() if row[col_lib] else ""
                    if v and v.lower() not in ("none", "nan"):
                        lbl_txt = v
                if not lbl_txt and col_type:
                    v = str(row[col_type]).strip() if row[col_type] else ""
                    if v and v.lower() not in ("none", "nan"):
                        lbl_txt = v

                if lbl_txt and geom.area >= SEUIL_LABEL_M2:
                    try:
                        rp = geom.representative_point()
                        rx, ry = float(rp.x), float(rp.y)
                    except Exception:
                        rx = None
                    if rx is not None and x0 <= rx <= x1 and y0 <= ry <= y1:
                        _M = echelle * 0.0254 / 72
                        _pts_pos = (
                            [(pt["x"], pt["y"]) for pt in extremal] +
                            [(pt["x"] + (pt["ann_dx"] + 46) * _M,
                              pt["y"] + (pt["ann_dy"] + 26) * _M) for pt in extremal]
                        )
                        _trop_proche = any(
                            ((rx - px)**2 + (ry - py)**2) < (geo_w * 0.07)**2
                            for px, py in _pts_pos
                        )
                        if not _trop_proche:
                            txt_color = (COULEURS_PLU[tz]["ec"]
                                         if tz else "#888888")
                            ax.text(rx, ry, lbl_txt,
                                    fontsize=8, fontweight="bold",
                                    color=txt_color,
                                    ha="center", va="center",
                                    bbox=dict(boxstyle="round,pad=0.2",
                                              fc="white", alpha=0.80, ec="none"),
                                    zorder=20, clip_on=True)


    # ── Zones non couvertes par le PLU = RNU ou hors périmètre GPU ─────────
    # On calcule la différence entre l'emprise clippée et l'union des zones PLU tracées
    if clip_geom is not None and gdf_plu is not None and len(gdf_plu) > 0:
        try:
            zones_plu_tracees = []
            for _, row in gdf_plu.iterrows():
                g = row.geometry
                if g is None or g.is_empty:
                    continue
                try:
                    g = g.intersection(clip_geom)
                except Exception:
                    continue
                if g and not g.is_empty:
                    zones_plu_tracees.append(g)

            if zones_plu_tracees:
                couverture_plu = unary_union(zones_plu_tracees)
                zone_non_couverte = clip_geom.difference(couverture_plu)
            else:
                zone_non_couverte = clip_geom

            # N'afficher le gris RNU que si la zone non couverte est significative
            SEUIL_RNU_M2 = 10000   # 1 ha minimum pour afficher
            if not zone_non_couverte.is_empty and zone_non_couverte.area > SEUIL_RNU_M2:
                rnu_detecte = True
                draw_hatch(ax, zone_non_couverte,
                           ec="#999999", fc="#CCCCCC", hatch="////",
                           alpha_fill=0.25, lw=0.8, zorder=1)
                # Label RNU centré sur la zone non couverte
                rnu_pt = zone_non_couverte.representative_point()
                rx, ry = float(rnu_pt.x), float(rnu_pt.y)
                if x0 <= rx <= x1 and y0 <= ry <= y1:
                    ax.text(rx, ry, "RNU",
                            fontsize=11, fontweight="bold", color="#555555",
                            ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.40", fc="white",
                                      alpha=0.90, ec="#999999", lw=1.0),
                            zorder=20)
        except Exception as e:
            print("Calcul zone RNU échoué : {}".format(e))
        _ts("PLU rendu + calcul RNU termine")

        # ── Couches de base ───────────────────────────────────────────────────────
    # Périmètre 600m — halo sombre + trait blanc pour lisibilité sur fond aérien
    draw_geom(ax, buf600, fc="none", ec="#333333", lw=3.5, alpha_fill=0, ls=(0,(4,5)), zorder=2)
    draw_geom(ax, buf600, fc="none", ec="#FFFFFF", lw=1.8, alpha_fill=0, ls=(0,(4,5)), zorder=2)
    draw_geom(ax, terrain,  fc="none",    ec="#CC0000", lw=2.5, alpha_fill=0,    ls="-",       zorder=10)

    # ── Zone capteurs : tirets-points bleu roi, 1 tracé par cluster
    _ZC_LS = (0, (6, 2, 1, 2))
    if capteurs_clusters:
        for cluster in capteurs_clusters:
            if cluster is None or cluster.is_empty:
                continue
            draw_geom(ax, cluster, fc="none", ec="#000000", lw=2.8, alpha_fill=0, ls=_ZC_LS, zorder=5)
            draw_geom(ax, cluster, fc="none", ec="#1A6FBF", lw=1.5, alpha_fill=0, ls=_ZC_LS, zorder=5)
    else:
        # Aucun KML : zone capteurs standard (buffer négatif terrain)
        draw_geom(ax, capteurs, fc="none", ec="#000000", lw=2.8, alpha_fill=0, ls=_ZC_LS, zorder=5)
        draw_geom(ax, capteurs, fc="none", ec="#1A6FBF", lw=1.5, alpha_fill=0, ls=_ZC_LS, zorder=5)

    # ── Zones humides (si présentes) ──────────────────────────────────────────
    if zh_geom is not None:
        draw_hatch(ax, zh_geom,
                   ec="#005B9F",    # bleu foncé contour
                   fc="#AEE4FF",    # bleu clair remplissage
                   hatch="///",
                   alpha_fill=0.20,
                   lw=0.8,
                   zorder=2)

    # ── Éléments techniques (si présents) — formes des KML ───────────────────
    # Rendu unifié orange : panneaux (lignes+polys) + pistes (lignes+points)
    if gdf_elts is not None and len(gdf_elts) > 0:
        # Simplification vectorisée avant la boucle (plus rapide que par géométrie)
        # 1 m sans perte visuelle à 1/5000 (1 m = 0.2 mm sur la carte)
        gdf_elts = gdf_elts.copy()
        gdf_elts["geometry"] = gdf_elts.geometry.simplify(1.0, preserve_topology=True)
        for geom in gdf_elts.geometry:
            if geom is None or geom.is_empty:
                continue
            gtype = geom.geom_type
            if gtype in ("Polygon", "MultiPolygon"):
                draw_geom(ax, geom, fc="#E8A020", ec="#d94701",
                          lw=0.7, alpha_fill=0.35, ls="-", zorder=7)
            elif gtype in ("LineString", "MultiLineString"):
                draw_geom(ax, geom, fc="none", ec="#d94701",
                          lw=0.8, alpha_fill=0, ls="-", zorder=7)
            elif gtype in ("Point", "MultiPoint"):
                pts_list = [geom] if gtype == "Point" else list(geom.geoms)
                for pt in pts_list:
                    # coords[0] peut être (x, y) ou (x, y, z) — on prend juste x,y
                    c = pt.coords[0]
                    ax.plot(c[0], c[1], "s", color="#d94701", markersize=5,
                            zorder=7, markeredgecolor="#B85000", markeredgewidth=0.5)

    _ts("Tracé géométries (terrain/capteurs/ZH/elts) termine")
    # ── Points extrémaux ──────────────────────────────────────────────────────
    c_txt = "white" if fond_ok else "#222"
    for pt in extremal:
        ax.plot(pt["x"], pt["y"], "o", color="#990000", markersize=8,
                zorder=9, markeredgecolor="white", markeredgewidth=1.2)
        ax.annotate("{}\n{}\n{}".format(pt["label"], pt["lat"], pt["lon"]),
                    xy=(pt["x"], pt["y"]),
                    xytext=(pt["ann_dx"], pt["ann_dy"]),
                    textcoords="offset points",
                    fontsize=7.5, color="#660000", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="#990000", lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              alpha=0.92, ec="#990000", lw=0.8), zorder=15)

    # ── Barre d'échelle ───────────────────────────────────────────────────────
    sb_x = x1 - geo_w * 0.04 - 500
    sb_y = y0 + geo_h * 0.035
    ax.annotate("", xy=(sb_x + 500, sb_y), xytext=(sb_x, sb_y),
                arrowprops=dict(arrowstyle="|-|, widthA=0.5, widthB=0.5", color=c_txt, lw=2))
    ax.text(sb_x + 250, sb_y + geo_h * 0.013, "500 m", ha="center",
            fontsize=9, fontweight="bold", color=c_txt,
            bbox=dict(fc="#00000066" if fond_ok else "white", alpha=0.7, ec="none"))

    # ── Flèche Nord ───────────────────────────────────────────────────────────
    arr_h = geo_h * 0.07
    nx = x0 + geo_w * 0.07
    ny = y1 - geo_h * 0.18
    ax.annotate("", xy=(nx, ny + arr_h), xytext=(nx, ny),
                arrowprops=dict(arrowstyle="-|>", color=c_txt, lw=4.5))
    ax.text(nx, ny + arr_h * 1.18, "N", ha="center", fontsize=14,
            fontweight="bold", color=c_txt)

    # ── Légende ───────────────────────────────────────────────────────────────
    legend_items = [
        mlines.Line2D([], [], color="#CC0000", linewidth=2.5,
                     label="Terrain d'implantation"),
    ]
    legend_items.append(
        mlines.Line2D([], [], color="#1A6FBF", linewidth=1.5, linestyle=(0,(6,2,1,2)),
                      label="Zone d'implantation des capteurs")
    )
    legend_items += [
        mlines.Line2D([], [], color="#AAAAAA", linewidth=2.0, linestyle=(0,(4,5)),
                     label="P\u00e9rim\u00e8tre de 600 m"),
        mlines.Line2D([], [], color="#990000", marker="o", linestyle="None",
                      markersize=8, label="Points de coordonn\u00e9es WGS84"),
    ]

    # Entrées légende conditionnelles ZH
    if zh_geom is not None:
        legend_items.append(
            mpatches.Patch(facecolor="#AEE4FF", edgecolor="#005B9F",
                           alpha=0.6, linewidth=1.8, hatch="////",
                           label="Zone(s) humide(s)")
        )

    # Entrées légende conditionnelles éléments techniques
    if gdf_elts is not None and len(gdf_elts) > 0:
        legend_items.append(
            mlines.Line2D([], [], color="#d94701", linewidth=1.0,
                          label="\u00c9l\u00e9ments techniques centrale PV")
        )
    # Zones PLU — RNU en premier si détecté, puis une entrée par catégorie
    if rnu_detecte:
        legend_items.append(
            mpatches.Patch(facecolor="#CCCCCC", edgecolor="#999999",
                           alpha=0.55, linewidth=0.8, hatch="////",
                           label="Commune sous RNU")
        )
    for tz, couleur in legende_plu.items():
        if tz == "?" or couleur is None:
            continue  # zones inconnues gérées par la note
        lbl = COULEURS_PLU[tz]["label"] if tz in COULEURS_PLU else "Zone urbanisme"
        legend_items.append(
            mpatches.Patch(facecolor=couleur["fc"], edgecolor=couleur["ec"],
                           alpha=0.4, linewidth=0.5, label=lbl)
        )
    # Note en bas si PLU demandé (disclaimer données GPU)
    _NOTE = "Sans couleur = info non disponible (Géoportail de l'Urbanisme)"
    _show_note = (urba_terrain or urba_buffer)
    if _show_note:
        legend_items.append(
            mlines.Line2D([], [], color="none", linewidth=0, label=_NOTE)
        )

    leg = ax.legend(handles=legend_items, loc="lower left",
                    fontsize=9, framealpha=0.97, edgecolor="#cccccc",
                    bbox_to_anchor=(0.01, 0.01), bbox_transform=ax.transAxes)
    leg.set_zorder(25)

    # Mise en forme de la note (italique gris, handle invisible)
    if _show_note:
        for txt in leg.get_texts():
            if txt.get_text() == _NOTE:
                txt.set_color("#888888")
                txt.set_style("italic")
                txt.set_fontsize(7.5)
        try:
            handles = leg.legend_handles
        except AttributeError:
            handles = leg.legendHandles
        for handle, txt in zip(handles, leg.get_texts()):
            if txt.get_text() == _NOTE:
                handle.set_visible(False)

    # ── Encart urbanisme ──────────────────────────────────────────────────────
    if urbanisme.strip():
        ax.text(x1 - geo_w * 0.01, y1 - geo_h * 0.01,
                "Document d'urbanisme applicable\nau terrain d'implantation\n{}\n{}".format("\u2500" * 10, urbanisme),
                ha="right", va="top", fontsize=10, weight="bold", linespacing=1.7,
                bbox=dict(boxstyle="round,pad=0.6", fc="white",
                          alpha=0.97, ec="#cccccc", lw=0.8), zorder=25)

    # ── Axes WGS84 ────────────────────────────────────────────────────────────
    _tr_wgs = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    _tr_l93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    _cx_mid, _cy_mid = (x0 + x1) / 2, (y0 + y1) / 2
    _lon_mid, _lat_mid = _tr_wgs.transform(_cx_mid, _cy_mid)

    def _fmt_lon(v, _):
        lon, _ = _tr_wgs.transform(v, _cy_mid)
        return "{:.3f}\u00b0 {}".format(abs(lon), "E" if lon >= 0 else "O")

    def _fmt_lat(v, _):
        _, lat = _tr_wgs.transform(_cx_mid, v)
        return "{:.3f}\u00b0 {}".format(abs(lat), "N" if lat >= 0 else "S")

    lon0, lat0 = _tr_wgs.transform(x0, y0)
    lon1, lat1 = _tr_wgs.transform(x1, y1)
    lon_ticks = np.arange(math.ceil(lon0 / tick_deg) * tick_deg,
                          math.floor(lon1 / tick_deg) * tick_deg + tick_deg / 2, tick_deg)
    lat_ticks = np.arange(math.ceil(lat0 / tick_deg) * tick_deg,
                          math.floor(lat1 / tick_deg) * tick_deg + tick_deg / 2, tick_deg)
    ax.set_xticks([_tr_l93.transform(lon, _lat_mid)[0] for lon in lon_ticks])
    ax.set_yticks([_tr_l93.transform(_lon_mid, lat)[1] for lat in lat_ticks])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_lon))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_lat))
    ax.set_xlabel("Longitude (WGS84)", fontsize=9)
    ax.set_ylabel("Latitude (WGS84)", fontsize=9)
    ax.tick_params(labelsize=8)

    # ── Titre ─────────────────────────────────────────────────────────────────
    src = "\u00a9 IGN G\u00e9oportail" if fond_ok else "fond neutre"
    ax.set_title(
        "UNITe PV \u2014 {}\nPlan de situation  |  \u00c9chelle\u202f: {}  |  {}".format(
            nom_projet, echelle_lbl, src),
        fontsize=15, fontweight="bold", pad=16)

    # ── Logo UNITe (haut-droite, meme hauteur que le titre) ───────────────────
    _LOGO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo_unite.png")
    if os.path.exists(_LOGO):
        from PIL import Image as _PILImg
        _logo = _PILImg.open(_LOGO).convert("RGBA")
        _target_h = max(int(MARGIN_TOP * dpi * 0.72), 10)
        _target_w = int(_target_h * _logo.width / _logo.height)
        _logo = _logo.resize((_target_w, _target_h), _PILImg.LANCZOS)
        _logo_arr = np.array(_logo)
        # yo est TOUJOURS compte depuis le BAS du canvas, meme avec origin="upper" :
        # origin ne controle que le sens de lecture du tableau de pixels.
        # On vise le centre vertical de la bande de marge haute (celle du titre) :
        # on part du haut de l'axe cartographique, puis on centre le logo dedans.
        _axes_top_px = int((MARGIN_BOT + ax_h_in) * dpi)
        _yo = _axes_top_px + (int(MARGIN_TOP * dpi) - _target_h) // 2
        _xo = int(fig_w_in * dpi) - _target_w - int(0.10 * dpi)
        fig.figimage(_logo_arr, xo=_xo, yo=_yo, origin="upper")

    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    _ts("Annotations/legende/titre terminees")

    # ── Export bytes ──────────────────────────────────────────────────────────
    fmt = format.lower().strip()
    if fmt not in ("png", "pdf"):
        raise ValueError("format doit etre 'png' ou 'pdf', recu : {!r}".format(fmt))
    buf = io.BytesIO()
    _ts("savefig debut ({}, dpi={})".format(fmt, dpi if fmt == "png" else 300))
    if fmt == "pdf":
        # convert("RGB") est LE point critique : en RGBA, PIL monte a 2940 Mo
        # contre 690 Mo en RGB a dpi identique. Ne jamais retirer cette conversion.
        # dpi 200 = 0,63 m/pixel au sol a 1/5000, pic mesure 347 Mo.
        # Garde-fou : on plafonne le nombre total de pixels pour les tres grands
        # sites, sinon la memoire croit avec le carre de l'emprise.
        from PIL import Image as _PIL
        MAX_MPX  = 24e6
        _dpi_pdf = int(min(200, (MAX_MPX / (fig_w_in * fig_h_in)) ** 0.5))
        _buf_png = io.BytesIO()
        plt.savefig(_buf_png, dpi=_dpi_pdf, facecolor="white", format="png")
        plt.close()
        _buf_png.seek(0)
        _img = _PIL.open(_buf_png).convert("RGB")
        _img.save(buf, format="PDF", resolution=_dpi_pdf)
        _img.close()
        _buf_png.close()
    else:
        plt.savefig(buf, dpi=dpi, facecolor="white", format="png")
        plt.close()
    buf.seek(0)
    _ts("savefig termine")

    print("Surface : {:.2f} ha  |  \u00c9chelle : {}".format(
        terrain.area / 10000, echelle_lbl))
    for pt in extremal:
        print("  {}  {}   {}".format(pt["label"], pt["lat"], pt["lon"]))

    return buf.read()


# ════════════════════════════════════════════════════════════════
# LANCEMENT DIRECT (terminal)
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    png_bytes = generer_carte(
        shp_path=SHP_PATH, nom_projet=NOM_PROJET,
        recul_capteurs=RECUL_CAPTEURS, urbanisme=URBANISME,
        echelle=ECHELLE, fond_aerien=FOND_AERIEN,
        dpi=DPI, buffer_carte=BUFFER_CARTE,
        zh_path=ZH_PATH, elements_path=ELEMENTS_PATH,
    )
    _slug = re.sub(r"[^\w]+", "_", NOM_PROJET).strip("_")
    output_png = os.path.join(OUTPUT_DIR, "UNITe_CETI_PV_{}.png".format(_slug))
    with open(output_png, "wb") as f:
        f.write(png_bytes)
    print("\n\u2705 Carte : {}".format(output_png))