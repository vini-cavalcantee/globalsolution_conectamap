"""Preparação ÚNICA das geometrias municipais do dashboard (não roda durante o uso do app).

Fonte: Malha Municipal 2022 do IBGE, na versão simplificada distribuída pelo pacote geobr (IPEA), release
v1.7.0 no GitHub (um GeoPackage por UF, SIRGAS 2000 / EPSG:4674). O geobr padroniza a malha oficial do IBGE;
os códigos municipais são os códigos IBGE de 7 dígitos.

Procedimento:
1. baixa (se necessário) os 27 arquivos `<código UF>municipality_2022_simplified.gpkg` para uma pasta temporária;
2. lê cada GeoPackage com sqlite3 (sem GDAL/geopandas) e converte a geometria com shapely;
3. simplifica cada município (tolerância TOLERANCIA graus, preservando a topologia do próprio polígono)
   e arredonda as coordenadas em CASAS_DECIMAIS casas;
4. grava `assets/municipios_2022.geojson` (id = código IBGE) e `assets/ufs_2022.geojson` (contorno das UFs,
   pela união dos municípios), além de `assets/geometrias_metadados.json` com a origem e os hashes.

Uso:  python dashboard/preparar_geometrias.py [pasta_com_os_gpkg]
Dependência extra só para esta etapa: shapely (o app não precisa dela).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import sys
import urllib.request
from datetime import date
from pathlib import Path

URL_BASE = "https://github.com/ipeaGIT/geobr/releases/download/v1.7.0/"
# áreas da malha que não são municípios (espelhos d'água da Lagoa Mirim e da Lagoa dos Patos, RS)
NAO_MUNICIPIOS = {"4300001", "4300002"}
CODIGOS_UF = [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 35, 41, 42, 43, 50, 51, 52, 53]
TOLERANCIA = 0.01           # graus (~1 km): mantém a forma dos municípios pequenos com arquivo leve
CASAS_DECIMAIS = 3          # ~110 m, abaixo da tolerância de simplificação
ASSETS = Path(__file__).resolve().parent / "assets"


def _wkb_do_gpkg(blob: bytes) -> bytes:
    """Remove o cabeçalho GeoPackage (magic 'GP', versão, flags, srs_id, envelope) e devolve o WKB."""
    if blob[:2] != b"GP":
        raise ValueError("blob não é uma geometria GeoPackage")
    flags = blob[3]
    tam_envelope = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 0b111]
    return blob[8 + tam_envelope:]


def baixar(pasta: Path) -> list[Path]:
    pasta.mkdir(parents=True, exist_ok=True)
    arquivos = []
    for uf in CODIGOS_UF:
        nome = f"{uf}municipality_2022_simplified.gpkg"
        destino = pasta / nome
        if not destino.exists():
            urllib.request.urlretrieve(URL_BASE + nome, destino)
        arquivos.append(destino)
    return arquivos


def _arredondar(coords):
    if isinstance(coords[0], (float, int)):
        return [round(coords[0], CASAS_DECIMAIS), round(coords[1], CASAS_DECIMAIS)]
    return [_arredondar(c) for c in coords]


def preparar(pasta_gpkg: Path):
    import shapely
    from shapely.geometry import mapping

    feicoes, por_uf, origem = [], {}, {}
    for arq in baixar(pasta_gpkg):
        origem[arq.name] = hashlib.sha256(arq.read_bytes()).hexdigest()
        con = sqlite3.connect(arq)
        tabela = con.execute("select table_name from gpkg_contents").fetchone()[0]
        srs = con.execute("select srs_id from gpkg_geometry_columns").fetchone()[0]
        assert srs == 4674, f"{arq.name}: SRS inesperado {srs}"
        for geom, cod, nome, uf, regiao in con.execute(
                f"select geom, code_muni, name_muni, abbrev_state, name_region from '{tabela}'"):
            g = shapely.from_wkb(_wkb_do_gpkg(geom))
            s = shapely.make_valid(shapely.simplify(g, TOLERANCIA, preserve_topology=True))
            if s.is_empty:
                s = g
            if s.geom_type == "GeometryCollection":
                s = shapely.MultiPolygon([p for p in s.geoms if p.geom_type in ("Polygon", "MultiPolygon")
                                          for p in (p.geoms if p.geom_type == "MultiPolygon" else [p])])
            codigo = f"{int(cod):07d}"
            if codigo in NAO_MUNICIPIOS:
                continue
            gj = mapping(s)
            feicoes.append({"type": "Feature", "id": codigo,
                            "properties": {"id_municipio": codigo, "nome": nome, "uf": uf, "regiao": regiao},
                            "geometry": {"type": gj["type"], "coordinates": _arredondar(json.loads(json.dumps(gj["coordinates"])))}})
            por_uf.setdefault(uf, []).append(g)
        con.close()

    ufs = []
    for uf, geoms in sorted(por_uf.items()):
        u = shapely.simplify(shapely.union_all(geoms), TOLERANCIA * 2.5, preserve_topology=True)
        gj = mapping(u)
        ufs.append({"type": "Feature", "id": uf, "properties": {"sigla_uf": uf},
                    "geometry": {"type": gj["type"], "coordinates": _arredondar(json.loads(json.dumps(gj["coordinates"])))}})

    ASSETS.mkdir(exist_ok=True)
    saidas = {"municipios_2022.geojson": {"type": "FeatureCollection", "features": sorted(feicoes, key=lambda f: f["id"])},
              "ufs_2022.geojson": {"type": "FeatureCollection", "features": ufs}}
    hashes = {}
    for nome, obj in saidas.items():
        texto = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        (ASSETS / nome).write_text(texto, encoding="utf-8")
        hashes[nome] = hashlib.sha256(texto.encode("utf-8")).hexdigest()
    meta = {
        "fonte": "IBGE — Malha Municipal 2022, versão simplificada distribuída pelo pacote geobr (IPEA)",
        "url_distribuicao": URL_BASE + "<código UF>municipality_2022_simplified.gpkg",
        "ano_malha": 2022,
        "sistema_referencia": "SIRGAS 2000 (EPSG:4674), coordenadas geográficas em graus",
        "preparacao": f"leitura com sqlite3 + shapely; simplificação por município com tolerância de {TOLERANCIA} grau "
                      f"(preserve_topology=True); coordenadas arredondadas em {CASAS_DECIMAIS} casas; contorno das UFs "
                      "pela união dos municípios (tolerância 2,5× maior)",
        "municipios": len(feicoes),
        "excluidos": "4300001 e 4300002 (Lagoa Mirim e Lagoa dos Patos): áreas de água da malha, sem município",
        "data_preparacao": date.today().isoformat(),
        "sha256_arquivos_origem": origem,
        "sha256_saidas": hashes,
    }
    (ASSETS / "geometrias_metadados.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


if __name__ == "__main__":
    pasta = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / ".cache" / "conectamap_geobr_2022"
    m = preparar(pasta)
    print(json.dumps({k: m[k] for k in ("municipios", "sha256_saidas")}, indent=1))
    for f in sorted(ASSETS.glob("*.geojson")):
        print(f.name, round(f.stat().st_size / 1e6, 2), "MB")
