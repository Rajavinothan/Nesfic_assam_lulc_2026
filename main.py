"""LandWatch Assam API: Earth Engine processing and tile rendering."""
import os
import ee
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
app = FastAPI(title="LandWatch Assam API")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])

# Placeholder Level-I classes. Replace with the exact NRSC codes agreed with ASSAC.
CLASSES = [(1, "Built-up", "#d1493f"), (2, "Agriculture", "#e6c35c"), (3, "Forest", "#2f6b3a"),
           (4, "Wasteland", "#b79f84"), (5, "Wetland", "#5db7a6"), (6, "Water", "#1f5fa8"),
           (7, "Other", "#8d6bb8")]
UNCERTAIN = "#9aa0a6"  # class 0: below the confidence threshold, sent to review

# Dataset registry: add a dataset here and it appears in the UI.
DATASETS = {
    "sentinel2": {"label": "Sentinel-2 (optical)", "collection": "COPERNICUS/S2_SR_HARMONIZED"},
    "sentinel1": {"label": "Sentinel-1 (radar)", "collection": "COPERNICUS/S1_GRD"},
    "fused": {"label": "Sentinel-2 + Sentinel-1 (fused)", "collection": "both"},
}
SEASONS = {  # Assam season windows for a given year
    "pre": lambda y: (f"{y}-03-01", f"{y}-06-01"),
    "mon": lambda y: (f"{y}-06-01", f"{y}-10-01"),
    "post": lambda y: (f"{y}-10-01", f"{y}-12-01"),
    "win": lambda y: (f"{y-1}-12-01", f"{y}-03-01"),
}
_ready = False


def init_ee():
    global _ready
    if _ready:
        return
    project, sa, key = os.environ.get("EE_PROJECT"), os.getenv("EE_SERVICE_ACCOUNT"), os.getenv("EE_KEY_FILE")
    if not project:
        raise HTTPException(500, "EE_PROJECT is not set. Copy backend/.env.example to backend/.env.")
    if sa and key:
        ee.Initialize(ee.ServiceAccountCredentials(sa, key), project=project)
    else:
        ee.Initialize(project=project)
    _ready = True


def district_geom(name: str):
    fc = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq("ADM1_NAME", "Assam")) \
        .filter(ee.Filter.eq("ADM2_NAME", name))
    if fc.size().getInfo() == 0:
        raise HTTPException(404, f"District not found: {name}")
    return fc.geometry()


def optical(geom, s, e):
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    col = (ee.ImageCollection(DATASETS["sentinel2"]["collection"]).filterBounds(geom).filterDate(s, e)
           .linkCollection(cs, ["cs_cdf"]).map(lambda i: i.updateMask(i.select("cs_cdf").gte(0.6))))
    i = col.median().divide(10000)
    return i.select(["B2", "B3", "B4", "B8", "B11", "B12"]).addBands([
        i.normalizedDifference(["B8", "B4"]).rename("NDVI"),
        i.normalizedDifference(["B3", "B11"]).rename("MNDWI"),
        i.normalizedDifference(["B11", "B8"]).rename("NDBI")])


def radar(geom, s, e):
    col = (ee.ImageCollection(DATASETS["sentinel1"]["collection"]).filterBounds(geom).filterDate(s, e)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    i = col.select(["VV", "VH"]).median()
    return i.addBands(i.select("VV").subtract(i.select("VH")).rename("VVminusVH"))


def build(dataset, geom, s, e):
    if dataset == "sentinel2":
        return optical(geom, s, e)
    if dataset == "sentinel1":
        return radar(geom, s, e)
    return optical(geom, s, e).addBands(radar(geom, s, e))


def season_stack(dataset, geom, year):
    imgs = []
    for name, win in SEASONS.items():
        img = build(dataset, geom, *win(year))
        imgs.append(img.rename(img.bandNames().map(lambda b: ee.String(name).cat("_").cat(ee.String(b)))))
    return ee.Image.cat(imgs).clip(geom)


def classify(stack, rf, thr):
    """Level-I class with confidence-adaptive fallback: low confidence becomes class 0 (review)."""
    label = stack.classify(rf).rename("class").toInt()
    p = stack.classify(rf.setOutputMode("MULTIPROBABILITY"))
    conf = p.arrayReduce(ee.Reducer.max(), [0]).arrayGet([0]).rename("confidence")
    return label.where(conf.lt(thr), 0), conf


def class_vis(img):
    return img.getMapId({"min": 0, "max": len(CLASSES), "palette": [UNCERTAIN] + [c[2] for c in CLASSES]})


def tiles(map_id):
    return map_id["tile_fetcher"].url_format


def bounds_of(geom):
    c = geom.bounds().coordinates().get(0).getInfo()
    lons, lats = [p[0] for p in c], [p[1] for p in c]
    return [[min(lats), min(lons)], [max(lats), max(lons)]]


class Preview(BaseModel):
    district: str
    dataset: str
    start: str
    end: str


class Analyze(BaseModel):
    district: str
    dataset: str
    t1_year: int
    t2_year: int
    conf_threshold: float = 0.6
    cva_threshold: float = 0.5  # change-vector magnitude on index bands
    scale: int = 30  # statistics scale in metres


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/datasets")
def datasets():
    return [{"id": k, "label": v["label"]} for k, v in DATASETS.items()]


@app.get("/api/districts")
def districts():
    init_ee()
    fc = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq("ADM1_NAME", "Assam"))
    return sorted(fc.aggregate_array("ADM2_NAME").getInfo())


@app.post("/api/preview")
def preview(req: Preview):
    """Render the chosen satellite data for a date range."""
    init_ee()
    if req.dataset not in DATASETS:
        raise HTTPException(400, "Unknown dataset")
    geom = district_geom(req.district)
    img = build(req.dataset, geom, req.start, req.end).clip(geom)
    vis = {"sentinel1": {"bands": ["VV", "VH", "VVminusVH"], "min": [-25, -30, 0], "max": [0, -5, 15]}}.get(
        req.dataset, {"bands": ["B4", "B3", "B2"], "min": 0, "max": 0.3})
    return {"layers": {f"{DATASETS[req.dataset]['label']} {req.start} to {req.end}": tiles(img.getMapId(vis))},
            "bounds": bounds_of(geom)}


@app.post("/api/analyze")
def analyze(req: Analyze):
    """Classify T1 and T2, detect change, and return tiles and the transition matrix."""
    init_ee()
    if req.dataset not in DATASETS:
        raise HTTPException(400, "Unknown dataset")
    asset = os.environ.get("TRAINING_ASSET")
    if not asset:
        raise HTTPException(500, "TRAINING_ASSET is not set. Provide labelled points first.")
    geom = district_geom(req.district)
    train_year = int(os.getenv("TRAIN_YEAR", "2021"))
    train_stack = season_stack(req.dataset, geom, train_year)
    samples = train_stack.sampleRegions(collection=ee.FeatureCollection(asset), properties=["class"],
                                        scale=10, tileScale=4)
    rf = ee.Classifier.smileRandomForest(200).train(samples, "class", train_stack.bandNames())

    s1, s2 = season_stack(req.dataset, geom, req.t1_year), season_stack(req.dataset, geom, req.t2_year)
    c1, _ = classify(s1, rf, req.conf_threshold)
    c2, _ = classify(s2, rf, req.conf_threshold)

    # Dual-evidence change: class differs, spectral change is large, both dates are confident.
    idx = ".*_(NDVI|MNDWI|NDBI)"
    mag = s2.select(idx).subtract(s1.select(idx)).pow(2).reduce(ee.Reducer.sum()).sqrt()
    changed = c1.neq(c2).And(c1.gt(0)).And(c2.gt(0)).And(mag.gt(req.cva_threshold)).selfMask()

    pair = c1.multiply(10).add(c2).rename("pair")
    stats = ee.Image.pixelArea().divide(10000).addBands(pair).reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName="pair"), geometry=geom,
        scale=req.scale, maxPixels=1e11, bestEffort=True, tileScale=8).getInfo()
    names = {0: "Uncertain", **{c[0]: c[1] for c in CLASSES}}
    matrix = [{"from": names.get(int(g["pair"]) // 10), "to": names.get(int(g["pair"]) % 10),
               "hectares": round(g["sum"], 1)} for g in stats.get("groups", [])]
    return {
        "layers": {f"Classes {req.t1_year}": tiles(class_vis(c1)), f"Classes {req.t2_year}": tiles(class_vis(c2)),
                   "Significant change": tiles(changed.getMapId({"palette": ["#ff0033"]}))},
        "bounds": bounds_of(geom), "matrix": matrix,
        "classes": [{"name": "Uncertain", "color": UNCERTAIN}] + [{"name": c[1], "color": c[2]} for c in CLASSES],
    }
