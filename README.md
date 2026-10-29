# Car Damage Inspector

Upload photos of a car from different angles. The app finds which parts are damaged, what kind of damage it is (dent, scratch, crack, shattered glass, broken lamp, flat tire), how bad it is, and produces a repair report with a rough cost estimate.

It is a visual damage tool. It cannot find mechanical or internal faults from photos.

## How it works

```
photos -> parts segmentation (YOLO) ----\
                                         +-> overlay: damage mask inside part mask
photos -> damage segmentation (YOLO) ---/        |
                                                 v
                          severity (damage area / part area, weighted by damage type)
                                                 |
                                                 v
                 merge findings across photos -> repair action + cost range -> PDF + DB
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Get data and train

You need two datasets in YOLO segmentation format (Roboflow can export this directly, and gives you a `data.yaml`):

- Damage: CarDD (dent, scratch, crack, glass shatter, lamp broken, tire flat), or a "car damage" set from Roboflow Universe or Kaggle.
- Parts: a "car parts segmentation" set from Roboflow Universe.

Check each dataset's license before using it in anything public.

```bash
python train.py --task damage --data datasets/damage/data.yaml --epochs 100
python train.py --task parts  --data datasets/parts/data.yaml  --epochs 100
```

Use `--device 0` for a GPU. Google Colab's free GPU is fine for this. The best weights are copied to `models/damage.pt` and `models/parts.pt`, and the mAP scores are printed. Put those numbers in your resume and README.

Class names in `app/config.py` (severity weights, repair table) must match your dataset's labels. Edit them if your dataset names things differently.

## 2. Run the demo UI

```bash
streamlit run streamlit_app.py
```

## 3. Or run the API

```bash
uvicorn app.api:app --reload
```

| Method | Path | What it does |
| --- | --- | --- |
| POST | `/inspections` | multipart upload (`files`, `car_label`), returns the report |
| GET | `/inspections` | list past inspections |
| GET | `/inspections/{id}` | one inspection with findings |
| PATCH | `/inspections/{id}/status` | `pending`, `in repair` or `repaired` |
| GET | `/inspections/{id}/report.pdf` | PDF report |

Docs are at http://localhost:8000/docs.

## Tests

```bash
pytest
```

These cover the matching, severity, report, PDF and database logic using fake detections, so they run without trained weights.

## Project layout

```
app/
  config.py      thresholds, severity weights, repair and cost tables
  detector.py    loads both YOLO models, turns results into masks
  matching.py    damage-to-part matching, severity, multi-photo merge
  pipeline.py    runs everything on a set of photos, draws annotated images
  report.py      repair actions, cost ranges, PDF export
  db.py          SQLite storage (SQLAlchemy)
  api.py         FastAPI service
streamlit_app.py demo UI
train.py         training script
tests/           logic tests
```

## Known limits

- Reflections, shadows, dirt and water cause false positives. Add failure cases to this README once you have trained models.
- Costs come from a small rule table in `config.py`. They are estimates, not quotes. Replace them with real local pricing if you have it.
- Two separate dents on the same part are merged into one entry (the worst one is kept).
- Severity is based on 2D area in the photo, so camera angle and distance affect it.

## Ideas to extend

- Before/after comparison to confirm a repair was done.
- Track a car's damage history across inspections.
- Export models to ONNX for faster CPU inference.
- Evaluate severity against labels from a real body shop.
