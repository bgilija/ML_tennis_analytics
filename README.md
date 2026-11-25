# Tennis analysis

Analiza teniskog snimka (statična TV kamera): detekcija igrača i loptice, tačke terena,
mini mapa, udarci, odskoci i statistika (brzina udarca, brzina i pređeni put igrača).

## Instalacija

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Modeli (nisu u git-u):

| fajl | opis |
|---|---|
| `models/yolo5_last.pt` | YOLOv5 detektor loptice (`training/tennis_ball_detector_training.ipynb`) |
| `models/keypoints_model.pth` | ResNet50, 14 tačaka terena (`training/tennis_court_keypoints_training.ipynb`) |
| `yolov8x.pt` | COCO YOLOv8x za igrače, ultralytics ga sam preuzme |

## Pokretanje

```
python main.py --input input_videos/input_video.avi
python main.py --input input_videos/ex04.avi --use-stubs   # koristi keširane detekcije
```

Izlaz: `output_videos/<ime>_output.avi`. Detekcije se keširaju u `tracker_stubs/<ime>_*.pkl`.
Ulaz mora biti `.avi`.

## Pipeline

1. `CourtLineDetector.predict_video` – 14 tačaka terena (medijana preko više frejmova).
2. `utils/court_reference.py` – homografija slika → metri terena (svih 14 tačaka, RANSAC).
3. `PlayerTracker` – YOLOv8 + tracking; dva igrača sa stabilnim ID-jem: 1 = dalji, 2 = bliži.
4. `BallTracker` – izbor kandidata po kontinuitetu putanje, uklanjanje outliera, interpolacija kratkih praznina.
5. `analysis/events.py` – udarci (promena smera loptice kod igrača) i odskoci (skok u ubrzanju y).
6. `analysis/stats.py` – brzine udaraca, brzina i pređeni put igrača, panel sa statistikom.
7. `MiniCourt` – mini mapa u metrima.

## Testovi

```
python -m unittest discover tests
```

Testovi ne koriste modele (sintetički podaci): homografija, izbor loptice, outlieri,
interpolacija, izbor igrača.

## Poznata ograničenja

- Kamera mora biti statična (tačke terena se računaju jednom za ceo snimak).
- Odskok se traži kao skok u ubrzanju putanje loptice; ako model ne vidi lopticu baš oko
  odskoka, pozicija može da kasni/žuri par frejmova.
- Kod servisa daljeg igrača vrh bacanja loptice može biti prepoznat kao udarac.
- Brzina udarca je prosečna brzina od igrača do odskoka, ne brzina na reketu.
