# StructuralAD
 Structural Knowledge-Based Anomaly Detection to Inspect Ball-Based Lens Actuators

![Image](https://github.com/user-attachments/assets/aac976a6-5e79-43c6-9be8-ab326f7f973a)

[https://doi.org/10.1109/ACCESS.2025.3622686](https://doi.org/10.1109/ACCESS.2025.3622686)

IEEE Access 2025, 13, 184110 <br>
ISSN:2169-3536      
eISSN:2169-3536

## Repo Structure

The repository is organized as follows:
- `custom/`: custom defined module
    - `detector`: proposal Structural AD model
- `datasets/`: dataset folder
    - `BALL`: Structural AD datasets; https://doi.org/10.21227/mysd-9b62
       - `ball`: default datasets
       - `enhanced_ball`: Augmented dataset
- `results/`: results storage folder
- `anomaly_detection.ipynb`: training and testing all models
- `img_enhancing.ipynb`: Create enhanced image datasets `enhanced_ball` for `detector`


## BALL Dataset

Image of faulty modules target to ball missing inspection collected at the optical cross-section inspection site: 

| Dataset  | Task  | Size  | Normal  | Anomalies  | 
|-------|--------|--------|--------|--------|
| BaLL | Anomaly Detection | 2352 x 2944 | 10,183 | 1,755 |


## Example

```bash
from custom.data import Ball, MVTec
from custom.models import Patchcore, EfficientAd, Detector
from custom.engine import Engine

options = {}
method = 'Detector' # EfficientAd Patchcore EfficientAd Dinomaly
if method != 'Patchcore':
    options['train_batch_size'] = 1

category = 'ball' # enhanced_ball
if 'ball' in category:
    options['category'] = category

datamodule = Ball(**options) if category == 'ball' or category == 'enhanced_ball' else MVTec(**options)
model = Patchcore() if method == 'Patchcore' else EfficientAd() if method == 'EfficientAd' else Detector

engine = None
if method == 'patchcore':
    engine = Engine()
elif method == 'Detector':
    engine = Engine(is_detector=True)
elif method == 'EfficientAd':
    engine = Engine(max_epochs=10)

engine.fit(datamodule=datamodule, model=model)

if 'ball' in category:
    datasets_name = 'Ball'
else:
    datasets_name = 'Mvtec'
    
predictions = engine.predict(
    datamodule=datamodule,
    model=model,
    ckpt_path=f'./results/{method}/{datasets_name}/{category}/lastest/weights/lightning/model.ckpt',
    return_predictions=True,
)

engine.report()
```

\* To run the code, you need to have **Python 3.10**. 

## Citation
If you use our code, please cite the paper below:
```bibtex
@article{jeon2025structural,
  title={Structural Knowledge-based Anomaly Detection to inspect Ball-based Lens Actuators},
  author={Jeon, Janggun and Ahn, Junho and Kim, Namgi},
  journal={IEEE Access},
  volume={13},
  pages={184110},
  year={2025},
  publisher={IEEE}
}
```
