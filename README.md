# AI-Collaboration-Disruption

The codebase is organized into two main components:
1.  **Disruption Prediction Model** 
2.  **Implementing the VirSci system** 

---

## 1. Disruption Prediction Model

Located in `prediction_model/`.

### Usage
1. Download the data from [Google Drive](https://drive.google.com/drive/folders/1AJlLzaL5xVNyx3KO2dElczEYIseFtOiE?usp=sharing).
2. Run training
```bash
cd prediction_model
bash train.sh
```

---

## 2. VirSci Simulation System
Located in Virtual-Scientists-v2-main/.

#### Baseline Virsci System
```bash
python run_fast.py 2>&1 | tee virsci_output.txt
```

#### VirSci w/o Social Attributes
```bash
python run_fast_no_social.py 2>&1 | tee virsci_no_social_output.txt
```

#### VirSci w/o Scientific Norms
```bash
python run_fast_no_real.py 2>&1 | tee virsci_no_real_output.txt
```

## Acknowledgements
This project is built upon [Virtual-Scientists-v2](https://github.com/RenqiChen/Virtual-Scientists-v2).

We utilize the [SciSciNet-v2](https://github.com/kellogg-cssi/SciSciNet) dataset for training our disruption prediction model.

## License
This repository is licensed under the Apache-2.0 License.