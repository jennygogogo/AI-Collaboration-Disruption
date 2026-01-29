# AI-Collaboration-Disruption

The codebase is organized into two main components:
1.  **Disruption Prediction Model** 
2.  **Implementing the VirSci system** 

---

## 1. Disruption Prediction Model

Located in `prediction_model/`.

### Usage
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