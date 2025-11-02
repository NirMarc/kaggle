# 🚀 Spaceship Titanic - Kaggle Competition

Predicting which passengers were transported to an alternate dimension during the Spaceship Titanic's collision with a spacetime anomaly.

**Competition:** [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)  
**Final Score:** 79.6% accuracy  
**Rank:** 982

---

## 📊 Project Overview

This project demonstrates a complete machine learning workflow:
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training & Evaluation
- Kaggle Submission

**Key Achievement:** Achieved 79.6% accuracy using XGBoost with minimal hyperparameter tuning, demonstrating the power of strong feature engineering.

---

## 🗂️ Project Structure
```
spaceship-titanic/
├── data/
│   ├── train.csv                    # Training data
│   ├── test.csv                     # Test data
│   └── sample_submission.csv
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory Data Analysis
│   └── 02_modeling.ipynb            # Model training & evaluation
├── submissions/
│   └── submission_xgboost.csv       # Final submission
├── donwload-data.sh                 # Script to download Kaggle data
├── requirements.txt                 # Python dependencies
├── utils.py                         # Preprocessing pipeline
└── README.md
```

---

## 🔍 Key Findings from EDA

### Strongest Predictive Features

| Feature | Signal Strength | Key Insight |
|---------|----------------|-------------|
| **Cabin_Deck** | 53% spread | Deck B: 73% transported, Deck T: 20% transported |
| **Is_Spender** | 49% spread | Non-spenders: 79% transported, Spenders: 30% transported |
| **CryoSleep** | 49% spread | CryoSleep passengers: 82% transported |
| **Age_Group** | 20% spread | Children: 67% transported, Young adults: 47% transported |
| **HomePlanet** | 21% spread | Europa: 64%, Earth: 43% transported |

### Feature Engineering Highlights

**Created Features:**
- `Is_Spender`: Binary indicator (spending > $0)
- `Total_Spendings`: Sum across all amenities (RoomService + FoodCourt + ShoppingMall + Spa + VRDeck)
- `Age_Group`: Categorical bins (Child: 0-12, Teen: 13-18, Young Adult: 19-30, Adult: 31-50, Senior: 51+)
- `Cabin_Deck`: Extracted from cabin string (deck location on ship: A, B, C, D, E, F, G, T)
- `Cabin_Side`: Extracted from cabin string (Port 'P' vs Starboard 'S')
- `Group_Size`: Number of passengers traveling together (extracted from PassengerId)

**Key Discovery:**
Passengers in CryoSleep had $0 spending across all amenities. The `Is_Spender` feature captured this pattern better than individual spending columns.

---

## 🛠️ Technical Approach

### Data Preprocessing

**Missing Value Strategy:**
- **Numerical features** (spending columns): Median imputation
- **CryoSleep**: False if Is_Spender=1, otherwise randomly sampled from training distribution
- **Categorical features**: "Unknown" category for HomePlanet, Destination, Cabin_Deck, and Cabin_Side
- **Age**: Median imputation (27), then grouped into age categories
- **VIP**: Filled with False (only 2% were True in training data)

**Encoding:**
- One-hot encoding for categorical features (HomePlanet, Destination, Cabin_Deck, Cabin_Side, Age_Group)
- `drop_first=True` to avoid multicollinearity

**Scaling:**
- StandardScaler on numerical features (RoomService, FoodCourt, ShoppingMall, Spa, VRDeck, Total_Spendings, Group_Size)
- Binary features (CryoSleep, VIP, Is_Spender) and one-hot encoded features left unscaled

### Models Tested

| Model | Validation Accuracy | Notes |
|-------|---------------------|-------|
| Logistic Regression | 79.07% | Baseline model |
| Logistic Regression (tuned) | 79.01% | C=10, minimal improvement from tuning |
| Random Forest | 80.45% | Better at capturing non-linear patterns |
| Random Forest (tuned) | 79.70% | max_depth=10, tuning hurt performance! |
| **XGBoost (default)** | **80.97%** | **Best validation score** |
| XGBoost (tuned) | 80.79% | learning_rate=0.3, max_depth=3, slightly worse |
| LightGBM (default) | 80.45% | Similar to Random Forest |
| LightGBM (tuned) | 80.39% | Tuning provided minimal gain |

**Winner:** XGBoost with default hyperparameters

---

## 📈 Results

### Final Performance

- **Validation Accuracy:** 80.97%
- **Public Leaderboard Score:** 79.6%
- **Gap:** 1.4% (acceptable, indicates good generalization)

### Confusion Matrix (Validation Set)
```
[[694 169]   ← Not Transported
 [162 714]]  ← Transported
```

**Balanced performance** across both classes (precision/recall ~0.81 for both).

---

## 💡 Key Learnings

### 1. Feature Engineering > Hyperparameter Tuning
- Default model parameters performed best
- Tuning often hurt performance (overfitting to CV folds)
- Strong features (Cabin_Deck, Is_Spender) made the biggest difference

### 2. EDA is Critical
- Systematic exploration revealed hidden patterns
- Cabin location was strongest predictor (53% spread)
- Spending behavior strongly tied to CryoSleep status

### 3. Simple is Often Better
- XGBoost with defaults: 80.97%
- XGBoost tuned extensively: 80.79%
- **Lesson:** Don't over-optimize!

### 4. Domain Knowledge Matters
- Understanding that CryoSleep passengers can't spend money led to `Is_Spender` feature
- Parsing structured data (Cabin format) revealed spatial patterns

---

## 🔧 Technologies Used

- **Python 3.10**
- **Data Analysis:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn, XGBoost, LightGBM
- **Environment:** Jupyter Notebooks, VS Code

---

## 🚀 How to Run

### 1. Clone Repository
```bash
git clone https://github.com/NirMarc/kaggle.git
cd kaggle/spaceship-titanic
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Data
Download competition data from [Kaggle](https://www.kaggle.com/competitions/spaceship-titanic/data) and place in `data/` folder, or use the provided script:
```bash
bash donwload-data.sh
```

### 4. Run Notebooks
```bash
jupyter notebook
```

Open `notebooks/01_eda.ipynb` to explore the analysis.

### 5. Generate Predictions
Run [notebooks/02_modeling.ipynb](notebooks/02_modeling.ipynb) to train the model and generate submission file.

**Note:** The modeling notebook uses the `prepare_data()` function from [utils.py](utils.py) to ensure consistent preprocessing between training and test data.

---

## 📚 Future Improvements

Ideas for pushing accuracy further:

- [ ] **Interaction features**: CryoSleep × Cabin_Deck, VIP × HomePlanet, Age_Group × Spending
- [ ] **Polynomial features**: Spending interactions (RoomService × FoodCourt)
- [ ] **Feature selection**: Remove low-importance features using XGBoost feature importance
- [ ] **Advanced ensembling**: Stacking with meta-learner
- [ ] **Group-based features**: Aggregate statistics per travel group (family patterns)
- [ ] **Neural networks**: Deep learning approach (though likely overkill for tabular data)
- [ ] **Cross-validation tuning**: More sophisticated hyperparameter optimization

---

## 📧 Contact

**Name:** Nir Marciano  
**GitHub:** https://github.com/NirMarc

---

## 🙏 Acknowledgments

- Kaggle for hosting the competition
- Anthropic's Claude for guidance throughout the project
- The ML community for shared insights and approaches

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).