# Video Game Sales – Machine Learning Coursework Project

This repository presents a machine-learning analysis of video game sales data, investigating how **console platform** and **game genre** influence **global sales performance**.  

---

## 1. Project Question and Motivation 

The video game industry is highly competitive and volatile, with commercial success depending on multiple interacting factors.  
Publishers and developers are particularly interested in understanding:

> **Do certain combinations of console platform and game genre consistently lead to higher global sales?**

The motivation for this project is to use historical sales data to:
- Quantify the influence of **platform (PS4, Xbox One, PC)** and **genre (Action, Adventure, Sports, Role-Playing)** on sales.
- Compare a **classical machine-learning approach (Random Forest)** with a **neural network model**, evaluating their ability to capture non-linear relationships in the data.
- Assess whether machine learning provides additional insight beyond simple empirical averages.

---

## 2. Dataset Description: What, Where, and Why

### What
The dataset contains historical video game sales data, including:
- Console platform
- Game genre
- Regional sales
- Total global sales (in millions of units)

### Where
- **Source:** Kaggle – *Video Game Sales Dataset*  
  [https://www.kaggle.com/datasets/lamskdna/video-games-sales](https://www.kaggle.com/datasets/lamskdna/video-games-sales)
- **File used in this project:**  
  `data/VideoGames_Sales.xlsx`

### Why
This dataset was chosen because it directly supports the project research question:
- It includes categorical variables (**console** and **genre**) suitable for both traditional ML and neural networks.
- It provides a continuous target variable (**total_sales(mil)**) appropriate for regression.
- The data volume is sufficient to train and evaluate models while remaining interpretable for coursework analysis.

### Key Columns Used
- `console`
- `genre`
- `total_sales (mil)`

---

## 3. Methods and Project Structure (README Presentation – 5%)

The project is organised in a progressive manner, starting from simple baseline analysis and moving toward more advanced models.

### Modelling Approach
1. **Baseline analysis**  
   Empirical mean sales by console and genre combinations.
2. **Classical machine learning**  
   Random Forest regression to model non-linear interactions.
3. **Neural network approach**  
   Multi-Layer Perceptron (MLP) models exploring training behaviour, epochs, batch size, and pre-training.

### Repository Structure
py/
│── functions.py # Shared helper functions (loading, cleaning, modelling, evaluation)
│
├── Q1/
│ └── Q1.ipynb # Baseline analysis and introductory methods
│
├── Q2/
│ └── Q2.ipynb # Random Forest regression and evaluation
│
└── Q3/
└── Q3.ipynb # Neural network modelling (assumes Q1/Q2 context)
