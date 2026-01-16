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

## 3. Methods and Project Structure

The project is organised in a progressive manner, starting from simple baseline analysis and moving toward more advanced models.

### Modelling Approach
1. **Baseline analysis**  
   Empirical mean sales by console and genre combinations.
2. **Classical machine learning**  
   Random Forest regression to model non-linear interactions.
3. **Neural network approach**  
   Multi-Layer Perceptron (MLP) models exploring training behaviour, epochs, batch size, and pre-training.

### Repository Structure
Q1_folder/
│── Q1.ipynb # Baseline analysis and Random Forest regression and evaluation

Q2_folder/
│── Q2.ipynb # Neural network modelling and evaluation

Q3_folder/
│── Q3.ipynb # Neural network modelling and training analysis

py/
│── functions.py # Shared helper functions (data loading, cleaning, modelling)

LICENSE
README.md
dependencies.txt



Each notebook builds on the previous one and is written in a tutorial-style format to clearly explain modelling decisions and results.

---

## 4. Dependencies and Software Versions (5%)

All required dependencies are listed and versioned to ensure reproducibility.

### Python Libraries
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

Exact package versions are specified in `dependencies.txt`.

## How to Run the Project
1)Clone the repository

git clone <your-repo-url>
cd video-game-sales-ml

2)Install dependencies

pip install -r dependencies.txt

3)Run the notebooks

Open the notebooks in Jupyter and run them in order:

1.  Q1_folder/Q1.ipynb

2.  Q2_folder/Q2.ipynb

3.  Q3_folder/Q3.ipynb
