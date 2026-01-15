# <ins> Video Game Sales ML Project

This repo investigates how **console** and **genre** relate to **global sales**, using a classic ML approach (Random Forest) and a neural network approach in later questions.

## Project idea (what + why)
The video game market is volatile: publishers care about what kinds of games succeed on which platforms.  
**Goal:** use historical sales data to explore whether **platform (PS4/XOne/PC)** and **genre (Action/Adventure/Sports/Role-Playing)** are informative predictors of **Global Sales**, and what combinations tend to perform best.

## Dataset (what + where + why)
- **Source:** [Kaggle – [Video Game Sales](https://www.kaggle.com/datasets/lamskdna/video-games-sales)]
- **Why this dataset:** It contains platform, genre, and regional/global sales, which directly match the project question.
- **File used:** `data/VideoGames_Sales.xlsx`
- **Key columns used (example):** `console`, `genre`, `total_sales(mil)`

## Repository structure
- `py/functions.py` – helper functions (loading, cleaning, filtering, feature engineering, train/test split, evaluation)
- `py/Q1/Q1.ipynb` – beginner-friendly tutorial notebook for Question 1
- `py/Q2/Q2.ipynb` – beginner-friendly tutorial notebook for Question 2
- `py/Q3/Q3.ipynb` – intermediate-level notebook for Question 3 (assumes Q1/Q2 context)

## How to run
### 1) Clone the repo
```bash
git clone <your-repo-url>
cd video-game-sales-ml

