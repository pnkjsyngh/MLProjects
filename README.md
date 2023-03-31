# Machine Learning Projects

<h3> 1. <a href="https://nbviewer.org/github/pnkjsyngh/MLProjects/blob/main/EDA/Project1.ipynb"><strong>Exploratory Data Analysis</strong></a></h3>

This project involves conducting an **Exploratory Data Analysis (EDA)** on [Cardio Good Fitness data](https://www.kaggle.com/datasets/saurav9786/cardiogoodfitness), which contains information on customers of treadmill products, including their age, gender, education, marital status, usage, fitness, income, and expected miles to run. The objective of the project is to identify differences between customers of each product, explore relationships between customer attributes, and propose additional relevant questions for the business. The project will deliver a customer profile for each product, perform univariate and multivariate analyses, and generate insights and recommendations for the company to target new customers. 

#### Skills gained
- **Descriptive Statistics**, **Inferential Statistics** and **Probability Distributions**
- Demonstrated effective use of **NumPy**, **Pandas** and **Seaborn** libraries to generate custom visualizations which captures the essence of the dataset and delivers the key message.

<h3> 2. <a href="https://nbviewer.org/github/pnkjsyngh/MLProjects/blob/main/LinearRegression/Project2.ipynb"><strong>Linear Regression</strong></a></h3>

The project involves developing a pricing model for used cars in the Indian market to help a tech start-up called Cars4U, which aims to find footholds in this market. The [data set](https://www.kaggle.com/datasets/sukhmanibedi/cars4u) contains different attributes of used cars sold in various locations, including brand and model name, location, manufacturing year, kilometers driven, fuel type, transmission, ownership type, engine displacement, power, mileage, seats, new car price, and used car price. The project objective is to **explore and visualize the dataset**, build a **linear regression model** to predict the prices of used cars, and generate insights and recommendations to help the business. The project deliverables include a pricing model, linear regression predictions, and actionable insights and recommendations. 

#### Skills gained
- **EDA**, **data pre-processing**, **linear regression**, **model building and evaluation**, **feature selection**
- Tested different version of linear regression with feature modifications and selection using [SFS](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html)

| Metric | Lin Reg sklearn	| Lin Reg sklearn w/ log ftrs	| Lin Reg sklearn w/ log ftrs and seats as cat	| Lin Reg sklearn w/ log, seats as cat and SFS ftrs|

RMSE	4.777414	4.141068	3.973063	4.028353
MAE	2.739404	1.853221	1.804374	1.814495
R-squared	0.807807	0.855597	0.867077	0.863351
Adj. R-squared	0.801880	0.851144	0.862506	0.860175
