# Projects done as the part of [*PGPAIML* @ UT Austin](https://la.utexas.edu/texasexeced/digitalVerification.html?key=cNbNF)

The objective of this portfolio is to demonstrate the application of AI and machine learning techniques to real-world business problems, such as fraud detection, customer segmentation, and demand forecasting. The projects covers topics such as data analysis, machine learning models, deep learning, natural language processing, and computer vision. 


<h3> 1. <a href="https://nbviewer.org/github/pnkjsyngh/MLProjects/blob/main/EDA/Project1.ipynb"><strong>Exploratory Data Analysis</strong></a></h3>

This project involves conducting an **Exploratory Data Analysis (EDA)** on [Cardio Good Fitness data](https://www.kaggle.com/datasets/saurav9786/cardiogoodfitness), which contains information on customers of treadmill products, including their age, gender, education, marital status, usage, fitness, income, and expected miles to run. The objective of the project is to identify differences between customers of each product, explore relationships between customer attributes, and propose additional relevant questions for the business. The project will deliver a customer profile for each product, perform univariate and multivariate analyses, and generate insights and recommendations for the company to target new customers. 

#### Skills gained
- **Descriptive Statistics**, **Inferential Statistics** and **Probability Distributions**
- Demonstrated effective use of **NumPy**, **Pandas** and **Seaborn** libraries to generate custom visualizations which captures the essence of the dataset and delivers the key message.

<h3> 2. <a href="https://nbviewer.org/github/pnkjsyngh/MLProjects/blob/main/LinearRegression/Project2.ipynb"><strong>Linear Regression</strong></a></h3>

The project involves developing a pricing model for used cars in the Indian market to help a tech start-up called Cars4U, which aims to find footholds in this market. The [data set](https://www.kaggle.com/datasets/sukhmanibedi/cars4u) contains different attributes of used cars sold in various locations, including brand and model name, location, manufacturing year, kilometers driven, fuel type, transmission, ownership type, engine displacement, power, mileage, seats, new car price, and used car price. The project objective is to **explore and visualize the dataset**, build a **linear regression model** to predict the prices of used cars, and generate insights and recommendations to help the business. The project deliverables include a pricing model, linear regression predictions, and actionable insights and recommendations. 

#### Skills gained
- *EDA*, *data pre-processing*, *linear regression*, *model building and evaluation*, *feature selection*
- Tested different version of linear regression with feature modifications and selection using [SFS](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html)
- Delivered the model with features selected using SFS as the most accurate model based on the performance metrics shown below.

| Metrics       | Lin Reg       | Lin Reg with log features | Lin Reg with log features using [SFS](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html)|
|:-------------:|:-------------:|:-------------------------:|:----------------------------------:|
| RMSE          | 4.777414	    | 4.141068                  |	4.028353                           |
| MAE           | 2.739404	    | 1.853221                  |	1.814495                           |
| R^2           | 0.807807	    | 0.855597	                |	0.863351                           |
| Adjusted R^2  | 0.801880	    | 0.851144                  |	0.860175                           |

<h3> 3. <a href="https://nbviewer.org/github/pnkjsyngh/MLProjects/blob/main/LogisticRegression/Project3.ipynb"><strong>Logistic Regression</strong></a></h3>

The goal of this project is to help AllLife Bank increase their loan business by building a model to identify potential customers who are likely to purchase a personal loan. The data contains customer attributes such as age, income, education level, and whether or not they accepted a personal loan in the past. The deliverables include predicting whether a liability customer will buy a personal loan, determining which variables are most significant, and identifying which customer segment should be targeted more.

#### Skills gained
- *Confusion matrix*, *perfromace metrics*
- *Decision Tree*, *Pruning*, *model performance and feature selection*
- *Logistic regression*, *ROC-AUC curve*, *Precision-Recall curve* 
- Delivered the models based on Logistic regression and Decision Tree, with metrics shown below.

| Metrics       | Logistic Regression | Decision Tree |
|:-------------:|:-------------------:|:-------------:|
| Accuracy      | 0.925333	          | 0.938000      |	
| Recall        | 0.812081	          | 0.979866      |	
| Precision     | 0.590244	          | 0.618644      |	
| F1            | 0.683616	          | 0.758442      |	

<h3> 4. <a href="https://nbviewer.org/github/pnkjsyngh/MLProjects/blob/main/EnsembleTechniques/Project4.ipynb"><strong>Ensemble Techniques</strong></a></h3>

As a Data Scientist for "Visit with us" tourism company, the task is to analyze customer data and information to provide recommendations for expanding the customer base through the introduction of a new wellness tourism package. The goal is to build a model that can predict potential customers who are likely to purchase the new offering, based on available data, to enable the company to target the right customers before they are contacted. The project aims to establish a viable business model to benefit the tourism sector by changing the ways of doing business.

#### Skills gained
- *Confusion matrix*, *recall*, *precision*, *F1-score*
- *Decision Tree*, *Random Forest*, *Bagging*, *Boosting*, *Gradient Boosting*, *XGBoost*
- *Hyperparameter tuning*, *Stacking classifier*, *Model Explainability*
- Delivered the models based on Ensemble techniques, with metrics shown below.


| Metrics       | Decision Tree       | XGBoost       | Stacking |
|:-------------:|:-------------------:|:-------------:|:--------:|
| Accuracy      | 0.69    	          | 0.89          |	0.91     |
| Recall        | 0.76     	          | 0.74          |	0.69     |
| Precision     | 0.36    	          | 0.71          |	0.84     |
| F1            | 0.49    	          | 0.72          |	0.76     |

<p align="center">
  <img src="https://user-images.githubusercontent.com/25642673/229942295-8272d3b5-fe56-4fa2-a4e9-0f157e5c6e5d.jpg" width=50% height=50%>
</p>


<h3> 5. <a href="https://nbviewer.org/github/pnkjsyngh/MLProjects/blob/main/FeatureSelection%26ModelTuning/Project5.ipynb"><strong>Feature Selection and Model Tuning</strong></a></h3>

The Thera bank is experiencing a decline in credit card users, which is affecting their revenue. They have provided a dataset containing customer information, including demographics, financial history, and transaction data. The bank wants to identify customers who are likely to leave their credit card services and the reasons for their departure, to improve their services and retain customers. The goal of this project is to develop a classification model to predict which customers are at risk of leaving, and to identify the best model for this purpose.

#### Skills gained
- *Pipelines*, *Ensemble methods*
- *Imbalanced datasets: Oversampling and Undersampling of training data*
- *Hyperparameter tuning: Grid and Randomized search*
- *K-Fold cross-validation*
- *Model Explainability: Feature importance and SHAP values*
- Delivered the models based on Gradient Boost Classifier (GBC), after comparing performance metrics over various Ensemble based models while using normal/oversampled/undersampled training data.

<p align="center">
  <img src="https://user-images.githubusercontent.com/25642673/229965805-bd5beddf-9f55-457d-a7be-1d8c2a7aa371.png" width=50% height=50%>
</p>


<h3> 6. <a href="https://nbviewer.org/github/pnkjsyngh/MLProjects/blob/main/UnsupervisedLearning/Project6.ipynb"><strong>Unsupervised Learning</strong></a></h3>

This project is an impactful example of unsupervised learning problem because it aims to use clustering algorithms to identify different segments of customers based on their spending patterns and past interactions with the bank, without having any pre-defined labels or target variable. The goal is to discover hidden patterns and relationships in the data that can help the bank improve its marketing and service delivery to different customer groups. The unsupervised learning techniques used in this project can have a significant impact on the bank's business outcomes by enabling more personalized and effective customer engagement strategies.

#### Skills gained
- *K-Means clustering*, *Elbow and Silhoutte plot for cluster optimization*
- *Hierarchial clustering*, *Distance based clustering*, *Cophenactic correlation*
- *Principal Component Analysis (PCA)*
- Explored various models for clustering and optimized their performance. Subsequently, used PCA to reduce the number of features followed by clustering as shown in the image below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/25642673/230365717-42d04034-9c93-413f-a01f-f585b9d03cfb.png" width=50% height=50%>
</p>

<h3> 7. <a href="https://nbviewer.org/github/pnkjsyngh/MLProjects/blob/main/NeuralNetworks/Project7.ipynb"><strong>Neural Networks</strong></a></h3>

This project is an impactful example of a neural network application in market analytics and customer prediction. It uses a dataset from Kaggle and aims to help businesses like banks address the problem of customer churn by building a neural network-based classifier to predict whether a customer will leave in the next 6 months. By analyzing customer information such as credit score, location, age, and account balance, this project can help businesses identify factors that influence customer decisions and improve their services accordingly to retain customers.

#### Skills gained
- *Neural Networks*, *TensorFlow*
- *Batch Normalization*, *Regularization*, *Dropout layers*, *Imbalanced data: Under/Over sampling*
- *Hyperparameter optimization*, *Performance metrics*, $F_\beta$ *score*
- Delivered a NN model using dropout layers and L2 regularization with the best $F_\beta$ score = 0.656. The model performance and explainability is shown below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/25642673/230371095-6884ded5-d1fd-4afb-9361-074adc392009.jpg" width=50% height=50%>  
  <img src="https://user-images.githubusercontent.com/25642673/230371860-fcc77f0b-922d-4926-8edf-fbae46bc482d.jpg" width=50% height=50%>
</p>
<h3> 8. <a href="https://nbviewer.org/github/pnkjsyngh/MLProjects/blob/main/ComputerVision/Project8.ipynb"><strong>Computer Vision</strong></a></h3>

As a data scientist with expertise in computer vision, we have been tasked with a highly impactful project in the field of agriculture. The mission is to utilize deep learning techniques and convolutional neural networks to build a model that can classify images of plant seedlings into 12 distinct species. This application of computer vision has the potential to revolutionize the trillion-dollar agricultural industry by greatly reducing the amount of manual labor required for plant recognition, resulting in increased crop yields, more sustainable practices, and the freeing up of valuable human resources for higher-level decision making.

#### Skills gained
- *CNN*, *Data Augmentation*
- *Transfer learning*
- After comparing various CNN architecture, we delivered a performance of 86% accuracy in a model based on data augmentation.

<p align="center">
  <img src="https://user-images.githubusercontent.com/25642673/230534171-01cbf185-9a13-4e9c-8c84-8519ed308212.png" width=50% height=50%>  
</p>

<h3> 9. <a href="https://nbviewer.org/github/pnkjsyngh/MLProjects/blob/main/NaturalLanguageProcessing/Project9.ipynb"><strong>Natural Language Processing</strong></a></h3>

This project involves sentiment analysis on tweets about major U.S. airlines, with the aim of identifying negative mentions and reasons for negative sentiment. The project uses natural language processing (NLP) techniques to preprocess, vectorize, and classify the tweets. The application of sentiment analysis in social media marketing is an impactful example of NLP, as it allows companies to quickly and efficiently monitor customer sentiment and respond appropriately to negative mentions.

#### Skills gained
- *Text cleaning*, *Tokenization*, *Dataset creation using Countvectorizer and TF-IDF*
- *Sentiment classification using Ensemble techniques*
- Delivered a model with XGBoost created input dataset using Countvectorizer. This model delivered an accuracy of 80% in sentiment classification.

<p align="center">
  <img src="https://user-images.githubusercontent.com/25642673/230536351-d89c17ec-30d0-42cc-91ca-75f8e2cf258d.png" width=50% height=50%>  
</p>
