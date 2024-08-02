# 2024_ia651_summer_shi

## Dataset

- **Source**: The dataset used for this project is `athlete_events.csv`, obtained from [[Kaggle([https://www.kaggle.com](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results))].
- **Key Fields**:
  - `ID`: Unique identifier for each athlete.
  - `Name`: Name of the athlete.
  - `Sex`: Gender of the athlete.
  - `Age`: Age of the athlete.
  - `Height`: Height of the athlete.
  - `Weight`: Weight of the athlete.
  - `Team`: Team or country the athlete represents.
  - `NOC`: National Olympic Committee country code.
  - `Games`: Year and season of the Olympic Games.
  - `Year`: Year of the Olympic Games.
  - `Season`: Season of the Olympic Games (Summer or Winter).
  - `City`: Host city of the Olympic Games.
  - `Sport`: Sport in which the athlete competed.
  - `Event`: Specific event within the sport.
  - `Medal`: Medal won (Gold, Silver, Bronze, or NaN if no medal).

- **Objective**: The goal of this project is to build a predictive model that can determine whether an athlete will win a medal based on various features such as age, height, weight, sex, team, sport, event, and home game advantage.

## Practical Applications

- **Athlete Training**: Coaches can use predictions to enhance training programs and allocate resources to athletes with higher medal-winning potential.
- **Event Strategy**: NOCs can make strategic decisions on athlete participation, leveraging home advantage effectively.

## Exploratory Data Analysis (EDA)

### Features

- **X-variables**:
  1. `Sex`: Gender of the athlete.
  2. `Age`: Age of the athlete.
  3. `Height`: Height of the athlete.
  4. `Weight`: Weight of the athlete.
  5. `Team`: Team or country the athlete represents.
  6. `Sport`: Sport in which the athlete competed.
  7. `Event`: Specific event within the sport.
  8. `HomeGame`: Indicates if the athlete’s team matches the host country of the Olympics.

- **Y-variable**:
  - `Medal_Win`: Binary variable indicating whether the athlete won a medal (1 if the athlete won a medal, 0 otherwise).
  - Since the Y-varibale is a binary variable that indicates either an athlete won(1) or did not won(0), this project is classification.
  - Number of Observations: 271,116
   - Feature to obs: 8/271,116 = 0.0000295

- **Data Balance**:
  - `Medal_Win` is highly imbalanced with the majority class being no medal (0) and the minority class being medal (1).

### Distributions and Correlations

- **Numerical Feature Distributions**:
  - **Age**: The majority of athletes are between 20 and 30 years old.
  - **Height**: The majority of athletes have a height around 170-180 cm.
  - **Weight**: The majority of athletes have a weight around 70-80 kg.
![7493433b-bf7c-48ff-828b-5edbf8b9e871](https://github.com/user-attachments/assets/cd805acd-8b6c-46f3-b4c4-acd996d03c22)

  - **Sex**: Majority are male.
  - **Team**: A large number of unique teams with some teams having significantly more athletes.
  - **Sport**: A large number of unique sports with some sports having significantly more athletes.
  - **Event**: Highly varied with many unique events.
  - **HomeGame**: Majority will not have the home game advantage (0), while a smaller subset will (1).

- **Correlation**:
  - Height and weight are strongly correlated with a correlation of 0.8, indicating potential multicollinearity.

### Categorical Variable Analysis

- **Count Plots**:
  - **Sex**:
    - Male: Majority
    - Female: Minority ![d6927b17-0396-47ec-8e84-4e8c6247a7d6](https://github.com/user-attachments/assets/b880276c-9b40-4b5c-9cef-d0b1236460c0)


  - **Team**: Distribution shows a large number of unique teams, with some teams having significantly more athletes, leading to sparse data for some teams.![20c20e78-2396-4623-817f-c8a82904b314](https://github.com/user-attachments/assets/259c2e58-b438-4721-b387-db8331acc597)


  - **Sport**: Distribution indicates a large number of unique sports, with some sports having significantly more athletes.![fa75d795-e28a-4f95-94ac-14ea0247b399](https://github.com/user-attachments/assets/4d3e73bf-1fc7-4da3-a149-adfed8923ea6)

  - **Event**: Distribution is highly varied with many unique events, leading to sparse data for some events.
  - **HomeGame**: Majority of observations will not have the home game advantage (0), while a smaller subset will (1).
![b44aeca2-7a08-4e93-bdd6-ef179a2d9421](https://github.com/user-attachments/assets/3a1874dd-dfb8-4470-a483-41a15e0afa4b)


- **Cross-tabulations and Heatmaps**:
  - **Sex and Medal_Win**: Cross-tabulation and heatmap show the distribution of medal wins by gender.![7f776821-3726-46c0-b98b-726d5c26b460](https://github.com/user-attachments/assets/4eec2ed5-7768-4a62-a39d-389a28960cf1)


  - **Sport and Medal_Win**: Cross-tabulation and heatmap show the distribution of medal wins by sport.
![f540b67b-4d75-413f-be35-9b215cd6ecef](https://github.com/user-attachments/assets/919b2515-0089-404a-b518-d4db8b82d266)

  - **NOC and Medal_Win**: Cross-tabulation and heatmap illustrate the distribution of medal wins by country.![613bf139-0843-4f93-9079-2d0760aa6ec2](https://github.com/user-attachments/assets/388eef04-26f7-4486-9aec-28acb66ba672)


  - **Sport and Sex**: Cross-tabulation and heatmap reveal the distribution of athletes by sport and gender.
![a70b36fe-85f6-413c-990d-d14de0ce3eb4](https://github.com/user-attachments/assets/5bc1216b-48db-41b4-b721-e01443b26840)


## Feature Engineering

- **Categorical Features**: `Sex`, `Team`, `Sport`, `Event` were encoded using one-hot encoding.
- **Composite Feature**: The HomeGame feature was created to indicate if the athlete’s team matched the host country for the particular Olympic event year.
  
```python
host_countries = {
    '1896': 'Greece', '1900': 'France', '1904': 'USA', '1908': 'UK', '1912': 'Sweden', 
    '1920': 'Belgium', '1924': 'France', '1928': 'Netherlands', '1932': 'USA', '1936': 'Germany', 
    '1948': 'UK', '1952': 'Finland', '1956': 'Australia', '1960': 'Italy', '1964': 'Japan', 
    '1968': 'Mexico', '1972': 'Germany', '1976': 'Canada', '1980': 'Russia', '1984': 'USA', 
    '1988': 'South Korea', '1992': 'Spain', '1996': 'USA', '2000': 'Australia', '2004': 'Greece', 
    '2008': 'China', '2012': 'UK', '2016': 'Brazil'
}

# Calculate HomeGame feature
data['HomeGame'] = data.apply(lambda row: 1 if host_countries.get(str(row['Year'])) == row['Country'] else 0, axis=1)
```

## Model Fitting

### Data Preparation

- **Train/Test Splitting**: Random splitting into training (80%) and testing (20%) sets.
- **Preprocessing**:
  - **Numerical Features**: Imputed using the median strategy and scaled.
  - **Categorical Features**: Imputed using a constant value and one-hot encoded.

### Model Selection

- **Initial Model**: Logistic Regression
  - **Rationale**: Simplicity and interpretability.
- **Hyperparameter Tuning**:
  - Grid Search and Cross-Validation were employed to find the optimal parameters.

### Metrics

- **Primary Metrics**:
  - **Precision**: Ensures resources are not wasted on athletes unlikely to win medals.
  - **Recall**: Ensures no potential medal winners are overlooked.
  - **F1-score**: Given the class imbalance, it provides more insight into model performance on the minority class than accuracy.
- **Confusion Matrix**: Helps understand model errors, particularly in the context of class imbalance.

## Model Performance

### Logistic Regression report

<img width="431" alt="Screenshot 2024-08-02 at 16 51 18" src="https://github.com/user-attachments/assets/8f25763b-ceb4-4a13-8964-256784953bed">

### No Medal:
- **Precision:** 0.87
  - This indicates that 87% of the instances predicted as "No Medal" are actually "No Medal."
- **Recall:** 0.99
  - This indicates that 99% of the actual "No Medal" instances are correctly identified by the model.
- **F1-Score:** 0.93
  - The harmonic mean of precision and recall, indicating a good balance between the two for this class.

### Medal:
- **Precision:** 0.67
  - This indicates that 67% of the instances predicted as "Medal" are actually "Medal."
- **Recall:** 0.15
  - This indicates that only 15% of the actual "Medal" instances are correctly identified by the model.
- **F1-Score:** 0.24
  - The harmonic mean of precision and recall, indicating poor performance in identifying this class.


### Overall Metrics:
- **Accuracy:** 0.87
  - The model correctly predicts 87% of all instances.
- **Macro Avg (Macro Average):**
  - **Precision:** 0.77
    - The average precision across both classes, not considering the class imbalance.
  - **Recall:** 0.57
    - The average recall across both classes, not considering the class imbalance.
  - **F1-Score:** 0.59
    - The average F1-score across both classes, indicating overall model performance without considering class imbalance.
- **Weighted Avg (Weighted Average):**
  - **Precision:** 0.84
    - The average precision weighted by the number of instances in each class.
  - **Recall:** 0.87
    - The average recall weighted by the number of instances in each class.
  - **F1-Score:** 0.83
    - The average F1-score weighted by the number of instances in each class, indicating overall model performance considering class imbalance.

## Random Forest Classification Report:

<img width="431" alt="Screenshot 2024-08-02 at 16 52 05" src="https://github.com/user-attachments/assets/516259ba-da0c-49c2-b203-33f4f79e3a24">



### No Medal:
- **Precision:** 0.91
  - This indicates that 91% of the instances predicted as "No Medal" are actually "No Medal."
- **Recall:** 0.96
  - This indicates that 96% of the actual "No Medal" instances are correctly identified by the model.
- **F1-Score:** 0.94
  - The harmonic mean of precision and recall, indicating excellent performance for this class.

### Medal:
- **Precision:** 0.67
  - This indicates that 67% of the instances predicted as "Medal" are actually "Medal."
- **Recall:** 0.45
  - This indicates that 45% of the actual "Medal" instances are correctly identified by the model.
- **F1-Score:** 0.54
  - The harmonic mean of precision and recall, indicating moderate performance for this class.

### Overall Metrics:
- **Accuracy:** 0.89
  - The model correctly predicts 89% of all instances.
- **Macro Avg (Macro Average):**
  - **Precision:** 0.79
    - The average precision across both classes, not considering the class imbalance.
  - **Recall:** 0.71
    - The average recall across both classes, not considering the class imbalance.
  - **F1-Score:** 0.74
    - The average F1-score across both classes, indicating overall model performance without considering class imbalance.
- **Weighted Avg (Weighted Average):**
  - **Precision:** 0.88
    - The average precision weighted by the number of instances in each class.
  - **Recall:** 0.89
    - The average recall weighted by the number of instances in each class.
  - **F1-Score:** 0.88
    - The average F1-score weighted by the number of instances in each class, indicating overall model performance considering class imbalance.
   
### Underfitting Model:
  - Even though the metrics looks generally decent with Accuracy(0.89), Precition(0.79) and Recall(0.71), the imbalance in 'Medal' class is realatively low, resulting an identification of underfitting model. another reason of this situation is the natrual trait of this dataset. because of most of the athlete did not win a medal, the classification fo 'medal' only takes a tiny portion of the dataset.

### Production:
  - Keep track of different versions of the model with clear documentation of changes, performance metrics, and hyperparameters
  - Continiously detect degradation in accuracy over time.
  - pay extra attetion to biases and monitor the model for fairness.

### Going Further:
  - How to improve the model
    - More Data: Joint the dataset with more dataset related to the athlete to generate more composite features 
    - Feature Enginerring: Figure out more features that are more related to the model prediction. For example, the historcial performance/ record of athletes, and their traning data if possible.
    - Hyperparameter Optimization: Perform hyperparameter tuning techniques like Grid Search, Random Search, or Bayesian Optimization to find the optimal set of hyperparameters.
      
