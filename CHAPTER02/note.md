# CHAPTER 2: End-to-End Machine Learning Project

Here are the main steps we will walk through:
1. Look at the big picture.
2. Get the data.
3. Explore and visualize the data to gain insights.
4. Prepare the data for machine learning algorithms.
5. Select a model and train it.
6. Fine-tune your model.
7. Present your solution.
8. Launch, monitor, and maintain your system.

## 1. Look at the big picture.
## 2. Get the data.
## 3. Explore and visualize the data to gain insights.
## 4. Prepare the data for machine learning algorithms.
### 4.1 Clean the data
### 4.2 Handling text and Categorical Attributes
### 4.3 Feature Scaling and Transformations
* Machine learning algorithms may struggle when numerical attributes have differing scales.
* In the housing data example, there's a significant scale difference between the number of rooms and median incomes.
* Without scaling, ***models might overlook median income in favor of the number of rooms***.
* Two primary methods for scaling attributes are <font color="red">***min-max scaling and standardization***.</font>

---
[**Warning**]
* Only fit scalers to the training data, avoiding the use of fit() or fit_transform() on other datasets.
* Once a scaler is trained, it can transform other sets like validation, test, or new data.
* Training set values will always fit the specified range, but outliers in new data might not.
* To ensure outliers remain within the range, ***set the clip hyperparameter to True.***
---

* <font color="red">Min-max scaling (also known as normalization) rescales attribute values to a range of 0 to 1.</font>
* It involves shifting and rescaling values by subtracting the minimum and dividing by the range (difference between min and max).
* Scikit-Learn offers a transformer called MinMaxScaler for this purpose.
* MinMaxScaler allows you to adjust the range using the feature_range hyperparameter.
* It can be useful, for example, ***when neural networks require inputs in the range of -1 to 1 for optimal performance.***

```python
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)
```

*  <font color="red"> Standardization involves two steps: subtracting the mean value and dividing by the standard deviation.</font>
* This process gives standardized values a mean of 0 and a standard deviation of 1.
* Unlike min-max scaling, standardization doesn't restrict values to a specific range (e.g., 0 to 1).
* Standardization is less affected by outliers, making it robust in such cases.
* Scikit-Learn provides a transformer called StandardScaler for standardization.

```python
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)
```
---
[**Warning**]
* If you want to scale a sparse matrix without converting it to a dense matrix first, you can use a StandardScaler with its ***with_mean hyperparameter set to False***: it will only divide the data by the standard deviation, without subtracting the mean (as this would break sparsity).
---

* Features with ***heavy-tailed distributions***, where values far from the mean are not exponentially rare, can cause problems when using min-max scaling or standardization.

* Machine learning models tend to perform poorly when most feature values are squeezed into a small range.

* Before applying scaling methods, it's advisable to transform the feature to mitigate the heavy tail and aim for a roughly symmetrical distribution.

* For positively skewed features with a heavy right tail, a common approach is to ******replace the feature with its square root or raise it to a power between 0 and 1.***

* In cases where the feature exhibits a long and heavy tail, such as a power law distribution, transforming the feature by taking its logarithm can be a beneficial strategy.

* ***For instance, population data often follows a power law distribution, and applying the logarithm can make it resemble a Gaussian distribution (bell-shaped) for improved modeling.***

![Figure 2-17. Transforming a feature to make it closer to a Gaussian distribution](images/end_to_end_project/Figure%202-17.%20Transforming%20a%20feature%20to%20make%20it%20closer%20to%20a%20Gaussian%20distribution.png)

* [***Handling Heavy-Tailed Features:***]
* Another approach is to bucketize the feature by dividing its distribution into roughly equal-sized buckets.
* Each feature value is then replaced with the index of the bucket it belongs to, similar to creating categorical features based on percentiles.
* Equal-sized bucketization results in a feature with an almost uniform distribution, eliminating the need for further scaling.
* Alternatively, you can divide the bucket indices by the number of buckets to normalize the values to the 0–1 range.

* [***Handling Multimodal Distributions:***]
* For features with ***multimodal distributions (multiple clear peaks)***, like housing_median_age, bucketizing can also be beneficial.

* In this case, treat the bucket IDs as categories rather than numerical values.

* Encode the bucket indices, for example using a OneHotEncoder.
This approach allows regression models to learn different rules for different ranges of the feature values, accommodating variations in the data.

* For more nuanced transformations of multimodal features, you can add a feature for each mode, representing the similarity between the feature and that particular mode.

* The similarity measure is typically computed using a radial basis function (RBF) that depends on the distance between the input value and a fixed point.

* The Gaussian RBF is commonly used, with its output value decaying exponentially as the input moves away from the fixed point.
The hyperparameter γ (gamma) determines the decay rate of the similarity measure as the input moves away from the fixed point.

* Scikit-Learn's rbf_kernel() function can be used to create a new Gaussian RBF feature that measures the similarity between the feature and a chosen point, such as 35 in the case of housing median age.