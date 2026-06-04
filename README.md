# Data Science Project: Recipes and Ratings

## Introduction
This project centers around two datasets that are originally from [food.com](https://food.com/), an online community for cooking lovers where users can share, review, and rate recipes. The two datasets are **recipes** (731927 rows, 5 columns), containing information about recipes (each row is an individual recipe), and **interactions** (83782 rows, 12 columns), containing customers' reviews and ratings on the recipes (each row is a review). 

There are a lot of variations among the recipes, with some recipes taking only one or two minutes to make while others taking months. Users of [food.com](https://food.com/) also have different perceptions of the recipes, with some recipes receiving a five out of five rating while others being rated only one out of five. Looking through the data, one might wonder: what makes a recipe more popular than others? One guess would be the complexity of a recipe, as in how manys steps it takes to make the recipe or how many ingredients one need to use to make it. The time it takes to make a recipe would also reflect its complexity. People might like a complex recipe better because the food will turn out to be more interesting and delicious, or they might prefer a simpler recipe because it takes less time and effort. Therefore, we explored the question **How does the complexity of a recipe relate to users' rating of it?** by conducting some data analysis. Later on, we also looked into how we can **predict the rating of a recipe from its characteristics** so that when we see a recipe, we can infer whether people will like it or not, and thereby decide if it's worthwhile trying.

For the purposes of answering our question, the following columns from the datasets were used. From **recipes**, we used the columns `minutes`, `nutrition`, `n_steps` and `n_ingredients`, which are attributes that can help answer our question and make predictions. From **interactions**, we used the column `rating` to collect the many ratings each recipes have received.


## Data Cleaning and Exploratory Data Analysis

### Data Cleaning
**recipes** and **interactions** are loaded as Pandas dataframes. Right now, the characteristics of each recipe and their ratings are in separate datasets. We merged the datasets by the recipe id's so that we could work with only a single dataset. Then, we filled all ratings of 0 in the merged dataset with NAN (not a number) values. This is because in [food.com](https://www.food.com/), ratings of recipes are between 1 and 5. When we see a rating of 0, it means that a user interacted with the recipe, made a review, but did not make a rating. Therefore, when we see a rating of 0, it does not mean that a user rated a recipe 0, but rather that specific rating is invalid. It is therefore correct for us to record that rating as NAN, or in other words, missing.

From the merged dataset, we calculate the average rating of each recipe and save that information as a Pandas series. Then, we add it back to the **recipes** dataset. This is because the merged dataframe has multiple rows for a single recipe corresponding to the different instances of user reviews. By adding this 'average rating' dataset back to **recipes**, we end up with a dataframe with each row representing a different recipe, with all its attributes in the columns plus its average rating from the multiple reviews. 

Next, we parse the `nutrition` column into separate columns based on the nutrition category. The `nutrition` column has entries in the format "[calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), and carbohydrates (PDV)]". By parsing it we create a separate column for each nutrition category, making it easier for later data analysis.

Then, we noticed a suspicious value in the `minutes` column. The column has maximum value 1051200, which is almost two years. This is a suspicious value. To investigate, we take a closer look at the top ten recipes that take the most time to make. We found that most of them are preserves, pickles, vinegar, or alcohol, which are indeed food that can take months to make, so they are reasonable values. However, the recipe that takes the longest time is different, as it takes way more time than the others. After investigating that specific recipe, we found out that it is not a real recipe, just a joke about relationship that was thrown in here. The `steps` of this recipe shows this:


| steps          | ['be careful in your selection', "don't choose too young", 'when selected , give your entire thoughts to preparation for domestic use', 'some wives insist upon keeping them in a pickle , others are constantly getting them into hot water', 'this may make them sour , hard and sometimes bitter', 'even poor varieties may be made sweet , tender and good by garnishing with patience , well sweetened with love and seasoned with kisses', 'wrap them in a mantle of charity', 'keep warm with a steady fire of domestic devotion and serve with peaches and cream', 'thus prepared , they will keep for years'] |

Therefore, we dropped that row because it is not a valid observation and would add noise to our analysis.

We end up with a cleaned **recipes** dataframe, with the first three rows looking like this:

| name                                 |     id |   minutes |   contributor_id | submitted   | tags                                                                                                                                                                                                                        | nutrition                                    |   n_steps | steps                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | description                                                                                                                                                                                                                                                                                                                                                                       | ingredients                                                                                                                                                                    |   n_ingredients |   avg_rating |   calories |   total fat |   sugar |   sodium |   protein |   saturated fat |   carbohydrates |
|:-------------------------------------|-------:|----------:|-----------------:|:------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------|----------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|-------------:|-----------:|------------:|--------:|---------:|----------:|----------------:|----------------:|
| 1 brownies in the world    best ever | 333281 |        40 |           985201 | 2008-10-27  | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'for-large-groups', 'desserts', 'lunch', 'snacks', 'cookies-and-brownies', 'chocolate', 'bar-cookies', 'brownies', 'number-of-servings'] | [138.4, 10.0, 50.0, 3.0, 3.0, 19.0, 6.0]     |        10 | ['heat the oven to 350f and arrange the rack in the middle', 'line an 8-by-8-inch glass baking dish with aluminum foil', 'combine chocolate and butter in a medium saucepan and cook over medium-low heat , stirring frequently , until evenly melted', 'remove from heat and let cool to room temperature', 'combine eggs , sugar , cocoa powder , vanilla extract , espresso , and salt in a large bowl and briefly stir until just evenly incorporated', 'add cooled chocolate and mix until uniform in color', 'add flour and stir until just incorporated', 'transfer batter to the prepared baking dish', 'bake until a tester inserted in the center of the brownies comes out clean , about 25 to 30 minutes', 'remove from the oven and cool completely before cutting']                                                  | these are the most; chocolatey, moist, rich, dense, fudgy, delicious brownies that you'll ever make.....sereiously! there's no doubt that these will be your fav brownies ever for you can add things to them or make them plain.....either way they're pure heaven!                                                                                                              | ['bittersweet chocolate', 'unsalted butter', 'eggs', 'granulated sugar', 'unsweetened cocoa powder', 'vanilla extract', 'brewed espresso', 'kosher salt', 'all-purpose flour'] |               9 |            4 |      138.4 |          10 |      50 |        3 |         3 |              19 |               6 |
| 1 in canada chocolate chip cookies   | 453467 |        45 |          1848091 | 2011-04-11  | ['60-minutes-or-less', 'time-to-make', 'cuisine', 'preparation', 'north-american', 'for-large-groups', 'canadian', 'british-columbian', 'number-of-servings']                                                               | [595.1, 46.0, 211.0, 22.0, 13.0, 51.0, 26.0] |        12 | ['pre-heat oven the 350 degrees f', 'in a mixing bowl , sift together the flours and baking powder', 'set aside', 'in another mixing bowl , blend together the sugars , margarine , and salt until light and fluffy', 'add the eggs , water , and vanilla to the margarine / sugar mixture and mix together until well combined', 'add in the flour mixture to the wet ingredients and blend until combined', 'scrape down the sides of the bowl and add the chocolate chips', 'mix until combined', 'scrape down the sides to the bowl again', 'using an ice cream scoop , scoop evenly rounded balls of dough and place of cookie sheet about 1 - 2 inches apart to allow for spreading during baking', 'bake for 10 - 15 minutes or until golden brown on the outside and soft & chewy in the center', 'serve hot and enjoy !'] | this is the recipe that we use at my school cafeteria for chocolate chip cookies. they must be the best chocolate chip cookies i have ever had! if you don't have margarine or don't like it, then just use butter (softened) instead.                                                                                                                                            | ['white sugar', 'brown sugar', 'salt', 'margarine', 'eggs', 'vanilla', 'water', 'all-purpose flour', 'whole wheat flour', 'baking soda', 'chocolate chips']                    |              11 |            5 |      595.1 |          46 |     211 |       22 |        13 |              51 |              26 |
| 412 broccoli casserole               | 306168 |        40 |            50969 | 2008-05-30  | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                        | [194.8, 20.0, 6.0, 32.0, 22.0, 36.0, 3.0]    |         6 | ['preheat oven to 350 degrees', 'spray a 2 quart baking dish with cooking spray , set aside', 'in a large bowl mix together broccoli , soup , one cup of cheese , garlic powder , pepper , salt , milk , 1 cup of french onions , and soy sauce', 'pour into baking dish , sprinkle remaining cheese over top', 'bake for 25 minutes or until cheese is lightly browned', 'sprinkle with rest of french fried onions and bake until onions are browned and cheese is bubbly , about 10 more minutes']                                                                                                                                                                                                                                                                                                                              | since there are already 411 recipes for broccoli casserole posted to "zaar" ,i decided to call this one  #412 broccoli casserole.i don't think there are any like this one in the database. i based this one on the famous "green bean casserole" from campbell's soup. but i think mine is better since i don't like cream of mushroom soup.submitted to "zaar" on may 28th,2008 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground black pepper', 'salt', 'milk', 'soy sauce', 'french-fried onions']          |               9 |            5 |      194.8 |          20 |       6 |       32 |        22 |              36 |               3 |

### Univariate Analysis
We used **histogram** to analyze the relevant variables for univariate analysis. Here is an example histogram for the column `n_steps`:

<iframe
  src="assets/n_steps_hist.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


We can see that `n_steps` is right-skewed. 

Similar EDA are done for the columns `n_ingredients`, `avg_rating`, `minutes`, `calories` and `protein`. Here is an example of `avg_rating`.

<iframe
  src="assets/avg_rating_hist.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

For `avg_rating` we see that it is left skewed with most recipes having a rating of 4 or 5.

Also, Since `minutes` is highly right-skewed, the histogram is hard to read. We log-transformed it and created the following histogram.

<iframe
  src="assets/log_minutes_hist.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

From the histogram we see that `log_minutes` is still right-skewed but more symmetric than `minutes`. We see a unimodal distribution, with a mode around 3.5, which is around 33 minutes. Most of the recipes take at least one or two minutes and at most three days.

### Bivariate Analysis

For bivariate analysis, we analyzed the relation between features we are interested in and the average rating of a recipe, using **boxplots**. An example boxplot between `avg_rating` and `n_steps` is as follows, showing the distribution of `avg_rating` across four groups of `n_steps` divided by quartiles.

<iframe
  src="assets/box1_hist.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Just from the boxplot, the distribution of `avg_rating` across four quartile group of `n_steps` look very similar. We did the same for `n_ingredients` and `minutes`, and found similar results.

### Aggregate Analysis

To look more into bivariate patterns that are hard to see in the plots, we made some grouped tables to analyze aggregate statistics. Here is an example where we grouped **recipes** by the quartile groups of `n_steps` defined earlier for the boxplot, and looked at the statistics (mean, median, count, standard deviation) of `avg_rating` within each group. 


| steps_bins   |   mean |   median |   count |   std |
|:-------------|-------:|---------:|--------:|------:|
| Q1           |  4.631 |        5 |   25070 | 0.629 |
| Q2           |  4.621 |        5 |   19740 | 0.632 |
| Q3           |  4.616 |        5 |   18477 | 0.645 |
| Q4           |  4.631 |        5 |   17885 | 0.662 |

We see that across the quartile groups, the center statistics (mean and median) are pretty consistent and we see no obvious pattern. However, we see that from the first group to the fourth group, the standard deviation of `avg_rating` within the group increases. This hints that there might be an association between the value of `n_steps` and the standard deviation of `avg_ratings`. Is it true that recipes with more steps receive more polarized ratings? We will look into this through a hypothesis test.


## Assessment of Missingness

Before we head into hypothesis testing, let's assess the missingness of **recipes**. In **recipes**, three columns: `name` and `description` and `avg_rating` have missing data. `avg_rating` has the highest number of missing entries. First, we wanted to examine if any of the columns with missing values could be NMAR (not missing at random, or missingness in a column depends on the values in that column). We could make educated guesses of how the missingness might depend on other columns, thereby ruling out NMAR. For `avg_rating`, it's missingness could depend on a lot of columns. For instance, some recipes are not rated because nobody bothered to try them, and this might be because the recipes is too complicated (missingess depends on `n_steps`), because it takes too long (missingess depends on `minutes`), or because the food isn't appealing (missingess depends on `ingredients`). For `description`, there isn't a good reason why it would be missing because of what the description would have been. Probably some recipe providers just didn't bother writing the description. This also makes NMAR not very plausible for `description`. Finally, only one entry of `name` is missing. This is just an arbitrary instance of missing data, so it is probably not NMAR either (likely MCAR, but we cannot say for sure without statistical testing). Therefore, we were convinced that none of the three columns are likely to be NMAR.

Moving away from the possibility of NMAR, we wanted to test if the columns are MAR or MCAR, or in other words, does the missingness of a column depend on other columns. We chose to analyze `avg_rating` because it has the most number of missing entries, and its missingess would have a significant impact on our analyses.

In order to determine if the missingness of `avg_rating` is MAR or MCAR, we used permutation test to see if the distribution of another column differs depending on whether `avg_rating` is missing. One of the columns we tested was `n_steps`. We observed that the mean `n_steps` for recipes where `avg_rating` is not missing is 10.058961, and the mean `n_steps` for recipes where `avg_rating` is missing turns out to be 11.551936. Here, recipes with missing ratings have different mean number of steps then those that have ratings. Then, we ran a permutation test by shuffling the labels of whether `avg_rating` is missing to see if the observed difference might be produced by random chance. It turned out that, with a statistically significant p-value of 0.0, the observed difference was very unlikely to be produced by random chance (see graph below). Therefore, recipes with missing ratings indeed have different mean number of steps then those where ratings are not missing, which shows that the distribution of recipes' number of steps differ for recipes with and without missing `avg_rating`. Therefore, we conclude that the missingness of `avg_rating` depends on `n_steps`, and that the missingness of `avg_rating` is MAR since it depends on at least one other column in the dataset.

<iframe
  src="assets/miss_fig1.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

We also conducted the same test on other columns. The same missingness permutation test produced a p-value of 0.194 for the column `protein`, showing that the missingness of `avg_rating` **does not depend** on `protein`. 

<iframe
  src="assets/miss_fig2.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

## Hypothesis Testing

Earlier at the end of the EDA section, we asked the question **"Do recipes with more steps receive more polarized ratings?"**
This motivated the hypothesis we wanted to test: 

**Null Hypothesis: The standard deviation of average ratings of recipes with a number of steps greater than the median is the same as that of recipes with a number of steps less than or equal to the median. Any observed difference is due to random chance.**

**Alternative Hypothesis: The standard deviation of average ratings of recipes with a number of steps greater than the median is higher than that of recipes with a number of steps less than or equal to the median.**

**Test statistic: (Standard deviation of the average ratings of recipes with a number of steps greater than the median) - (Standard deviation of the average rating of recipes with a number of steps less than or equal to the median)**

Here we divided all recipes into two groups: *many steps* and *few steps* based on whether their number of steps is greater than or less than or equal to the median number of steps across all recipes. If we can show that the group with number of steps above the median has more spread out ratings, this implies that higher number of steps of a recipe is related with more polarizing user opinion.

We conducted a permutation test to test our hypothesis. Below are the results:

**observed test statistic: 0.023195841346820067**
**P-value: 0.002**

At a significant level of 0.05, we **reject** the null hypothesis. The permutation test provides evidence for our alternative hypothesis, implying that **the more steps a recipe has, the higher the standard deviation of the ratings are, which reflects that people's opinion on a recipe becomes more polarizing as the number of steps in a recipe increases**. This makes sense intuitively, since longer and more complex recipes can be more exquisite and interesting,but also more time-consuming and requires more effort. People who are interested in cooking and are willing to put in the time and effort like them a lot, but others might get bored and frustrated due to the complexity.

*Note: The permutation test was only conducted on recipes with non-NAN average ratings. While dropping rows could cause bias, we previously showed that the missingness of `avg_rating` depends on whether `n_steps` is high or low, and we are testing the spread of `avg_rating` given high or low `n_steps`, so even though some rows are dropped, the remaining datapoints in each group still roughly give a good estimate of the distribution of `avg_rating` in the corresponding group of high or low `n_steps`. In other words, the missingness mechanism was already accounted for by the grouping. Nevertheless, it is still worth pointing out that simply dropping rows, without conducting the appropriate imputations, might introduce some bias to our analyses. Specifically, the unrated complex recipes may be systematically more extreme than rated ones, potentially understating the true difference in standard deviation. Our estimation of the difference is likely smaller than the real world difference across all recipes"*

**If we use the number of steps as a proxy for the complexity of a recipe, this answers the big question we posed at the beginning: the more complex a recipe is, the more polarizing is users' perception of it.**

Below is a graph for the distribution of the simulated test statistics, with the vertica red line showing the observed test statistic from our dataset.

<iframe
  src="assets/test_fig1.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


## Framing a Prediction Problem

Now that we have answered the question of association between complexity and rating, let us move on to make some predictions of recipe ratings based on its attributes.

The prediction problem: Predict average ratings of recipes based on its features.

Type of prediction: Regression (because `avg_rating` is numerical)

Response variable: `avg_rating` is already a direct measurement of the ratings of each recipe, so we will use it as the response variable.

Evaluation metric: RMSE. This metric especially penalizes wrong predictions with a squared error, so a high prediction error gets penalized much more than a low prediction error. This is actually what we need, because the rating scale (1-5) is very narrow, and a high error in prediction is very misleading of how the recipe could be perceived. We also prefer RMSE over MSE or R<sup>2</sup> because RMSE is interpretable in the original units of the response variable.

The features we can use are any feature that is an attribute of a recipe, such `n_steps`, `n_ingredients`, `ratings`, the columns we have broken up from `nutrients`, the date `submitted`, `tags` and `description`.


## Baseline Model

For the baseline model, we will use the features `n_steps` (quantitative), `n_ingredients` (quantitative) and `minutes` (quantitative) to predict `avg_rating`. We will transform the features using StandardScaler to make the coefficients more interpretable.

Model equation:

**`avg_rating` = w<sub>0</sub> + w<sub>1</sub> * `n_steps` + w<sub>2</sub> * `n_ingredients` + w<sub>3</sub> * `minutes`**

Results:

**intercept=4.625936770036079  coefficients=[0.00509091 -0.00561886 -0.00170918]**

**Training RMSE: 0.6414118972545692 Testing RMSE: 0.6380591007634507**

Compared to the testing RMSE of the constant model (always predicting the mean of training targets): 0.6380469851545696

The model has similar training and testing performance, so it is not overfitting. However, the testing RMSE is almost the same as the testing RMSE we get from a constant model, so this baseline linear regression model is not much improvement from the constant model. Therefore, the model is not yet very good and needs improvement.

## Final Model

In this final model, we used polynomial regression. In addition to `n_steps`, `n_ingredients` and `minutes` in the baseline model, we added some of the nutrition columns, including `calories`, `total fat`, `protein`, and `carbohydrate`. This is because people's perception of a recipe may depend on whether the food is healthy and nutritious, and whether it causes them to gain weight. I chose a few of the nutrition columns that I thought were the most important nutrition elements, plus calories. I discarded the others because otherwise we would have too many features and may cause the model to overfit noise. For polynomial regression, we will transform only a subset of the features: `n_steps` and `n_ingredients` because they represent recipe complexity and were the features we cared the most about. We didn't create power and interaction terms for other features to prevent an overfitting model. We chose the best model by using GridSearchCV to choose the best hyperparameter (degrees: [1, 2, 3]) for the features that we applied polynomial transformation. For the GridSearchCV we used the scoring metric of 'neg_root_mean_squared_error' and the default 5-fold cross validation.

Model results:
From GridSearchCV, we found out that the best hyperparameter for the polynomial transformation is 2. The best model has a **Training RMSE of 0.6403710190379198** and a **Testing RMSE of 0.6373620492335419**. *We see a slight improvement from the baseline model*.


## Fairness Analysis

We evaluated whether the model perform equally well on all recipes regardless of their number of steps. We asked the question: **does the final model perform worse when it comes to recipes with many steps?**

**Group X: Recipes with number of steps greater than the median number of steps across recipes**
**Group Y: Recipes with number of steps less than or equal to the median number of steps across recipes**

Evaluation metric: RMSE

Null Hypothesis: Our model is fair. The RMSE of the final model on recipes with a number of steps greater than the median is the same as the RMSE of the final model on recipes with a number of steps less than or equal to the median.

Alternative Hypothesis: Our model is unfair. The RMSE of the final model on recipes with a number of steps greater than the median is greater than the RMSE of the final model on recipes with a number of steps less than or equal to the median.

Test statistic: (RMSE of the final model on recipes with a number of steps greater than the median - RMSE of the final model on recipes with a number of steps less than or equal to the median)

Again, we conducted a permutation test.

Test result: many_steps_RMSE: 0.6393543621494383, few_steps_RMSE: 0.6357364004317855
observed statistic: 0.6393543621494383 - 0.6357364004317855 = 0.003617961718
p-value: 0.417

At the significance level of 0.05, we fail to reject the null hypothesis. There is not enough evidence that the model performs worse on recipes with more steps, and what we observed that the model performs worse on recipes with above median number of steps is likely due to random chance. Therefore, we believe that the model performs fairly with respect to recipes with different number of steps.