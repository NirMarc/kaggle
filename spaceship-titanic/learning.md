### Download Data
- In kaggle you can download data using bash, creating a script makes it look more professional and it takes a minute to do.
- Keep all the data together in a different repo - easy to access and clean

### Working
- Create separate directories for each type of thing you are working with (data, notebooks, etc.)

### EDA
#### First step - UNDERSTAND the data
- Start with just .head() -> How many columns are there? What is target column? What type of columns are there? Any col looks weird?
- train.info() -> Total number of rows, real type of cols, missing values inside rows (Which later we will dive deeper)
- df_train[TARGET].value_counts() -> How True/False? Balanced/Imbalanced? Does it matter?
- Check missing values- How much is missing?

#### Second step - EXPLORE patterns
- Pick a feature:
    - Before running any code! Make a hypothesis - Do you expect this feature to affect your target label? Why?
    - Check distribution (histogram at first) - What shape do you see? Any outliers?
    - Check correlation of feature by Target
    - Check Target rate by feature (groupby and then valuecount)
- Try to see if some features have a correlation between them, maybe they tell a similar story, maybe their stories complete each other.
- Some features can be combined together
- Always think if maybe I'm spending too much time exploring features without enough gain

##### Handling missing values
- ALWAYS keep training stats when applying to test set. We are trying to predict the future using the past- If I had only one test row coming each time I would have to use only my training stats without questions, same even when they are coming as a full data set.
- To fill missing values we have a lot of options: Median, Mean, True/False, random based on distribution, "Unknown" and probably many more, the trick is to find what suits each feature best (maybe some features could tell us how to handle missing values for other features - "if someone spent money he's not in cryosleep..")


### Modeling
- Start with processing based on EDA
- Do a data audit - what columns do I have? what are their types? It's going to affect directly how I normalize and encode the features.
- Always use drop_first=True -> you need k-1 binary cols to represent k categories!
- Always align at the end -> Sometimes there are missing categories at the test set (like "unknown" for something), align makes sure you will have the same number of features in test as in train.
- Pretty easy trying models when just calling their API. It seems after trying all these models + grid search that the most important thing is good feature engineering.