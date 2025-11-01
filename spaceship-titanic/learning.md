### Download Data
- In kaggle you can download data using bash, creating a script makes it look more professional and it takes a minute to do.
- Keep the all the data together in a different repo- easy to access and clean

### Working
- Create seprate directories for each type of thing you are working with (data, notebooks, etc..)

### EDA
#### First step- UNDERSTAND the data
- Start with just .head() -> How many colums are there? What is target coloum? What type of columns are there? Any col looks weird?
- train.info() -> Total number of rows, real type of cols, missing values inside rows (Which later we will dive deeper)
- df_train[TARGET].value_counts() -> How True/False? Balanced/Imbalanced? Does it matter?
- Check missing values- How much is missing?