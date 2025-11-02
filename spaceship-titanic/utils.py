import pandas as pd
import numpy as np

def prepare_data(df: pd.DataFrame, train_stats: dict = None, is_train: bool = True) -> pd.DataFrame:
    np.random.seed(42)
    df = df.copy(deep=True)
    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

    if is_train:
        train_stats = {
            'age_median': df["Age"].median(),
            "spending_medians": {col: df[col].median() for col in spending_cols},
            "cryo_dist": df["CryoSleep"].value_counts(normalize=True)
        }

    df["Age"] = df["Age"].fillna(train_stats["age_median"])
    df['Age_Group'] = pd.cut(df["Age"],
                             bins=[0, 12, 18, 30, 50, 100],
                             labels=["Child", "Teen", "Young_Adult", "Adult", "Senior"],
                             include_lowest=True)
    for col in spending_cols:
        df[col] = df[col].fillna(train_stats["spending_medians"][col])
    df["Total_Spendings"] = df[spending_cols].sum(axis=1)
    df["Is_Spender"] = (df["Total_Spendings"] > 0).astype(int)
    df.loc[df["Is_Spender"] == 1, "CryoSleep"] = df.loc[df["Is_Spender"] == 1, "CryoSleep"].fillna(False)
    df_missing_cryo = df["CryoSleep"].isnull()
    df.loc[df_missing_cryo, "CryoSleep"] = np.random.choice(train_stats["cryo_dist"].index,
                                                            size=df_missing_cryo.sum(),
                                                            p=train_stats["cryo_dist"].values
                                                            )

    df["Cabin_Deck"] = df["Cabin"].str.split("/").str[0]
    df["Cabin_Side"] = df["Cabin"].str.split("/").str[2]
    df["Cabin_Deck"] = df["Cabin_Deck"].fillna("Unknown")
    df["Cabin_Side"] = df["Cabin_Side"].fillna("Unknown")

    df["Group"] = df["PassengerId"].str.split("_").str[0]
    df["Group_Size"] = df.groupby("Group")["PassengerId"].transform("count")

    df["VIP"] = df["VIP"].fillna(False)
    df["HomePlanet"] = df["HomePlanet"].fillna("Unknown")
    df["Destination"] = df["Destination"].fillna("Unknown")
    
    df.drop(["Cabin", "Age", "PassengerId", "Name", "Group"], axis=1, inplace=True)

    if is_train:
        return df, train_stats
    else:
        return df