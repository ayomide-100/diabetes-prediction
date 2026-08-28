import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def dataset_overview(df):
    """
    Displays the basic structure of the dataset.
    """

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    (df.head())

    print("\nDescriptive statistics:")
    (df.describe())






def missing_value_analysis(df):
    """
    Calculates and visualizes missing values in the dataset.
    """

    missing = df.isnull().sum()
    missing_percentage = (missing / len(df)) * 100

    result = pd.DataFrame({
        "Missing Values": missing,
        "Percentage": missing_percentage
    })

    result = result.sort_values("Missing Values", ascending=False)

    print("=" * 60)
    print("MISSING VALUE ANALYSIS")
    print("=" * 60)

    (result)

    if missing.sum() == 0:
        print("\nConclusion: No explicit missing values were found.")
    else:
        print("\nConclusion: Missing values are present and should be investigated.")






def zero_value_analysis(df):
    """
    Identifies zero values in numerical variables.
    """

    numeric_columns = df.select_dtypes(include=np.number).columns

    zero_counts = (df[numeric_columns] == 0).sum()
    zero_percentage = (zero_counts / len(df)) * 100

    result = pd.DataFrame({
        "Zero Values": zero_counts,
        "Percentage": zero_percentage
    })

    result = result.sort_values("Zero Values", ascending=False)

    print("=" * 60)
    print("ZERO VALUE ANALYSIS")
    print("=" * 60)

    (result)

    return result





def target_distribution(df, target="Outcome"):
    """
    Displays the class distribution of the target variable.
    """

    counts = df[target].value_counts().sort_index()
    percentages = df[target].value_counts(normalize=True).sort_index() * 100

    result = pd.DataFrame({
        "Count": counts,
        "Percentage": percentages
    })

    print("=" * 60)
    print("TARGET DISTRIBUTION")
    print("=" * 60)

    (result)

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x=target)
    plt.title("Distribution of Diabetes Outcome")
    plt.xlabel("Diabetes Outcome")
    plt.ylabel("Number of Patients")
    plt.show()

    return result






def numerical_summary(df):
    """
    Generates an extended numerical summary.
    """

    numeric_columns = df.select_dtypes(include=np.number).columns

    summary = pd.DataFrame({
        "Mean": df[numeric_columns].mean(),
        "Median": df[numeric_columns].median(),
        "Std": df[numeric_columns].std(),
        "Minimum": df[numeric_columns].min(),
        "Maximum": df[numeric_columns].max(),
        "Skewness": df[numeric_columns].skew()
    })

    (summary.round(3))

    return summary




def plot_feature_distributions(df, target="Outcome"):
    """
    Plots histograms and KDE curves for numerical variables.
    """

    features = df.select_dtypes(include=np.number).columns
    features = features.drop(target, errors="ignore")

    for feature in features:

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x=feature,
            kde=True
        )

        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()





def plot_boxplots(df, target="Outcome"):
    """
    Creates boxplots for numerical variables to identify potential outliers.
    """

    features = df.select_dtypes(include=np.number).columns
    features = features.drop(target, errors="ignore")

    for feature in features:

        plt.figure(figsize=(8, 4))

        sns.boxplot(
            data=df,
            x=feature
        )

        plt.title(f"Boxplot of {feature}")
        plt.xlabel(feature)
        plt.show()






def calculate_outliers(df, target="Outcome"):
    """
    Detects potential outliers using the IQR method.
    """

    features = df.select_dtypes(include=np.number).columns
    features = features.drop(target, errors="ignore")

    results = []

    for feature in features:

        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df[
            (df[feature] < lower_bound) |
            (df[feature] > upper_bound)
        ]

        results.append({
            "Feature": feature,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Outliers": len(outliers),
            "Outlier Percentage": len(outliers) / len(df) * 100
        })

    result = pd.DataFrame(results)

    (result.round(3))

    return result








def correlation_analysis(df):
    """
    Calculates and visualizes correlations between numerical variables.
    """

    correlation_matrix = df.corr(numeric_only=True)

    print("=" * 60)
    print("CORRELATION MATRIX")
    print("=" * 60)

    (correlation_matrix.round(3))

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title("Correlation Matrix")
    plt.show()

    return correlation_matrix






def compare_features_by_target(df, target="Outcome"):
    """
    Compares the mean values of each numerical feature
    between target classes.
    """

    features = df.select_dtypes(include=np.number).columns
    features = features.drop(target, errors="ignore")

    comparison = df.groupby(target)[features].mean().T

    comparison.columns = [
        f"{target}=0",
        f"{target}=1"
    ]

    comparison["Difference"] = (
        comparison[f"{target}=1"] -
        comparison[f"{target}=0"]
    )

    (comparison.round(3))

    return comparison










def plot_features_by_target(df, target="Outcome"):
    """
    Compares feature distributions between target classes.
    """

    features = df.select_dtypes(include=np.number).columns
    features = features.drop(target, errors="ignore")

    for feature in features:

        plt.figure(figsize=(8, 5))

        sns.boxplot(
            data=df,
            x=target,
            y=feature
        )

        plt.title(f"{feature} by Diabetes Outcome")
        plt.xlabel("Diabetes Outcome")
        plt.ylabel(feature)

        plt.show()






def glucose_outcome_analysis(df, target="Outcome"):
    """
    Examines diabetes prevalence across glucose ranges.
    """

    bins = [0, 100, 126, 140, 200]

    labels = [
        "≤100",
        "101–126",
        "127–140",
        ">140"
    ]

    temp = df.copy()

    temp["Glucose_Range"] = pd.cut(
        temp["Glucose"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    result = (
        temp.groupby("Glucose_Range", observed=False)[target]
        .agg(["count", "mean"])
    )

    result["Diabetes Percentage"] = result["mean"] * 100

    (result.round(3))

    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=temp,
        x="Glucose_Range",
        y=target
    )

    plt.title("Diabetes Rate Across Glucose Ranges")
    plt.xlabel("Glucose Range")
    plt.ylabel("Diabetes Rate")

    plt.show()

    return result






def bmi_outcome_analysis(df, target="Outcome"):
    """
    Examines diabetes prevalence across BMI categories.
    """

    bins = [0, 18.5, 25, 30, 35, 100]

    labels = [
        "Underweight",
        "Normal",
        "Overweight",
        "Obesity I",
        "Obesity II+"
    ]

    temp = df.copy()

    temp["BMI_Category"] = pd.cut(
        temp["BMI"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    result = (
        temp.groupby("BMI_Category", observed=False)[target]
        .agg(["count", "mean"])
    )

    result["Diabetes Percentage"] = result["mean"] * 100

    (result.round(3))

    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=temp,
        x="BMI_Category",
        y=target
    )

    plt.title("Diabetes Rate Across BMI Categories")
    plt.xlabel("BMI Category")
    plt.ylabel("Diabetes Rate")
    plt.xticks(rotation=30)

    plt.show()

    return result






def age_outcome_analysis(df, target="Outcome"):
    """
    Examines diabetes prevalence across age groups.
    """

    bins = [0, 25, 35, 45, 55, 100]

    labels = [
        "≤25",
        "26–35",
        "36–45",
        "46–55",
        "56+"
    ]

    temp = df.copy()

    temp["Age_Group"] = pd.cut(
        temp["Age"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    result = (
        temp.groupby("Age_Group", observed=False)[target]
        .agg(["count", "mean"])
    )

    result["Diabetes Percentage"] = result["mean"] * 100

    (result.round(3))

    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=temp,
        x="Age_Group",
        y=target
    )

    plt.title("Diabetes Rate Across Age Groups")
    plt.xlabel("Age Group")
    plt.ylabel("Diabetes Rate")

    plt.show()

    return result







def pairwise_relationships(df, target="Outcome"):
    """
    Creates a pairplot to examine relationships between
    important variables and the target.
    """

    selected_features = [
        "Glucose",
        "BMI",
        "Age",
        "Pregnancies",
        target
    ]

    sns.pairplot(
        df[selected_features],
        hue=target,
        diag_kind="hist"
    )

    plt.show()





def run_complete_eda(df, target="Outcome"):
    """
    Runs the complete exploratory data analysis.
    """

    print("\n\n")
    print("#" * 70)
    print("COMPLETE EXPLORATORY DATA ANALYSIS")
    print("#" * 70)

    print("\n1. DATASET OVERVIEW")
    dataset_overview(df)

    print("\n2. MISSING VALUE ANALYSIS")
    missing_value_analysis(df)

    print("\n3. ZERO VALUE ANALYSIS")
    zero_value_analysis(df)

    print("\n4. NUMERICAL SUMMARY")
    numerical_summary(df)

    print("\n5. TARGET DISTRIBUTION")
    target_distribution(df, target)

    print("\n6. OUTLIER ANALYSIS")
    calculate_outliers(df, target)

    print("\n7. CORRELATION ANALYSIS")
    correlation_analysis(df)


    print("\n9. COMPARISON BY TARGET")
    compare_features_by_target(df, target)

    print("\n10. FEATURE DISTRIBUTIONS")
    plot_feature_distributions(df, target)

    print("\n11. FEATURE DISTRIBUTIONS BY TARGET")
    plot_features_by_target(df, target)

    print("\n12. GLUCOSE ANALYSIS")
    glucose_outcome_analysis(df, target)

    print("\n13. BMI ANALYSIS")
    bmi_outcome_analysis(df, target)

    print("\n14. AGE ANALYSIS")
    age_outcome_analysis(df, target)

    print("\n15. PAIRWISE RELATIONSHIPS")
    pairwise_relationships(df, target)

