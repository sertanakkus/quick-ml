import pandas as pd
import streamlit as st

from apply_algorithm import Algorithms


def get_options(cols):
    st.container(height=10, border=False)
    algorithm = st.selectbox(
        r"$\textsf{\large Algorithm}$", list(algorithm_list.keys())
    )

    st.container(height=10, border=False)
    target = st.selectbox(r"$\textsf{\large Target variable}$", cols)

    return algorithm, target


def main():
    st.title("Quick Machine Learning")
    st.container(height=10, border=False)

    file = st.file_uploader(r"$\textsf{\large Upload a csv file}$", type="csv")

    if file is not None:
        with open("temp.csv", "wb") as f, open("temp.csv", "r") as f_in, open(
            "temp.csv", "w"
        ) as f_out:
            f.write(file.read())
            context = f_in.read()
            new_context = context.replace(";", ",").replace("'", "").replace('"', "")
            f_out.write(new_context)

        df = pd.read_csv("temp.csv")

        cols = df.columns

        algorithm, target = get_options(cols)

        if st.button("Start"):
            result = Algorithms.apply(algorithm, target)

            st.write(result)


algorithm_list = {
    "Linear Regression": 0,
    "Logistic Regression": 1,
    "Decision Tree Regressor": 2,
    "Random Forest Regressor": 3,
    "KNN": 4,
    "SVR": 5,
    "SVC": 6,
}

if __name__ == "__main__":
    main()
