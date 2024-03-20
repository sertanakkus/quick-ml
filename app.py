import pandas as pd
import streamlit as st


def show_options(cols):
    st.container(height=20, border=False)
    st.selectbox(
        r"$\textsf{\large Algorithm}$",
    )
    st.container(height=20, border=False)
    st.selectbox(r"$\textsf{\large Target variable}$", cols)


def main():
    st.title("Quick Machine Learning")

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

            show_options(cols)


if __name__ == "__main__":
    main()
