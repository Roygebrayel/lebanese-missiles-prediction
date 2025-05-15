# main.py

from src.preprocess import load_and_filter_data, save_cleaned_data

if __name__ == "__main__":
    input_path = "data/raw/acled_data.csv"
    output_path = "data/processed/filtered_events.csv"

    df_clean = load_and_filter_data(input_path)
    save_cleaned_data(df_clean, output_path)

    print(f"Filtered data saved to: {output_path}")
