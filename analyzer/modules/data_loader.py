import pandas as pd
import os

DATA_PATH = os.path.join("data", "final_cleaned.csv")

def load_bookings(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the cleaned flight bookings dataset.
    Performs light preprocessing on critical categorical fields.
    """
    try:
        if not os.path.exists(filepath):
            print(f"⚠️ File not found: {filepath}")
            return pd.DataFrame()

        df = pd.read_csv(filepath)

        # ✅ Convert Departure_Date to datetime
        if "Departure_Date" in df.columns:
            df["Departure_Date"] = pd.to_datetime(df["Departure_Date"], errors="coerce")
            df["Month"] = df["Departure_Date"].dt.month
            df["Year"] = df["Departure_Date"].dt.year
            df["Weekday"] = df["Departure_Date"].dt.day_name()

        # ✅ Normalize Booking_Status (critical categorical field)
        if "Booking_Status" in df.columns:
            df["Booking_Status"] = (
                df["Booking_Status"]
                .astype(str)
                .str.strip()
                .str.title()  # Confirmed, Cancelled, Pending
            )

        # ✅ Normalize Travel_Class (Economy, Business, First)
        if "Travel_Class" in df.columns:
            df["Travel_Class"] = (
                df["Travel_Class"]
                .astype(str)
                .str.strip()
                .str.title()
            )

        return df

    except Exception as e:
        print(f"❌ Error loading bookings data: {e}")
        return pd.DataFrame()
