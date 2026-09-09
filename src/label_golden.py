import pandas as pd
from pathlib import Path


INPUT_FILE = "data/golden_set/golden.csv"
OUTPUT_FILE = "data/golden_set/golden.csv"


INTENTS = [
    "ios_update_issue",
    "battery_issue",
    "device_performance",
    "screen_display_issue",
    "app_issue",
    "icloud_issue",
    "camera_issue",
    "connectivity_issue",
    "account_security",
    "hardware_repair",
    "product_information",
    "general_support",
]


def show_intents():

    print("\nAvailable Intents:")

    for i, intent in enumerate(INTENTS, start=1):
        print(f"{i}. {intent}")


def choose_intent(suggested):

    while True:

        show_intents()

        print(f"\nSuggested intent: {suggested}")

        choice = input(
            "\nEnter intent number "
            "(or press Enter to accept suggestion): "
        ).strip()

        if choice == "":
            return suggested

        if choice.isdigit():

            number = int(choice)

            if 1 <= number <= len(INTENTS):
                return INTENTS[number - 1]

        print("❌ Invalid choice. Try again.")


def choose_action():

    while True:

        print("\nAction:")
        print("1. auto_handle")
        print("2. escalate")

        choice = input("Choose 1 or 2: ").strip()

        if choice == "1":
            return "auto_handle"

        if choice == "2":
            return "escalate"

        print("❌ Invalid choice.")


def choose_quality():

    while True:

        print("\nHistorical Reply Quality:")
        print("1 = Very Poor")
        print("2 = Poor")
        print("3 = Acceptable")
        print("4 = Good")
        print("5 = Excellent")

        choice = input("Enter 1-5: ").strip()

        if choice in ["1", "2", "3", "4", "5"]:
            return int(choice)

        print("❌ Enter a number between 1 and 5.")


def main():

    path = Path(INPUT_FILE)

    if not path.exists():

        print(
            f"\n❌ File not found:\n{INPUT_FILE}"
        )

        return

    df = pd.read_csv(path)

    # IMPORTANT:
    # Convert empty gold columns to string dtype.
    # Otherwise pandas may treat them as float64
    # and fail when we insert text such as "app_issue".

    gold_columns = [
        "gold_intent",
        "gold_action",
        "gold_reply_quality",
        "label_notes",
    ]

    for column in gold_columns:

        if column not in df.columns:
            df[column] = ""

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
        )

    print("\n======================================")
    print(" AppleSupport Golden Set Labeler")
    print("======================================")

    print(
        f"Total examples: {len(df)}"
    )

    for index, row in df.iterrows():

        # Skip already labelled examples
        if (
            str(row["gold_intent"]).strip() != ""
        ):
            continue

        print("\n")
        print("=" * 70)

        print(
            f"Example {index + 1}/{len(df)}"
        )

        print("=" * 70)

        print("\nCUSTOMER MESSAGE:")
        print("-" * 70)
        print(row["customer_text"])

        print("\nHISTORICAL APPLE SUPPORT REPLY:")
        print("-" * 70)
        print(row["historical_reply"])

        print("\nCURRENT MODEL SUGGESTION:")
        print("-" * 70)
        print(row["suggested_intent"])

        # -------------------------
        # Human intent label
        # -------------------------

        gold_intent = choose_intent(
            row["suggested_intent"]
        )

        # -------------------------
        # Human routing label
        # -------------------------

        gold_action = choose_action()

        # -------------------------
        # Human reply quality
        # -------------------------

        gold_quality = choose_quality()

        # -------------------------
        # Notes
        # -------------------------

        print(
            "\nOptional note "
            "(press Enter if no note):"
        )

        note = input("> ").strip()

        # -------------------------
        # Save labels
        # -------------------------

        df.at[index, "gold_intent"] = str(
            gold_intent
        )

        df.at[index, "gold_action"] = str(
            gold_action
        )

        df.at[index, "gold_reply_quality"] = str(
            gold_quality
        )

        df.at[index, "label_notes"] = str(
            note
        )

        # Save after EVERY example
        df.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print("\n✅ Saved!")

        command = input(
            "\nPress Enter for next example "
            "or type q to quit: "
        ).strip().lower()

        if command == "q":

            print("\nProgress saved.")
            print(
                "Run the script again to continue."
            )

            break

    print("\n======================================")
    print("Labeling session finished")
    print("======================================")

    labelled = (
        df["gold_intent"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    print(
        f"Labelled: {labelled}/{len(df)}"
    )

    print(
        f"Remaining: {len(df) - labelled}"
    )


if __name__ == "__main__":
    main()