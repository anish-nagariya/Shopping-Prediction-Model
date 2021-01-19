import csv
import tensorflow as tf

from sklearn.model_selection import train_test_split


# Read data in from file
def main():
    evidence = []
    labels = []

    months = dict(Jan=0, Feb=1, Mar=2, Apr=3,
                  May=4, June=5, Jul=6, Aug=7,
                  Sep=8, Oct=9, Nov=10, Dec=11)

    with open("shopping.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            evidence.append(
                [
                    int(row["Administrative"]),
                    float(row["Administrative_Duration"]),
                    int(row["Informational"]),
                    float(row["Informational_Duration"]),
                    int(row["ProductRelated"]),
                    float(row["ProductRelated_Duration"]),
                    float(row["BounceRates"]),
                    float(row["ExitRates"]),
                    float(row["PageValues"]),
                    float(row["SpecialDay"]),
                    months[row["Month"]],
                    int(row["OperatingSystems"]),
                    int(row["Browser"]),
                    int(row["Region"]),
                    int(row["TrafficType"]),
                    1 if row["VisitorType"] == "Returning_Visitor" else 0,
                    1 if row["Weekend"] == "TRUE" else 0
                ]
            )

            labels.append(1 if row["Revenue"] == "TRUE" else 0)

    # Separate data into training and testing groups
    X_training, X_testing, y_training, y_testing = train_test_split(
        evidence, labels, test_size=0.2
    )

    # Create a neural network
    model = tf.keras.models.Sequential()

    # Add a hidden layer with 8 units, with ReLU activation
    model.add(tf.keras.layers.Dense(64, input_shape=(17,), activation="relu"))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(16, activation="relu"))

    # Add output layer with 1 unit, with sigmoid activation
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    # Train neural network
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(X_training, y_training, epochs=50)

    # Evaluate how well model performs
    model.evaluate(X_testing, y_testing, verbose=2)


if __name__ == "__main__":
    main()
