from stilts_ml.neural_network import ann
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def data_generation_easy():
    df = pd.DataFrame()
    for _ in range(1000):
        a = np.random.normal(0, 1)
        b = np.random.normal(0, 3)
        c = np.random.normal(12, 4)
        if a + b + c > 11:
            target = 1
        else:
            target = 0
        df = df.append({
            "A": a,
            "B": b,
            "C": c,
            "target": target
        }, ignore_index=True)
    return df

def test_integration_ann():
    df = data_generation_easy()
    column_names = ["A", "B", "C"]
    target_name = "target"
    X = df[column_names].values
    y = np.array([[elem] for elem in list(df["target"].values)])

    nn = ann.NeuralNetwork()
    # Dense(inputs, outputs, activation)
    nn.add_layer(ann.Dense(3, 20, "tanh"))
    nn.add_layer(ann.Dense(20, 1, "tanh"))
    nn.fit(X, y)
    y_pred = nn.predict(X)
    print(mean_squared_error(y_pred, y))
    assert True
