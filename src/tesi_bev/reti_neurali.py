"""Reti neurali (Keras).

Differenza principale rispetto alla versione originale: l'early stopping usa un set di
validazione estratto dal training, mai il test set, che resta intatto fino alla valutazione finale.
"""
from __future__ import annotations

import os

import numpy as np

from . import config as C

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ARCHITETTURE = {
    # Equivalente al "modello 1" della tesi: 4 strati densi con dropout
    "ANN base": {"strati": (16, 64, 128, 64), "dropout": 0.2, "l2": 0.0, "lr": 1e-3},
    # Rete compatta con regolarizzazione L2 + dropout (equivalente ai modelli 2-3 della tesi)
    "ANN regolarizzata": {"strati": (64, 32), "dropout": 0.3, "l2": 1e-3, "lr": 1e-3},
    # Modello avanzato della tesi ("model2_rid"): 189 feature polinomiali di grado 2, LeakyReLU,
    # regolarizzazione ElasticNet (L1 + L2), BatchNormalization sull'ultimo strato e ottimizzatore AdamW
    "ANN avanzata (tesi)": {"strati": (64, 128, 256, 128, 64), "dropout": 0.3, "l1": 1e-3, "l2": 1e-3,
                            "lr": 1e-3, "attivazione": "leaky_relu", "batch_norm": True,
                            "ottimizzatore": "adamw", "polinomiali": True, "batch_size": 32,
                            "monitor": "val_loss", "pazienza": 20, "pazienza_lr": 5},
}

# Chiavi di ARCHITETTURE che non sono argomenti di crea_rete ma impostazioni di addestramento
_IMPOSTAZIONI = ("polinomiali", "batch_size", "monitor", "pazienza", "pazienza_lr")


def crea_rete(n_input: int, strati=(64, 32), dropout=0.3, l1=0.0, l2=1e-3, lr=1e-3, attivazione="relu",
              batch_norm=False, ottimizzatore="adam", seed=C.SEED):
    import tensorflow as tf
    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    reg = keras.regularizers.l1_l2(l1=l1, l2=l2) if (l1 or l2) else None
    modello = keras.Sequential([keras.Input(shape=(n_input,))])
    for i, unita in enumerate(strati):
        modello.add(keras.layers.Dense(unita, kernel_regularizer=reg))
        modello.add(keras.layers.LeakyReLU(negative_slope=0.3) if attivazione == "leaky_relu"
                    else keras.layers.Activation(attivazione))
        if batch_norm and i == len(strati) - 1:
            modello.add(keras.layers.BatchNormalization())
        if dropout:
            modello.add(keras.layers.Dropout(dropout))
    modello.add(keras.layers.Dense(1, activation="sigmoid"))
    opt = keras.optimizers.AdamW(lr) if ottimizzatore == "adamw" else keras.optimizers.Adam(lr)
    modello.compile(optimizer=opt, loss="binary_crossentropy",
                    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return modello


class ReteNeurale:
    """Interfaccia stile scikit-learn: scaler (ed eventuali feature polinomiali) addestrati sul training + rete Keras."""

    def __init__(self, nome="ANN regolarizzata", epoche=300, batch_size=64, quota_validazione=0.2, seed=C.SEED):
        self.nome, self.epoche, self.batch_size = nome, epoche, batch_size
        self.quota_validazione, self.seed = quota_validazione, seed
        self.parametri = ARCHITETTURE[nome]

    def fit(self, X, y):
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        from tensorflow import keras

        X_tr, X_val, y_tr, y_val = train_test_split(
            np.asarray(X, dtype="float32"), np.asarray(y), test_size=self.quota_validazione,
            random_state=self.seed, stratify=y)
        p = self.parametri
        # Come nella tesi: standardizzazione e poi feature polinomiali, entrambe stimate sul solo training
        passi = [StandardScaler()] + ([PolynomialFeatures(degree=2, include_bias=False)] if p.get("polinomiali") else [])
        self.scaler_ = make_pipeline(*passi).fit(X_tr)
        X_tr_t, X_val_t = self.scaler_.transform(X_tr), self.scaler_.transform(X_val)
        self.n_feature_ = X_tr_t.shape[1]
        self.modello_ = crea_rete(self.n_feature_, seed=self.seed,
                                  **{k: v for k, v in p.items() if k not in _IMPOSTAZIONI})
        monitor = p.get("monitor", "val_auc")
        callback = [
            keras.callbacks.EarlyStopping(monitor=monitor, mode="max" if monitor == "val_auc" else "min",
                                          patience=p.get("pazienza", 20), restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=p.get("pazienza_lr", 8),
                                              min_lr=1e-5),
        ]
        storia = self.modello_.fit(
            X_tr_t, y_tr, validation_data=(X_val_t, y_val),
            epochs=self.epoche, batch_size=p.get("batch_size", self.batch_size), callbacks=callback, verbose=0)
        self.storia_ = {k: [float(v) for v in vals] for k, vals in storia.history.items()}
        return self

    def predict_proba(self, X):
        p = self.modello_.predict(self.scaler_.transform(np.asarray(X, dtype="float32")), verbose=0).ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
