import numpy as np
import gradio as gr
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Charger le dataset MNIST
mnist = fetch_openml('mnist_784', version=1, parser="auto")
X = mnist.data.astype(np.float32)  # Convertir en float32 pour éviter les erreurs
y = mnist.target.astype(int)  # Convertir en entiers

# Diviser les données en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraîner un modèle Random Forest (plus rapide que SVM)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Fonction de prédiction
def predict_digit(image):
    try:
        # Convertir le tableau NumPy en objet PIL
        image = Image.fromarray(image).convert("L").resize((28, 28))

        # Transformer en tableau numpy
        image_array = np.array(image)

        # Inverser les couleurs (Gradio utilise fond blanc et traits noirs)
        image_array = 255 - image_array

        # Aplatir et normaliser
        image_array = image_array.flatten().reshape(1, -1).astype(np.float32)
        image_array = scaler.transform(image_array)  # Appliquer le scaler

        # Prédiction du modèle
        prediction = model.predict(image_array)[0]
        return str(prediction)
    except Exception as e:
        return f"Erreur: {str(e)}"

# Interface Gradio
iface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(image_mode="L"),
    outputs="text",
    title="Reconnaissance de Chiffres Manuscrits",
    description="Dessinez un chiffre entre 0 et 9 sur le canevas. Le modèle prédit votre chiffre."
)

iface.launch()
