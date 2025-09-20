import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2
# Fetch variables from Streamlit secrets
USER = st.secrets["DB_USER"] #os.getenv("user")
PASSWORD = st.secrets["DB_PASSWORD"]# os.getenv("password")
HOST = st.secrets["DB_HOST"] #os.getenv("host")
PORT = st.secrets["DB_PORT"] #os.getenv("port")
DBNAME = st.secrets["DB_NAME"] #os.getenv("dbname")

# Configuración de la página
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")

# Función para insertar datos
def save_prediction(sepal_l, sepal_w, petal_l, petal_w, pred):
    try:
        conn = psycopg2.connect(user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME)
        cur = conn.cursor()
        cur.execute("INSERT INTO public.table_iris (longitud_sepalo, ancho_sepalo, longitud_petalo, ancho_petalo, prediction) VALUES (%s, %s, %s, %s, %s)", (sepal_l, sepal_w, petal_l, petal_w, pred))
        conn.commit()
        cur.close()
        conn.close()
        return True, (sepal_l, sepal_w, petal_l, petal_w, pred)
    except:
        return False, None

# Connect to the database
try:
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )
    print("Connection successful!")
    
    # Create a cursor to execute SQL queries
    cursor = connection.cursor()
    
    # Example query
    cursor.execute("SELECT NOW();")
    result = cursor.fetchone()
    print("Current Time:", result)
    # Close the cursor and connection
    cursor.close()
    connection.close()
    print("Connection closed.")

except Exception as e:
    st.write(str(e))



# Función para cargar los modelos
@st.cache_resource
def load_models():
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo en la carpeta 'models/'")
        return None, None, None

# Título
st.title("🌸 Predictor de Especies de Iris")

# Cargar modelos
model, scaler, model_info = load_models()

if model is not None:
    # Inputs
    st.header("Ingresa las características de la flor:")
    st.write(result)
    
    sepal_length = st.number_input("Longitud del Sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width = st.number_input("Ancho del Sépalo (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    petal_length = st.number_input("Longitud del Pétalo (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    petal_width = st.number_input("Ancho del Pétalo (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    # Botón de predicción
    if st.button("Predecir Especie"):
        # Preparar datos
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Estandarizar
        features_scaled = scaler.transform(features)
        
        # Predecir
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Mostrar resultado
        target_names = model_info['target_names']
        predicted_species = target_names[prediction]
        
        st.success(f"Especie predicha: **{predicted_species}**")
        st.write(f"Confianza: **{max(probabilities):.1%}**")
        
        # Mostrar todas las probabilidades
        st.write("Probabilidades:")
        for species, prob in zip(target_names, probabilities):
            st.write(f"- {species}: {prob:.1%}")
        
        # Guardar en base de datos y mostrar confirmacion
        success, data_saved = save_prediction(sepal_length, sepal_width, petal_length, petal_width, predicted_species)
        if success and data_saved:
            st.info(f"Datos cargados correctamente: {data_saved}")