import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
with open("labels.txt", "r") as f:
    class_names = f.readlines()

# Define the precautions for each skin condition
precautions = {
    '0 Acne': """
        1. Gentle Cleansing: Wash your face twice daily with a gentle, non-comedogenic cleanser.
        2. Avoid Touching Your Face: Keep your hands away from your face to prevent transferring dirt and bacteria.
        3. Use Non-Comedogenic Products: Choose oil-free and non-comedogenic makeup, sunscreen, and skincare products.
    """,
    '1 Allergic reactions': """
        1. Identify and Avoid Allergens: Understand your triggers through allergy tests and avoid exposure.
        2. Use Medications as Directed: Take antihistamines or nasal sprays to reduce symptoms.
        3. Personal Hygiene and Protection: Wash your hands and shower after outdoor activities to remove allergens.
    """,
    '2 Cosmetic disorder': """
        1. Skin Care Precautions: Use a mild, non-irritating cleanser suitable for your skin type.
        2. Avoid Skin Irritants: Be cautious with products containing alcohol or harsh chemicals.
        3. Diet and Hydration: Eat a balanced diet and stay hydrated to support skin health.
    """,
    '3 Genetics': """
        1. Genetic Counseling: Consult a genetic counselor if you have a family history of genetic disorders.
        2. Healthy Lifestyle Choices: Maintain a healthy lifestyle with regular exercise.
        3. Manage Stress and Mental Health: Practice stress management techniques like yoga or meditation.
    """,
    '4 Hives': """
        1. Identify and Avoid Triggers: Avoid allergens, medications, and stress that may trigger hives.
        2. Use Antihistamines: Take antihistamines to alleviate symptoms as recommended by your doctor.
        3. Avoid Scratching: Resist scratching to prevent skin damage. Use soothing treatments like hydrocortisone cream.
    """,
    '5 Hyperpigmentation': """
        1. Treat Skin Gently: Avoid harsh scrubs and opt for gentle exfoliants.
        2. Use Brightening and Lightening Agents: Use products with Vitamin C or Niacinamide.
        3. Exfoliate with Care: Use gentle exfoliation methods and apply sunscreen daily.
    """,
    '6 Infections': """
        1. Hand Hygiene: Wash your hands frequently and avoid touching your face.
        2. Vaccination: Keep up-to-date with recommended vaccinations.
        3. Personal Protective Equipment (PPE): Wear appropriate protective gear in high-risk environments.
    """,
    '7 Psoriosis': """
        1. Moisturize Regularly: Use thick, fragrance-free moisturizers to keep skin hydrated.
        2. Stay Hydrated: Drink plenty of water to maintain skin hydration.
        3. Be Cautious with Alcohol and Smoking: Limit alcohol and avoid smoking to prevent flare-ups.
    """,
    '8 Skin cancer': """
        1. Dress Appropriately: Wear protective clothing to shield your skin from the sun.
        2. Avoid Tanning Beds: Do not use tanning beds as they increase the risk of skin cancer.
        3. Consider Vitamin D Sources Other than Sunlight: Get vitamin D from food or supplements.
    """,
    '9 Sun damage': """
        1. Use Sunscreen Regularly: Apply sunscreen daily with SPF 30 or higher.
        2. Seek Shade: Avoid direct sun exposure during peak hours (10 a.m. to 4 p.m.).
        3. Be Mindful of Reflective Surfaces: Avoid reflective surfaces that intensify UV exposure.
    """
}

# Streamlit app
st.title('Skin Disease Detection')
st.write('Upload an image to get the classification result.')

# Upload image via Streamlit
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open the uploaded image
    image = Image.open(uploaded_image).convert("RGB")
    
    # Resize the image to 224x224
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Prepare the image for prediction
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image_array = np.asarray(image)
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Make a prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    
    # Display prediction results
    st.write(f"Class: {class_name}")
    st.write(f"Confidence Score: {confidence_score:.2f}")

    # Display precautions based on the predicted class
    if class_name in precautions:
        st.subheader('Precautions:')
        st.write(precautions[class_name])
    else:
        st.write("No precautions available for this condition.")
