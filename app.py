import streamlit as st
import numpy as np
import tensorflow as tf
import re
from transformers import BertTokenizer

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = tf.saved_model.load('Users/Peach/OneDrive/Desktop/modelbert300000/modelbert')  
    return tokenizer, model


def main():
    st.title("Welcome to Phish-Off!")
    tokenizer, model = get_model()
    user_input = st.text_area('Enter your URL here...')
    button = st.button("Analyze")

    d = {
        0: 'Safe',
        1: 'Unsafe'
    }

    # Preprocessing function
    def preprocess_link(link):
        # Convert the link to lowercase
        processed_link = link.lower()

        # Apply necessary transformations or feature engineering based on the existing code
        processed_link = re.sub(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '', processed_link)
        processed_link = re.sub(r'^https?:\/\/', '', processed_link)
        processed_link = re.sub(r'www\.', '', processed_link)
        processed_link = re.sub(r'\.(com|org|net|mil|edu|COM|ORG|NET|MIL|EDU)$', '', processed_link)

        return processed_link

    if user_input and button:
        input = user_input
        user_input = preprocess_link(input)

        # Tokenize the input
        inputs = tokenizer.encode_plus(
            user_input,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )

        # Convert input tensors to numpy arrays
        input_ids = inputs['input_ids'].numpy()
        attention_mask = inputs['attention_mask'].numpy()
        token_type_ids = inputs['token_type_ids'].numpy()

        # Get the signature_def
        signature_def = model.signatures["serving_default"]

        # Run the model prediction
        output = signature_def(
            input_ids=tf.constant(input_ids),
            attention_mask=tf.constant(attention_mask),
            token_type_ids=tf.constant(token_type_ids)
        )

        # Print the output tensor keys
        print(output.keys())

        # Perform the prediction using the correct output key
        y_pred = np.argmax(output['logits'].numpy(), axis=1)
        st.write("Prediction:", d[y_pred[0]])


if __name__ == "__main__":
    main()
