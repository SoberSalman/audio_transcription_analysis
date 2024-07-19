import streamlit as st
import pandas as pd
import os
import difflib
import re

def load_data():
    data = pd.read_csv('Finetuning vs Not_Finetuning Results.csv')
    return data

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation from the beginning and end of the sentence only
    text = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', text)  # Removes non-word and non-space characters from start and end
    # Split the text into words
    words = text.split()
    return words


def style_text(original, transcription):
    original_words = preprocess_text(original)
    transcription_words = preprocess_text(transcription)
    styled_text = []

    s = difflib.SequenceMatcher(None, original_words, transcription_words)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'equal':
            styled_text.append(f"<span style='color: green;'>{' '.join(transcription_words[j1:j2])}</span>")
        else:
            if tag in ['replace', 'delete']:
                styled_text.append(f"<span style='color: red;'>{' '.join(transcription_words[j1:j2])}</span>")
            if tag == 'insert':
                styled_text.append(f"<span style='color: red;'>{' '.join(transcription_words[j1:j2])}</span>")

    return ' '.join(styled_text)

def main():
    st.set_page_config(layout="wide")
    st.title("Audio Transcription Analysis")

    data = load_data()

    st.header('Transcription Comparisons')

    # Adjust the number of columns based on new headers
    header_cols = st.columns([1, 3, 3, 3, 1, 1, 1, 1])  # Adjusted for new metrics
    headers = ["Play Audio", "Ground Truth", "Before Finetuning Transcription", "Finetuned Transcription",
               "BF_CER", "AF_CER", "BF_WER", "AF_WER"]
    for i, header in enumerate(headers):
        header_cols[i].write(f"**{header}**")

    # Custom CSS to change text color
    st.markdown("""
    <style>
    .white-text {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    # Data rows in an expander
    expander = st.expander("Show Transcriptions", expanded=True)
    for index, row in data.iterrows():
        with expander:
            cols = st.columns([1, 3, 3, 3, 1, 1, 1, 1])  # Adjusted for new metrics
            audio_path = os.path.join('Data Collection', row['audio_path']) if pd.notna(row['audio_path']) else "No audio file"
            if audio_path != "No audio file":
                cols[0].audio(audio_path, format='audio/wav')
            else:
                cols[0].write("No audio file")

            # Displaying other text data
            cols[1].write(row['ground_truth'])
            cols[2].markdown(style_text(row['ground_truth'], row['Before_Finetuning_transcription']), unsafe_allow_html=True)  # Styled Before Finetuning
            cols[3].markdown(style_text(row['ground_truth'], row['Finetuned_transcription']), unsafe_allow_html=True)  # Styled Finetuned


            # Displaying new metrics data with white text color
            cols[4].markdown(f"<span class='white-text'>{row.get('BF_CER', 'N/A')}</span>", unsafe_allow_html=True)
            cols[5].markdown(f"<span class='white-text'>{row.get('AF_CER', 'N/A')}</span>", unsafe_allow_html=True)
            cols[6].markdown(f"<span class='white-text'>{row.get('BF_WER', 'N/A')}</span>", unsafe_allow_html=True)
            cols[7].markdown(f"<span class='white-text'>{row.get('AF_WER', 'N/A')}</span>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
