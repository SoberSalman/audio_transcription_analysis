import streamlit as st
import pandas as pd
import os
import difflib
import re
import editdistance
import string

def calculate_cer(ground_truth, transcription):
    # Normalize and remove punctuation
    ground_truth = ground_truth.strip().lower().strip(string.punctuation)
    transcription = transcription.strip().lower().strip(string.punctuation)

    ground_truth = re.sub(r'[^\w\s]', '', ground_truth)  # \w matches any alphanumeric character, \s matches any whitespace character
    transcription = re.sub(r'[^\w\s]', '', transcription)  # \w matches any alphanumeric character, \s matches any whitespace character
    
    # print(f"Normalized ground truth: '{ground_truth}' (length: {len(ground_truth)})")
    # print(f"Normalized transcription: '{transcription}' (length: {len(transcription)})")
    
    # Calculate edit distance at character level
    distance = editdistance.eval(ground_truth, transcription)
    
    # Calculate CER
    cer = distance / len(ground_truth) if len(ground_truth) > 0 else float('inf')
    
    return cer


def calculate_wer(reference, hypothesis):
  
    reference = reference.lower().strip(string.punctuation).strip()
    hypothesis = hypothesis.lower().strip(string.punctuation).strip()

    reference = re.sub(r'[^\w\s]', '', reference) 
    hypothesis = re.sub(r'[^\w\s]', '', hypothesis)
    
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    distance = editdistance.eval(ref_words, hyp_words)
    
    wer = distance / len(ref_words) if len(ref_words) > 0 else float('inf')
    
    return wer


def load_data():
    data = pd.read_csv('Finetuning vs Not_Finetuning Results.csv')
    return data

def preprocess_text(text):
    # Replace hyphens with spaces
    text = text.replace('-', ' ')
    
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation from the beginning and end of the sentence only
    text = re.sub(r'[^\w\s]', '', text)  # \w matches any alphanumeric character, \s matches any whitespace character
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
            styled_text.append(f"<span style='color: #32CD32;'>{' '.join(transcription_words[j1:j2])}</span>")
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

    st.markdown("""
    <style>
    .white-text { color: white; }
    .light-green-text { color: #32CD32; }
    .red-text { color: red; }
    .yellow-text { color: yellow; }
    .bold-text { font-weight: bold; }
    .metrics-table { width: 100%; margin-top: 20px; text-align: left; }
    </style>
    """, unsafe_allow_html=True)
    
    show_blade_semantic = st.checkbox('Blade Semantic Analysis')
    show_minilm_l6 = st.checkbox('MiniLM-L6 Semantic Analysis')

    # Base columns layout
    base_columns_count = 8
    dynamic_columns = []
    if show_blade_semantic:
        dynamic_columns.extend(['Blade_Semantic_BF', 'Blade_Semantic_AF'])
    if show_minilm_l6:
        dynamic_columns.extend(['MiniLM-L6_Semantic_BF', 'MiniLM-L6_Semantic_AF'])

    total_columns = base_columns_count + len(dynamic_columns)
    cols = st.columns([1, 3, 3, 3, 1, 1, 1, 1] + [1] * len(dynamic_columns))

    headers = ["Play Audio", "Ground Truth", "Before Finetuning Transcription", "Finetuned Transcription",
               "BF_CER", "AF_CER", "BF_WER", "AF_WER"] + dynamic_columns
    
    for i, header in enumerate(headers):
        cols[i].write(f"**{header}**")
        
    total_bf_cer, total_af_cer, total_bf_wer, total_af_wer = 0, 0, 0, 0
    total_blade_bf, total_blade_af, total_minilm_bf, total_minilm_af = 0, 0, 0, 0
    count = 0

    expander = st.expander("Show Transcriptions", expanded=True)
    for index, row in data.iterrows():
        with expander:
            
            # Metric calculations
            bf_cer, af_cer = calculate_cer(row['ground_truth'], row['Before_Finetuning_transcription']), calculate_cer(row['ground_truth'], row['Finetuned_transcription'])
            bf_wer, af_wer = calculate_wer(row['ground_truth'], row['Before_Finetuning_transcription']), calculate_wer(row['ground_truth'], row['Finetuned_transcription'])
            blade_bf = float(row.get('Blade_Semantic_BF', 0))
            blade_af = float(row.get('Blade_Semantic_AF', 0))
            minilm_bf = float(row.get('MiniLM-L6_Semantic_BF', 0))
            minilm_af = float(row.get('MiniLM-L6_Semantic_AF', 0))

            total_bf_cer += bf_cer
            total_af_cer += af_cer
            total_bf_wer += bf_wer
            total_af_wer += af_wer
            total_blade_bf += blade_bf
            total_blade_af += blade_af
            total_minilm_bf += minilm_bf
            total_minilm_af += minilm_af
            count += 1
            
            
            cols = st.columns([1, 3, 3, 3, 1, 1, 1, 1] + [1] * len(dynamic_columns))
            
            audio_path = os.path.join('Data Collection', row['audio_path']) if pd.notna(row['audio_path']) else "No audio file"
            if audio_path != "No audio file":
                cols[0].audio(audio_path, format='audio/wav')
            else:
                cols[0].write("No audio file")

            cols[1].write(row['ground_truth'])
            cols[2].markdown(style_text(row['ground_truth'], row['Before_Finetuning_transcription']), unsafe_allow_html=True)
            cols[3].markdown(style_text(row['ground_truth'], row['Finetuned_transcription']), unsafe_allow_html=True)

                 

            # Determine the color class for each metric
            cer_class_bf, cer_class_af = ("yellow-text", "yellow-text") if bf_cer == af_cer else \
                (("light-green-text", "red-text") if bf_cer < af_cer else ("red-text", "light-green-text"))
            wer_class_bf, wer_class_af = ("yellow-text", "yellow-text") if bf_wer == af_wer else \
                (("light-green-text", "red-text") if bf_wer < af_wer else ("red-text", "light-green-text"))

            cols[4].markdown(f"<span class='{cer_class_bf}'>{bf_cer:.5f}</span>", unsafe_allow_html=True)
            cols[5].markdown(f"<span class='{cer_class_af}'>{af_cer:.5f}</span>", unsafe_allow_html=True)
            cols[6].markdown(f"<span class='{wer_class_bf}'>{bf_wer:.5f}</span>", unsafe_allow_html=True)
            cols[7].markdown(f"<span class='{wer_class_af}'>{af_wer:.5f}</span>", unsafe_allow_html=True)
            
            offset = 8
            if show_blade_semantic:
                blade_bf = float(row.get('Blade_Semantic_BF', 0))
                blade_af = float(row.get('Blade_Semantic_AF', 0))
                total_blade_bf += blade_bf
                total_blade_af += blade_af
                # Adjust the color logic: Green if After Finetuning is greater, Red if less, Yellow if same
                blade_color_bf = "light-green-text" if blade_bf > blade_af else ("red-text" if blade_bf < blade_af else "yellow-text")
                blade_color_af = "light-green-text" if blade_af > blade_bf else ("red-text" if blade_af < blade_bf else "yellow-text")
                cols[offset].markdown(f"<span class='bold-text {blade_color_bf}'>{blade_bf:.5f}</span>", unsafe_allow_html=True)
                cols[offset+1].markdown(f"<span class='bold-text {blade_color_af}'>{blade_af:.5f}</span>", unsafe_allow_html=True)
                offset += 2

            if show_minilm_l6:
                minilm_bf = float(row.get('MiniLM-L6_Semantic_BF', 0))
                minilm_af = float(row.get('MiniLM-L6_Semantic_AF', 0))
                total_minilm_bf += minilm_bf
                total_minilm_af += minilm_af
                # Adjust the color logic: Green if After Finetuning is greater, Red if less, Yellow if same
                minilm_color_bf = "light-green-text" if minilm_bf > minilm_af else ("red-text" if minilm_bf < minilm_af else "yellow-text")
                minilm_color_af = "light-green-text" if minilm_af > minilm_bf else ("red-text" if minilm_af < minilm_bf else "yellow-text")
                cols[offset].markdown(f"<span class='bold-text {minilm_color_bf}'>{minilm_bf:.5f}</span>", unsafe_allow_html=True)
                cols[offset+1].markdown(f"<span class='bold-text {minilm_color_af}'>{minilm_af:.5f}</span>", unsafe_allow_html=True)
            
            
    
            


     # Compute averages
    average_bf_cer = total_bf_cer / count if count > 0 else 0
    average_af_cer = total_af_cer / count if count > 0 else 0
    average_bf_wer = total_bf_wer / count if count > 0 else 0
    average_af_wer = total_af_wer / count if count > 0 else 0
    average_blade_bf = total_blade_bf / count if count > 0 else 0
    average_blade_af = total_blade_af / count if count > 0 else 0
    average_minilm_bf = total_minilm_bf / count if count > 0 else 0
    average_minilm_af = total_minilm_af / count if count > 0 else 0

    cer_color_bf = "light-green-text" if average_bf_cer < average_af_cer else ("red-text" if average_bf_cer > average_af_cer else "yellow-text")
    cer_color_af = "red-text" if average_bf_cer < average_af_cer else ("light-green-text" if average_bf_cer > average_af_cer else "yellow-text")
    wer_color_bf = "light-green-text" if average_bf_wer < average_af_wer else ("red-text" if average_bf_wer > average_af_wer else "yellow-text")
    wer_color_af = "red-text" if average_bf_wer < average_af_wer else ("light-green-text" if average_bf_wer > average_af_wer else "yellow-text")
    blade_color_bf_avg = "light-green-text" if average_blade_bf > average_blade_af else ("red-text" if average_blade_bf < average_blade_af else "yellow-text")
    blade_color_af_avg = "light-green-text" if average_blade_af > average_blade_bf else ("red-text" if average_blade_af < average_blade_bf else "yellow-text")

    # MiniLM-L6 averages comparison
    minilm_color_bf_avg = "light-green-text" if average_minilm_bf > average_minilm_af else ("red-text" if average_minilm_bf < average_minilm_af else "yellow-text")
    minilm_color_af_avg = "light-green-text" if average_minilm_af > average_minilm_bf else ("red-text" if average_minilm_af < average_minilm_bf else "yellow-text")

    # Display averages with conditional coloring in a styled table
    st.markdown(f"""
    <table class="metrics-table">
        <tr>
            <th>Metric</th>
            <th>Before Finetuning</th>
            <th>After Finetuning</th>
        </tr>
        <tr>
            <td>Average CER</td>
            <td class="{cer_color_bf}">{average_bf_cer:.5f}</td>
            <td class="{cer_color_af}">{average_af_cer:.5f}</td>
        </tr>
        <tr>
            <td>Average WER</td>
            <td class="{wer_color_bf}">{average_bf_wer:.5f}</td>
            <td class="{wer_color_af}">{average_af_wer:.5f}</td>
        </tr>
        <tr>
            <td>Average Blade Semantic BF</td>
            <td class="{blade_color_bf_avg}">{average_blade_bf:.5f}</td>
            <td class="{blade_color_af_avg}">{average_blade_af:.5f}</td>
        </tr>
        <tr>
            <td>Average MiniLM-L6 Semantic BF</td>
            <td class="{minilm_color_bf_avg}">{average_minilm_bf:.5f}</td>
            <td class="{minilm_color_af_avg}">{average_minilm_af:.5f}</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)
    
    
if __name__ == '__main__':
    main()
