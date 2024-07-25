import streamlit as st
import pandas as pd
import os
import difflib
import re
import editdistance
import streamlit.components.v1 as components

def load_data():
    data = pd.read_csv('Finetuning vs Not_Finetuning Results.csv')
    # Pre-calculate the metrics for baseline 'Before Finetuning'
    data['BF_CER'] = data.apply(lambda row: calculate_cer(row['Ground Truth'], row['Before Finetuning Transcription']), axis=1)
    return data

def preprocess_text(text):
    text = text.replace('-', ' ').lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def calculate_cer(ground_truth, transcription):
    ground_truth = re.sub(r'[^\w\s]', '', ground_truth.lower().strip())
    transcription = re.sub(r'[^\w\s]', '', transcription.lower().strip())
    distance = editdistance.eval(ground_truth, transcription)
    return distance / len(ground_truth) if len(ground_truth) > 0 else float('inf')

def calculate_wer(reference, hypothesis):
    reference = re.sub(r'[^\w\s]', '', reference.lower().strip())
    hypothesis = re.sub(r'[^\w\s]', '', hypothesis.lower().strip())
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    distance = editdistance.eval(ref_words, hyp_words)
    return distance / len(ref_words) if len(ref_words) > 0 else float('inf')

def style_text(original, transcription):
    original_words = preprocess_text(original)
    transcription_words = preprocess_text(transcription)
    styled_text = []
    s = difflib.SequenceMatcher(None, original_words, transcription_words)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'equal':
            styled_text.append(f"<span style='color: #32CD32;'>{' '.join(transcription_words[j1:j2])}</span>")
        else:
            styled_text.append(f"<span style='color: red;'>{' '.join(transcription_words[j1:j2])}</span>")
    return ' '.join(styled_text)

def compare_metric(base, compare):
    if base < compare:
        return "light-green-text"
    elif base > compare:
        return "red-text"
    else:
        return "yellow-text"

def calculate_averages(data):
    averages = {
        'Before Finetuning': {'CER': data['BF_CER'].mean(), 'Blade_Semantic': data['Blade_Semantic_BF'].mean()}
        
    }
    
    for transcription in ['Finetuned', 'XTTS_FineTuned', 'VITS_FineTuned', 'Piper_FineTuned']:
        cer_mean = data.apply(lambda row: calculate_cer(row['Ground Truth'], row[f'{transcription}_Transcription']), axis=1).mean()
        blade_semantic_key = f'Blade_Semantic_{transcription}'
        blade_semantic_mean = data[blade_semantic_key].mean() if blade_semantic_key in data else None
        averages[transcription] = {'CER': cer_mean, 'Blade_Semantic': blade_semantic_mean}
    
    return averages

def get_color_class_cer(value, min_value, max_value):
    if value == min_value:
        return 'light-green-text'
    else:
        return 'red-text'

def get_color_class_blade(value, min_value, max_value):
    if value == max_value:
        return 'light-green-text'
    else:
        return 'red-text'

def compare_blade_semantic(base, compare):
    if compare > base:
        return "light-green-text"
    elif compare < base:
        return "red-text"
    else:
        return "yellow-text"

def main():
    st.set_page_config(layout="wide")
    st.title("Audio Transcription Analysis")
    data = load_data()

    st.header('Transcription Comparisons')
    st.markdown("""
        <style>
            /* Targets the main container of Streamlit's layout */
            .reportview-container .main .block-container {
                max-width: none; /* Removes limitation on max width */
                width: 100%;
            }
            /* Targets the specific container that might hold your table, ensuring it can scroll horizontally */
            .reportview-container .main {
                overflow-x: auto; /* Enables horizontal scrolling */
            }
            .white-text { color: white; }
            .light-green-text { color: #32CD32; }
            .red-text { color: red; }
            .yellow-text { color: yellow; }
            .blue-text { color: blue; }
            .bold-text { font-weight: bold; }
            /* Additional styles for table headers and cells */
            th, td {
                padding: 8px; /* Padding for table cells */
                text-align: left; /* Align text to the left in cells */
                border-bottom: 1px solid #ddd; /* Adds a bottom border to table cells */
            }
        </style>
    """, unsafe_allow_html=True)
      
    st.subheader("Show Comparison with")
    show_finetuned = st.checkbox('Finetuned Transcription', value=True)
    show_xtts_finetuned = st.checkbox('XTTS FineTuned Transcription')
    show_vits_finetuned = st.checkbox('VITS FineTuned Transcription')
    show_piper_finetuned = st.checkbox('Piper FineTuned Transcription')
    
    st.subheader("Show Analysis of")
    show_blade_semantic = st.checkbox('Blade Semantic Analysis', value=True)


    headers = ["Audio", "Ground Truth", "Before Finetuning Transcription", "BF_CER","Blade_Semantic_BF"]
    transcriptions = {
        'Finetuned': show_finetuned,
        'XTTS_FineTuned': show_xtts_finetuned,
        'VITS_FineTuned': show_vits_finetuned,
        "Piper_FineTuned": show_piper_finetuned
    }

    for key in transcriptions:
        if transcriptions[key]:
            headers.extend([f"{key} Transcription", f"{key} CER"])
            if show_blade_semantic:
                headers.append(f"Blade_Semantic_{key}")
            

    table_html = "<div class='scrollable-container'><table class='scrollable-table'><thead><tr>"
    for header in headers:
        table_html += f"<th>{header}</th>"
    table_html += "</tr></thead><tbody>"

    for index, row in data.iterrows():
        table_html += "<tr>"
        
        audio_path = os.path.join('Data Collection', row['audio_path']) if pd.notna(row['audio_path']) else None
        if audio_path and os.path.exists(audio_path):

            table_html += f"<td>{row['audio_path']}</td>"
        else:
            table_html += "<td>No audio file</td>"
            
            
        table_html += f"<td>{row['Ground Truth']}</td>"
        table_html += f"<td>{style_text(row['Ground Truth'], row['Before Finetuning Transcription'])}</td>"

        bf_cer = row['BF_CER']
        blade_semantic_bf = row['Blade_Semantic_BF']
        bf_cer_color = 'white'
        blade_semantic_bf_color = 'white'

        for key in transcriptions:
            if transcriptions[key]:
                cer = calculate_cer(row['Ground Truth'], row[f'{key}_Transcription'])
                bf_cer_color = compare_metric(bf_cer, cer)
                
                blade_semantic_bf_color = compare_blade_semantic(row[f"Blade_Semantic_{key}"],blade_semantic_bf )
                

        table_html += f"<td class='{bf_cer_color}'>{bf_cer:.5f}</td>"
        table_html += f"<td class='{blade_semantic_bf_color}'>{blade_semantic_bf:.5f}</td>"
        
    

        for key in transcriptions:
            if transcriptions[key]:
                cer = calculate_cer(row['Ground Truth'], row[f'{key}_Transcription'])
                cer_color = compare_metric(cer, bf_cer)
                table_html += f"<td>{style_text(row['Ground Truth'], row[f'{key}_Transcription'])}</td>"
                table_html += f"<td class='{cer_color}'>{cer:.5f}</td>"
                
                if show_blade_semantic:
                    semantic_value = row[f"Blade_Semantic_{key}"]
                    semantic_color = compare_blade_semantic(blade_semantic_bf, semantic_value)
                    table_html += f"<td class='{semantic_color}'>{semantic_value:.5f}</td>"

        table_html += "</tr>"
    table_html += "</tbody></table></div>"

    st.markdown(table_html, unsafe_allow_html=True)
    
    averages = calculate_averages(data)
    # Ensure Blade_Semantic is in the averages dictionary
    blade_semantic_keys = [key for key in averages.keys() if 'Blade_Semantic' in averages[key]]

    
    # Find the minimum and maximum CER for conditional coloring
    min_cer = min(averages[transcription]['CER'] for transcription in averages.keys())
    max_cer = max(averages[transcription]['CER'] for transcription in averages.keys())
    
    # Find the minimum and maximum Blade Semantic for conditional coloring
    min_blade_semantic = min(averages[transcription]['Blade_Semantic'] for transcription in blade_semantic_keys)
    max_blade_semantic = max(averages[transcription]['Blade_Semantic'] for transcription in blade_semantic_keys)
    
    st.header('Average CER and Blade Semantic')
    avg_html = "<table class='scrollable-table'><thead><tr><th>Metric</th>"
    for transcription in averages.keys():
        avg_html += f"<th>{transcription}</th>"
    avg_html += "</tr></thead><tbody>"
    for metric in ['CER', 'Blade_Semantic']:
        avg_html += f"<tr><td>{metric}</td>"
        for transcription in averages.keys():
            if metric in averages[transcription] and averages[transcription][metric] is not None:
                if metric == 'CER':
                    color_class = get_color_class_cer(averages[transcription][metric], min_cer, max_cer)
                elif metric == 'Blade_Semantic':
                    color_class = get_color_class_blade(averages[transcription][metric], min_blade_semantic, max_blade_semantic)
                avg_html += f"<td class='{color_class}'>{averages[transcription][metric]:.5f}</td>"
            else:
                avg_html += "<td>N/A</td>"
        avg_html += "</tr>"
    avg_html += "</tbody></table>"

    st.markdown(avg_html, unsafe_allow_html=True)   


if __name__ == '__main__':
    main()


# THIS IS THE