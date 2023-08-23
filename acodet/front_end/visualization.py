import streamlit as st
from pathlib import Path
import pandas as pd
from acodet.create_session_file import read_session_file
conf = read_session_file()

def output():
    if st.session_state.run_option == 1:
        if st.session_state.preset_option == 0:
            disp = ShowAnnotationPredictions()
            disp.show_annotation_predictions()
            disp.create_tabs(
                additional_headings=['Filtered annotation files',
                                     'Hourly presence plots'])
            disp.show_stats()
            disp.show_individual_files()
            disp.show_individual_files(tab_number=2, 
                                      thresh_path=f"thresh_{conf['thresh']}")
            plots = ShowPresencePlots(disp)
            
        elif st.session_state.preset_option == 1:
            disp = ShowAnnotationPredictions()
            disp.show_annotation_predictions()
            disp.create_tabs()
            disp.show_stats()
            disp.show_individual_files()

class ShowAnnotationPredictions():

    def show_annotation_predictions(self):
        self.annots_path = Path(
            conf['generated_annotations_folder']
            ).joinpath(st.session_state.time_dir)
        st.markdown(
            f"""Your annotations are saved in the folder: 
            `{self.annots_path.resolve().as_posix()}`
            """)
        
    def create_tabs(self, additional_headings=None):    
        tabs = st.tabs(['Overall statistics', 
                        'Individual annotation files',
                        *additional_headings])
        for i, tab in enumerate(tabs):
            setattr(self, f'tab{i}', tab)

    def show_stats(self):
        with self.tab0:
            df = pd.read_csv(self.annots_path.joinpath('stats.csv'))
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            st.dataframe(df, hide_index=True)

    def show_individual_files(self, tab_number=1, thresh_path='thresh_0.5'):        
        with getattr(self, f'tab{tab_number}'):
            path = self.annots_path.joinpath(thresh_path)
            annot_files = [l for l in path.rglob('*.txt')]
            display_annots = [f.relative_to(path).as_posix() 
                                for f in annot_files]
            chosen_file = st.selectbox(
                label=f"""Choose a generated annotations file from 
                `{path.resolve()}`""", 
                options=display_annots, 
                key=f'file_selec_{tab_number}')
            st.write('All of these files can be imported into Raven directly.')
            df = pd.read_csv(path.joinpath(chosen_file), sep='\t')
            st.dataframe(df, hide_index=True)
    
# a class to visualize the results of the training
class ShowPresencePlots():
    def __init__(self, disp_obj, tab_number=3) -> None:
        self.plots_paths = [p for p in 
                            disp_obj.annots_path.rglob('*analysis*')]
        with getattr(disp_obj, f'tab{tab_number}'):
            datasets = [l.parent.stem for l in self.plots_paths]
            
            chosen_dataset = st.selectbox(
                label=f"""Choose a dataset:""", 
                options=datasets, 
                key=f'dataset_selec_{tab_number}')
            self.chosen_dataset = (disp_obj.annots_path
                                   .joinpath('thresh_0.5')
                                   .joinpath(chosen_dataset)
                                   .joinpath('analysis'))
            
            plots = [l for l in self.chosen_dataset.glob('*.png')]
            
            plot = st.selectbox(
                label=f"""Choose a dataset:""", 
                options=plots, 
                key=f'file_selec_{tab_number}')
            
            from PIL import Image
            st.image(Image.open(plot))