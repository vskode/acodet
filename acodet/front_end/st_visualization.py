import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
from acodet.front_end import utils
import plotly.express as px
from acodet.create_session_file import read_session_file
from acodet.front_end import help_strings

conf = read_session_file()


def output():
    conf = read_session_file()
    if st.session_state.run_option == 1:
        if st.session_state.preset_option == 0:
            disp = ShowAnnotationPredictions()
            disp.show_annotation_predictions()
            disp.create_tabs(
                additional_headings=[
                    "Filtered Files",
                    "Annot. Plots",
                    "Presence Plots",
                ]
            )
            disp.show_stats()
            disp.show_individual_files()
            disp.show_individual_files(
                tab_number=2, thresh_path=f"thresh_{conf['thresh']}"
            )
            plot_tabs = Results(disp)
            plot_tabs.create_tabs()

        elif st.session_state.preset_option == 1:
            disp = ShowAnnotationPredictions()
            disp.show_annotation_predictions()
            disp.create_tabs()
            disp.show_stats()
            disp.show_individual_files()

        elif st.session_state.preset_option == 2:
            conf = read_session_file()
            disp = ShowAnnotationPredictions()
            disp.show_annotation_predictions()
            disp.create_tabs()
            disp.show_stats()
            disp.show_individual_files(thresh_path=f"thresh_{conf['thresh']}")

        elif st.session_state.preset_option == 3:
            disp = ShowAnnotationPredictions()
            disp.show_annotation_predictions()
            disp.create_tabs(
                additional_headings=[
                    "Annot. Plots",
                    "Presence Plots",
                ]
            )
            disp.show_stats()
            disp.show_individual_files()
            plot_tabs = Results(disp, tab_number=2)
            plot_tabs.create_tabs()


class ShowAnnotationPredictions:
    def show_annotation_predictions(self):
        saved_annots_dir = Path(st.session_state.save_dir)
        if len(list(saved_annots_dir.parents)) > 1:
            self.annots_path = saved_annots_dir
        else:
            self.annots_path = Path(
                conf["generated_annotations_folder"]
            ).joinpath(saved_annots_dir)
        st.markdown(
            f"""Your annotations are saved in the folder: 
            `{self.annots_path.resolve().as_posix()}`
            """
        )
        utils.write_to_session_file(
            "generated_annotation_source", str(self.annots_path)
        )

    def create_tabs(self, additional_headings=[]):
        """
        Create the tabs to display the respective results.

        Parameters
        ----------
        additional_headings : list, optional
            list of additional headings, by default []
        """
        tabs = st.tabs(
            [
                "Stats",
                "Annot. Files",
                *additional_headings,
            ]
        )
        for i, tab in enumerate(tabs):
            setattr(self, f"tab{i}", tab)

    def show_stats(self):
        """
        Show stats file as pandas.DataFrame in a table.
        """
        with self.tab0:
            try:
                metadata_files = [
                    d for d in self.annots_path.glob('*.csv')
                ]
                file = st.selectbox(
                    label='Choose a metadata file',
                    options=metadata_files,
                    key='stats'
                )
                df = pd.read_csv(file)
                if "Unnamed: 0" in df.columns:
                    df = df.drop(columns=["Unnamed: 0"])
                st.dataframe(df, hide_index=True)
            except Exception as e:
                print(e)
                st.write(
                    """No stats.csv file found. Please run predefined settings 0, or 1 first
                    to view this tab."""
                )

    def show_individual_files(
        self, tab_number=1, thresh_path=conf["thresh_label"]
    ):
        with getattr(self, f"tab{tab_number}"):
            path = self.annots_path.joinpath(thresh_path)
            annot_files = [l for l in path.rglob("*.txt")]
            display_annots = [
                f.relative_to(path).as_posix() for f in annot_files
            ]
            if not display_annots:
                st.text(
                    'No annotations left after filtering with '
                    f'the threshold={thresh_path}.'
                    )
                return
            if st.session_state.multi_datasets_annot:
                datasets = [d.stem for d in path.iterdir() 
                           if not 'analysis' in d.stem]
                dataset = st.selectbox(
                    label="Choose a dataset",
                    options=datasets,
                    key=f'dataset_{tab_number}'
                )
                
            if st.session_state.multiclass:
                if st.session_state.multi_datasets_annot:
                    p = path / dataset
                else:
                    p = path
                labels = [d.stem for d in p.iterdir() 
                           if not 'Combined' in d.stem
                           or 'multiclass' in d.stem]
                lbl = st.selectbox(
                    label="Choose a class",
                    options=labels,
                    key=f'lbl_{tab_number}'
                )
                display_annots = [d for d in display_annots if lbl in d]
                
            chosen_file = st.selectbox(
                label=f"""Choose a generated annotations file from 
                `{path.resolve()}`""",
                index=0,
                options=display_annots,
                key=f"file_selec_{tab_number}",
                help=help_strings.ANNOT_FILES_DROPDOWN,
            )
            st.write("All of these files can be imported into Raven directly.")
            try:
                df = pd.read_csv(path.joinpath(chosen_file), sep="\t")
                st.dataframe(df, hide_index=True)
            except Exception as e:
                print('File couldnt be loaded due to: ', e)
                


class Results(utils.Limits):
    def __init__(self, disp_obj, tab_number=3) -> None:
        """
        Results class containing all of the data processings to prepare
        data for plots and tables.

        Parameters
        ----------
        disp_obj : object
            ShowAnnotationPredictions object to link processing to
            the respective streamlit widget
        tab_number : int, optional
            number of tab to show results in, by default 3
        """
        self.plots_paths = [
            [d for d in p.iterdir() if d.is_dir()]
            for p in disp_obj.annots_path.rglob("*analysis*")
        ][0]
        if not self.plots_paths:
            st.write(
                "No analysis files found for this dataset. "
                "Please run predefined settings 0, or 1 first."
            )
            st.stop()
        self.disp_obj = disp_obj
        self.tab_number = tab_number

    def create_tabs(self):
        """
        Create tabs for plots.
        """
        self.tabs = {
            "binary": getattr(self.disp_obj, f"tab{self.tab_number}"),
            "presence": getattr(self.disp_obj, f"tab{self.tab_number+1}"),
        }
        for key, tab in self.tabs.items():
            self.init_tab(tab=tab, key=key)

    def init_tab(self, tab, key):
        with tab:
            if st.session_state.multi_datasets_annot:
                top_dir = self.plots_paths[0].parent.parent
                datasets = [d.stem for d in top_dir.iterdir() 
                           if not 'analysis' in d.stem]
                dataset = st.selectbox(
                    label="Choose a dataset",
                    index=0,
                    options=datasets,
                    key=f'dataset_{key}'
                )
                classes = [d.stem for d in top_dir.joinpath(dataset).iterdir()
                          if not 'Combined' in d.stem or not 'multiclass' in d.stem]
            else:
                classes = [l.stem for l in self.plots_paths if not 'Combined' in l.stem]
                
            chosen_dataset = st.selectbox(
                label=f"""Choose a dataset:""",
                options=classes,
                key=f"dataset_selec_{key}",
            )
            self.chosen_dataset = (
                self.disp_obj.annots_path.joinpath(conf["thresh_label"])
                .joinpath("analysis")
                .joinpath(chosen_dataset)
            )

            limit = st.radio(
                "What limit would you like to set?",
                ("Simple limit", "Sequence limit"),
                key=f"limit_selec_{key}",
                help=help_strings.LIMIT,
            )

            super(Results, self).__init__(limit, key)

            results = PlotDisplay(self.chosen_dataset, tab, key)
            results.plot_df(self.limit_label)

            self.create_limit_sliders()
            self.rerun_computation_btn()

            self.save_selection_tables_with_limit_settings()

    def rerun_computation_btn(self):
        """
        Show rerun computation button after limits have been set and
        execute run.main.
        """
        rerun = st.button("Rerun computation", key=f"update_plot_{self.key}")
        st.session_state.progbar_update = st.progress(0, text="Updating plot")
        if rerun:
            utils.write_to_session_file(self.thresh_label, self.thresh)
            if self.sc:
                utils.write_to_session_file(self.limit_label, self.limit)

            import run

            run.main(
                dont_save_plot=True,
                sc=self.sc,
                fetch_config_again=True,
                preset=3,
                update_plot=True,
                chosen_dataset_stem=self.chosen_dataset.stem
            )


class PlotDisplay:
    def __init__(self, chosen_dataset, tab, key) -> None:
        self.chosen_dataset = chosen_dataset
        self.tab = tab
        self.key = key

        if key == "binary":
            self.path_prefix = "hourly_annotation"
            self.cbar_label = "Number of annotations"
            self.c_range = [0, conf["max_annots_per_hour"]]
        elif key == "presence":
            self.path_prefix = "hourly_presence"
            self.cbar_label = "Presence"
            self.c_range = [0, 1]

    def plot_df(self, limit_label):
        """

        Plot dataframe showing either hourly presence or annotation count
        in an interactive plotly visualization.

        TODO onclick display of scrollable spectrogram would be really sick.

        Parameters
        ----------
        limit_label : string
            key of config dict to acces simple or sequence limit
        """
        df = pd.read_csv(
            self.chosen_dataset.joinpath(
                f"{self.path_prefix}_{limit_label}.csv"
            )
        )
        if len(df) == 0:
            st.text(
                'There were too few annotations of this species to pass '
                f"the threshold of {conf['thresh']}. "
                'For this reasons no data ist displayed.'
                )
            st.stop()
        df.index = pd.DatetimeIndex(df.Date)
        df = df.reindex(
            pd.date_range(df.index[0], df.index[-1]), fill_value=np.nan
        )

        h_of_day_str = ["%.2i:00" % i for i in range(24)]
        h_pres = df.loc[:, h_of_day_str]

        fig = px.imshow(
            h_pres.T,
            labels=dict(x="Date", y="Time of Day", color=self.cbar_label),
            x=df.index,
            y=h_of_day_str,
            range_color=self.c_range,
            color_continuous_scale="blugrn",
        )

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        fig.update_layout(hovermode="x")

        st.plotly_chart(fig)
