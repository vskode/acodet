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
            plot_tabs = InitPlots(disp)
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
            plot_tabs = InitPlots(disp, tab_number=2)
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
        with self.tab0:
            try:
                df = pd.read_csv(self.annots_path.joinpath("stats.csv"))
                if "Unnamed: 0" in df.columns:
                    df = df.drop(columns=["Unnamed: 0"])
                st.dataframe(df, hide_index=True)
            except Exception as e:
                print(e)
                st.write(
                    """No stats.csv file found. Please run predefined settings 0, or 1 first
                    to view this tab."""
                )

    def show_individual_files(self, tab_number=1, thresh_path="thresh_0.5"):
        with getattr(self, f"tab{tab_number}"):
            path = self.annots_path.joinpath(thresh_path)
            annot_files = [l for l in path.rglob("*.txt")]
            display_annots = [
                f.relative_to(path).as_posix() for f in annot_files
            ]
            chosen_file = st.selectbox(
                label=f"""Choose a generated annotations file from 
                `{path.resolve()}`""",
                options=display_annots,
                key=f"file_selec_{tab_number}",
                help=help_strings.ANNOT_FILES_DROPDOWN,
            )
            st.write("All of these files can be imported into Raven directly.")
            df = pd.read_csv(path.joinpath(chosen_file), sep="\t")
            st.dataframe(df, hide_index=True)


class InitPlots:
    def __init__(self, disp_obj, tab_number=3) -> None:
        self.plots_paths = [
            p for p in disp_obj.annots_path.rglob("*analysis*")
        ]
        if not self.plots_paths:
            st.write(
                "No analysis files found for this dataset. "
                "Please run predefined settings 0, or 1 first."
            )
            st.stop()
        self.disp_obj = disp_obj
        self.tab_number = tab_number

    def create_tabs(self):
        self.tabs = {
            "binary": getattr(self.disp_obj, f"tab{self.tab_number}"),
            "presence": getattr(self.disp_obj, f"tab{self.tab_number+1}"),
        }
        for key, tab in self.tabs.items():
            self.init_tab(tab=tab, key=key)

    def init_tab(self, tab, key):
        with tab:
            datasets = [l.parent.stem for l in self.plots_paths]

            chosen_dataset = st.selectbox(
                label=f"""Choose a dataset:""",
                options=datasets,
                key=f"dataset_selec_{key}",
            )
            self.chosen_dataset = (
                self.disp_obj.annots_path.joinpath("thresh_0.5")
                .joinpath(chosen_dataset)
                .joinpath("analysis")
            )

            limit = st.radio(
                "What limit would you like to set?",
                ("Simple limit", "Sequence limit"),
                key=f"limit_selec_{key}",
                help=help_strings.LIMIT,
            )

            plot = PlotPresence(self, limit, tab, key)
            plot.plot_df()


class PlotPresence:
    def __init__(self, plot_tabs, limit, tab, key) -> None:
        self.plot_tabs = plot_tabs
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

        if limit == "Simple limit":
            self.limit = "simple_limit"
            self.thresh = "thresh"
            self.sc = False
            self.limit_max = 50
        elif limit == "Sequence limit":
            self.limit = "sequence_limit"
            self.thresh = "sequence_thresh"
            self.sc = True
            self.limit_max = 20

    def plot_df(self):
        df = pd.read_csv(
            self.plot_tabs.chosen_dataset.joinpath(
                f"{self.path_prefix}_{self.limit}.csv"
            )
        )
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

        self.create_sliders()

    def create_sliders(self):
        thresh = st.slider(
            "Threshold",
            0.35,
            0.99,
            conf[self.thresh],
            0.01,
            key=f"thresh_slider_{self.key}",
            help=help_strings.THRESHOLD,
        )

        if self.sc:
            limit = st.slider(
                "Limit",
                1,
                self.limit_max,
                conf[self.limit],
                1,
                key=f"limit_slider_{self.key}",
                help=help_strings.SC_LIMIT,
            )

        rerun = st.button("Rerun computation", key=f"update_plot_{self.key}")
        st.session_state.progbar_update = st.progress(0, text="Updating plot")
        if rerun:
            utils.write_to_session_file(self.thresh, thresh)
            if self.sc:
                utils.write_to_session_file(self.limit, limit)

            import run

            run.main(
                dont_save_plot=True,
                sc=self.sc,
                fetch_config_again=True,
                preset=3,
            )
