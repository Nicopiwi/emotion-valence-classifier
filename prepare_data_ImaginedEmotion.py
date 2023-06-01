import mne
import pickle
import pandas as pd
import argparse


class PrepareData:
    def __init__(self, args) -> None:
        self.args = args
        self.data_path = args.data_path
        self.export_path = args.export_path
        self.emotions = {
            "anger",
            "awe",
            "compassion",
            "content",
            "disgust",
            "excite",
            "fear",
            "frustration",
            "grief",
            "happy",
            "jealousy",
            "joy",
            "love",
            "relief",
            "sad",
        }
        self.selected_annotations = {"press", "press1"}

    def _rename_press_events(self, raw: mne.io.Raw) -> None:
        # Rename annotation names
        annotations = raw.annotations

        # Modify annotation names
        idx = 0
        while idx < len(annotations):
            desc = annotations.description[idx]
            if desc in self.emotions:
                j = idx + 1
                while annotations.description[j] in self.selected_annotations:
                    annotations.description[j] = f"{annotations.description[j]}_{desc}"
                    j += 1
                idx = j
            else:
                idx += 1

        # Update the annotations
        raw.set_annotations(annotations)

    def run(self, to_df=True):
        if to_df:
            self.convert_all_files()

    def set_to_df(
        self,
        raw_path,
        filename,
        tmin=-0.5,
        tmax=2,
        baseline=(-0.5, 0),
        initial_epoch_id=0,
    ) -> int:
        """
        Convert a single .set file to pandas df and save it to a pickle format.
        Returns the ID of the last epoch created.
        """
        raw = mne.io.read_raw_eeglab(raw_path, preload=True)
        raw.resample(128)
        self._rename_press_events(raw)
        events, event_id = mne.events_from_annotations(
            raw=raw, regexp="|".join(self.selected_annotations)
        )
        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
        )
        df = epochs.to_data_frame(long_format=True)
        df["epoch"] = df["epoch"].apply(lambda x: x + initial_epoch_id)
        df["emotion"] = df["condition"].apply(
            lambda x: x.split("_")[1] if x.startswith("press") else x
        )
        df.drop(["condition"], axis=1, inplace=True)
        filename = self.export_path + filename
        df.to_pickle(filename + ".pkl")
        del raw

        return df.iloc[-1]["epoch"]

    def convert_all_files(self):
        """
        Convert all files from EEGLAB into dataframe format in order to
        facilitate further modeling tasks
        """
        dataset_directory_path = self.data_path
        initial_epoch_id = 0

        for i in range(1, self.args.subjects_numbers + 1):
            try:
                raw_path = (
                    dataset_directory_path
                    + f"sub-{i:02d}/eeg/sub-{i:02d}_task-ImaginedEmotion_eeg.set"
                )
                filename = f"processed_sub-{i:02d}"
                initial_epoch_id = self.set_to_df(
                    raw_path, filename, initial_epoch_id=initial_epoch_id
                ) + 1
            except (RuntimeError, TypeError, NameError, FileNotFoundError):
                print(f"sub-{i:02d} file has an error")
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ######## Data ########
    parser.add_argument("--dataset", type=str, default="ImaginedEmotion")
    parser.add_argument(
        "--data-path",
        type=str,
        default="./dataset/",
    )
    parser.add_argument(
        "--export-path",
        type=str,
        default="./preprocessed_data/",
        help="Export directory path",
    )
    parser.add_argument("--subjects-numbers", type=int, default=35)

    args = parser.parse_args()

    PrepareData(args).run()
