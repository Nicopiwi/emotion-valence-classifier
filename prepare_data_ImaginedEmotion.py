import mne 
import pickle
import pandas as pd
import argparse

class PrepareData():
    def __init__(self, args) -> None:
        self.args = args
        self.data_path = args.data_path
        self.export_path = args.export_path
        self.selected_annotations = [
            'anger', 'awe', 'compassion', 'content', 'disgust', 'excite', 
            'fear', 'frustration', 'grief', 'happy', 'jealousy', 'joy', 
            'love', 'relief', 'sad']

    def run(self, to_df = True):
        if to_df:
            self.convert_all_files()


    def set_to_df(self, raw_path, filename, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0)):
        """
        Convert a single .set file to pandas df and save it to a pickle format
        """
        raw = mne.io.read_raw_eeglab(raw_path, preload=True) 
        raw.resample(128) 
        events, event_id = mne.events_from_annotations(raw = raw, 
                                                       regexp='|'.join(self.selected_annotations))
        epochs = mne.Epochs(raw, events = events, event_id = event_id, 
                            tmin=tmin, tmax=tmax, baseline=baseline)
        df = epochs.to_data_frame(long_format=True)
        filename = self.export_path + filename
        df.to_pickle(filename + '.pkl')
        del raw
    
    def convert_all_files(self):
        """
        Convert all files from EEGLAB into dataframe format in order to facilitate
        further modeling tasks
        """
        dataset_directory_path = self.data_path
        
        for i in range(1, self.args.subjects_numbers+1):
            try:
                raw_path = dataset_directory_path + f'sub-{i:02d}/eeg/sub-{i:02d}_task-ImaginedEmotion_eeg.set'
                filename = f'processed_sub-{i:02d}'
                self.set_to_df(raw_path, filename)
            except (RuntimeError, TypeError, NameError, FileNotFoundError):
                print(f'sub-{i:02d} file has an error')
                pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ######## Data ########
    parser.add_argument('--dataset', type=str, default='ImaginedEmotion')
    parser.add_argument('--data-path', type=str, default='/Users/noehsueh/UBA/Brainhack/Project/dataset/')
    parser.add_argument('--export-path', type=str, default='/Users/noehsueh/UBA/Brainhack/Project/preprocessed/', help='Export directory path')
    parser.add_argument('--subjects-numbers', type=int, default=35)

    args = parser.parse_args()
    
    PrepareData(args).run()

  

