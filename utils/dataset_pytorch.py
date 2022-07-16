import json
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


def get_frame_number_from_id(video_id, frame_id, img_ext='.png'):
    n = frame_id.replace(f'{video_id}/', '').replace('img-', '').replace(img_ext, '')

    try:
        return int(n)
    except ValueError:
        raise ValueError(f'Invalid frame id: {frame_id}')


def get_box_from_keypoints(keypoints, box_border=100):
    box_x1, box_y1 = keypoints[:, :2].min(axis=0)
    box_x2, box_y2 = keypoints[:, :2].max(axis=0)
    box_x1 -= box_border
    box_y1 -= box_border
    box_x2 += box_border
    box_y2 += box_border
    w = box_x2 - box_x1
    h = box_y2 - box_y1
    assert w > 0 and h > 0, f'Invalid box: {box_x1}, {box_x2}, {box_y1}, {box_y2}'
    box = (box_x1, box_y1, w, h, 1)  # confidence score 1
    return box


def normalise_keypoints(box, keypoints):
    x, y, w, h, _ = box
    xk = (keypoints[:, 0] - x) / w
    yk = (keypoints[:, 1] - y) / h
    nk = np.stack((xk, yk), axis=1)  # no need to stack the scores
    return nk


class BraceDataset(Dataset):
    def __init__(self, sequences_path, df, sample_length=900, max_length=None):
        assert (sample_length is None) != (max_length is None), 'Choose sample length or max length but not both'
        assert len(df) > 0
        self.df = df
        self.sequences = []
        self.clips = []
        self.clip_labels = []
        self.max_length = max_length
        self.clip_label_map = dict(toprock=0, footwork=1, powermove=2)

        clip_paths_by_video = {}
        pose_jsons = list(Path(sequences_path).rglob('**/*.json'))

        for video_id in self.df.video_id.unique():
            video_paths = [p for p in pose_jsons if p.stem.startswith(video_id)]
            clip_paths_by_video[video_id] = {}

            for vp in video_paths:
                splits = vp.stem.replace(f'{video_id}_', '').split('_')
                clip_start, clip_end = (int(x) for x in splits[0].split('-'))
                clip_paths_by_video[video_id][(clip_start, clip_end)] = vp

        for seq_t in tqdm(self.df.itertuples(), total=len(self.df), desc='Loading BRACE'):
            seq_clips = self.get_seq_clips(clip_paths_by_video, seq_t)
            assert len(seq_clips) > 0, f'Did not find any segments for sequence {seq_t}'
            clips = []

            for p in seq_clips:
                clip, _, _ = BraceDataset.load_clip(p)
                clips.append(clip)
                clip_label = None

                for cat in ('toprock', 'footwork', 'powermove'):
                    if cat in p.name:
                        assert clip_label is None, f'Trying to override clip labels for clip {p}'
                        clip_label = self.clip_label_map[cat]
                        break

                assert clip_label is not None
                self.clip_labels.append(clip_label)

            self.clips.extend(clips)
            seq = np.concatenate(clips, axis=0)
            self.sequences.append(seq)

        assert len(self.df) == len(self.sequences)
        self.clips = np.array(self.clips, dtype=object)
        self.clip_labels = np.array(self.clip_labels)

        if max_length is not None:
            max_seq_length = np.max([x.shape[0] for x in self.sequences])
            assert max_seq_length <= max_length, f'Found a sequence whose length {max_seq_length} is > than the max ' \
                                                 f'sequence allowed {max_length}. Adjust accordingly'
        else:
            avg_seq_length = np.mean([x.shape[0] for x in self.sequences])
            print(f'Going to sample {sample_length} frames from each sequence. '
                  f'Average sequence length is {avg_seq_length}')

        self.n_dancers = len(self.df.dancer_id.unique())
        self.sample_length = sample_length

    @staticmethod
    def get_seq_clips(clip_paths_by_video, seq_t):
        return [p for (start, end), p in sorted(clip_paths_by_video[seq_t.video_id].items(),
                                                key=lambda kv: kv[0][0])
                if seq_t.start_frame <= start <= seq_t.end_frame and seq_t.start_frame <= end <= seq_t.end_frame]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq_row = self.df.iloc[index]
        seq = np.array(self.sequences[index])
        metadata = seq_row.to_dict()

        if self.sample_length is None:
            missing = self.max_length - seq.shape[0]

            if missing > 0:
                zeros = np.zeros((missing, *seq[0].shape))
                seq = np.concatenate((seq, zeros), axis=0)
        else:
            idx = np.linspace(0, seq.shape[0] - 1, self.sample_length, dtype=int)
            seq = seq[idx, ...]

        return seq, metadata

    @staticmethod
    def load_clip(pose_path, img_ext='.png', broken_policy='skip_frame', policy_warning=False):
        with open(pose_path) as f:
            d = json.load(f)

        frame_ids = sorted(d.keys())
        video_id = frame_ids[0].split('/')[0]
        frame_numbers = [get_frame_number_from_id(video_id, f_id, img_ext=img_ext) for f_id in frame_ids]
        clip = []

        for i, (frame_id, frame_number) in enumerate(zip(frame_ids, frame_numbers)):
            keypoints = np.array(d[frame_id]['keypoints'])

            try:
                box = get_box_from_keypoints(keypoints, box_border=0)
                norm_kpt = normalise_keypoints(box, keypoints)
            except AssertionError as e:
                if broken_policy == 'skip_frame':

                    if policy_warning:
                        print(f'Got broken keypoints at frame {frame_id}. Skipping as per broken policy')
                    continue
                else:
                    raise e

            clip.append(norm_kpt)

        clip = np.stack(clip, axis=0)
        clip_id = pose_path.stem

        return clip, clip_id, video_id


def filter_set(sequences_df, set_df):

    pass


if __name__ == '__main__':
    # adjust csv paths if you don't run this script from its folder

    sequences_path_ = Path('../dataset')  # path where you download and unzipped the keypoints
    df_ = pd.read_csv(Path('../annotations/sequences.csv'))

    train_df = pd.read_csv('../annotations/sequences_train.csv')
    train_df = df_[df_.uid.isin(train_df.uid)]

    brace_train = BraceDataset(sequences_path_, train_df)
    print(f'Loaded BRACE training set! We got {len(brace_train)} training sequences')
    skeletons_train, metadata_train = brace_train.__getitem__(0)
    print(metadata_train)

    test_df = pd.read_csv('../annotations/sequences_test.csv')
    test_df = df_[df_.uid.isin(test_df.uid)]

    brace_test = BraceDataset(sequences_path_, test_df)
    print(f'Loaded BRACE test set! We got {len(brace_train)} testing sequences')
    skeletons_test, metadata_test = brace_test.__getitem__(0)
    print(metadata_test)
