import torch as th
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import random
import ffmpeg
import time
import re
import copy
import json
class HT100M_DataLoader(Dataset):
    """HowTo100M Video-Text loader."""

    def __init__(
            self,
            csv,
            video_root='',
            caption_root='',
            min_time=4.0,
            fps=10,
            num_frames=16,
            size=224,
            crop_only=False,
            center_crop=True,
            benchmark=False,
            token_to_word_path='../HowTo100M/data/dict.npy',
            max_words=20,
            num_candidates=1,
            random_left_right_flip=False,
    ):
        """
        Args:
        """
        assert isinstance(size, int)
        self.csv = pd.read_csv(os.path.join(os.path.dirname(__file__), csv))
        self.video_root = video_root
        self.caption_root = caption_root
        self.min_time = min_time
        self.size = size
        self.num_frames = num_frames
        self.fps = fps
        self.num_sec = self.num_frames / float(self.fps)
        self.crop_only = crop_only
        self.center_crop = center_crop
        self.benchmark = benchmark
        self.max_words = max_words
        token_to_word = np.load(os.path.join(os.path.dirname(__file__), token_to_word_path))
        self.word_to_token = {}
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1
        self.num_candidates = num_candidates
        self.random_flip = random_left_right_flip

    def __len__(self):
        return len(self.csv)

    def _sample_indices(self, record):
        """
        For each segment, chooses an index from where frames
        are to be loaded from.
        Args:
            record: VideoRecord denoting a video sample.
        Returns:
            List of indices of where the frames of each
            segment are to be loaded from.
        """
        
        segment_duration = (record.shape[1]) / self.num_frames
        # print(record.shape[1])
        # print(segment_duration)
        if int(segment_duration) > 0:
            offsets = np.array(list(map(int,np.multiply(list(range(self.num_frames)), segment_duration) ))) + np.random.randint(segment_duration, size=self.num_frames)

        # edge cases for when a video has approximately less than (num_frames*frames_per_segment) frames.
        # random sampling in that case, which will lead to repeated frames.
        else:
            offsets = np.sort(np.random.randint(record.shape[1], size=self.num_frames))

        return offsets

    def _get_video(self, video_path, start, end):
        #start_seek = random.randint(start, int(max(start, end - self.num_sec)))
        cmd = (
            ffmpeg
            .input(video_path, ss=start, t=end-start)
            .filter('fps', fps=self.fps)
        )
        if self.center_crop:
            aw, ah = 0.5, 0.5
        else:
            aw, ah = random.uniform(0, 1), random.uniform(0, 1)
        if self.crop_only:
            cmd = (
                cmd.crop('(iw - {})*{}'.format(self.size, aw),
                         '(ih - {})*{}'.format(self.size, ah),
                         str(self.size), str(self.size))
            )
        else:
            cmd = (
                cmd.crop('(iw - min(iw,ih))*{}'.format(aw),
                         '(ih - min(iw,ih))*{}'.format(ah),
                         'min(iw,ih)',
                         'min(iw,ih)')
                .filter('scale', self.size, self.size)
            )
        if self.random_flip and random.uniform(0, 1) > 0.5:
            cmd = cmd.hflip()
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, self.size, self.size, 3])
        video = th.from_numpy(video.copy())
        video = video.permute(3, 0, 1, 2)
        # print('start={},end={}'.format(start,end))
        return video[:,self._sample_indices(video)]


    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):
        words = [self.word_to_token[word] for word in words if word in self.word_to_token]
        if words:
            we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
            return we
        else:
            return th.zeros(self.max_words, dtype=th.long)

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = th.zeros(size - len(tensor)).long()
            return th.cat((tensor, zero), dim=0)

    def words_to_ids(self, x):
        return self._words_to_token(self._split_text(x))

    def _find_nearest_candidates(self, caption, ind):
        start, end = ind, ind
        diff = caption['end'][end] - caption['start'][start]
        n_candidate = 1
        while n_candidate < self.num_candidates:
           if start == 0:
               return 0
           elif end == len(caption) - 1:
               return start - (self.num_candidates - n_candidate)
           elif caption['end'][end] - caption['start'][start - 1] < caption['end'][end + 1] - caption['start'][start]:
               start -= 1
           else:
               end += 1
           n_candidate += 1
        return start
    def min_duration(self,start,end):
        if end - start < self.min_time:
            diff = self.min_time - end + start
            start = max(0, start - diff / 2)
            end = start + self.min_time
        return start ,end
    def _get_text(self, caption):
        caption_json = open(caption, 'r').read()
        cap = pd.DataFrame(json.loads(caption_json))
        ind = random.randint(0, len(cap) - 1)
        #if self.num_candidates == 1:
        words = self.words_to_ids(cap['text'].values[ind])
        '''
        else:
            words = th.zeros(self.num_candidates, self.max_words, dtype=th.long)
            cap_start = self._find_nearest_candidates(cap, ind)
            for i in range(self.num_candidates):
                words[i] = self.words_to_ids(cap['text'].values[max(0, min(len(cap['text']) - 1, cap_start + i))])
        '''
        start, end = cap['start'].values[ind], cap['end'].values[ind]
        #TODO: May need to be improved for edge cases. 
        start,end=self.min_duration(start,end)
        return words, int(start), int(end) ,ind,cap
    def video_candidates (self,start,end,cap,ind):
        duration=end-start
        return {'s':[cap['start'].values[max(0,min(len(cap['text'])-1,ind-self.num_candidates//2+i))] for i in range(self.num_candidates)],
                'e':[cap['end'].values[max(0,min(len(cap['text'])-1,ind-self.num_candidates//2+i))]   for i in range(self.num_candidates)]}
    def __getitem__(self, idx):
        video_file = self.csv['video_path'][idx]
        video_id = video_file.split('.')[0]
        video_path = os.path.join(self.video_root, video_file)
        text, start, end ,ind,cap= self._get_text(os.path.join(self.caption_root, video_id + '.json'))
        
        video_time=self.video_candidates(start,end,cap,ind)
        mk_candidates=0
        for i in range(self.num_candidates):
            start=video_time['s'][i]
            end=video_time['e'][i]
            start,end=self.min_duration(start,end)
            video=self._get_video(video_path, int(start), int(end))
            if mk_candidates==0:
                video_cand=th.zeros((self.num_candidates, *video.shape) ,dtype=th.uint8)
                mk_candidates+=1
            video_cand[i]=video
        return {'video': video_cand, 'text': text}
