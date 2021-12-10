import sys, pdb, random
from tqdm import tqdm
sys.path.append('../')
sys.path.append('../../')
from active_divergence.data.audio.transforms import AudioTransform, NotInvertibleError
from active_divergence.data.audio.metadata import *
from active_divergence.utils import checklist, checktensor, checknumpy, ContinuousList
import torch, torchaudio, os, dill, re, numpy as np, matplotlib.pyplot as plt, random, copy, lardon, numbers, math
from tqdm import tqdm
from torch.utils.data import Dataset, BatchSampler
from torch.nn.utils import rnn

def parse_audio_file(f, sr=None, len=None, mix=True):
    x, f_sr = torchaudio.load(f)
    if len is not None:
        if isinstance(len, float):
            target_len = int(len * f_sr)
            x = x[..., :target_len]
        elif isinstance(len, int):
            x = x[..., :len]
    if f_sr != sr:
        x = torchaudio.transforms.Resample(f_sr, sr)(x)
    return x, {'time':torch.Tensor([0.0]), 'sr':sr or f_sr}

def parse_folder(d, valid_exts):
    files = []
    for root, directory, files in os.walk(d):
        valid_files = filter(lambda x: os.path.splitext(x)[1] in valid_exts, files)
        valid_files = [root+'/'+f for f in valid_files]
        files.extend(valid_files)
    return files

def import_classes(f):
    classes = {}
    with open(f, 'r') as f:
        for line in f.readlines():
            idx, cl = line.split('\t')
            classes[cl[:-1]] = int(idx)
    classes['_length'] = len(classes.keys())
    return classes

def format_array(arr, sequence_idx):
    if isinstance(arr, list):
        return [format_array(a, sequence_idx) for a in arr]
    elif isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)
    elif torch.is_tensor(arr):
        pass
    else:
        arr = torch.Tensor(arr)
    if sequence_idx != 1:
        permute_idx = list(range(len(arr.shape)))
        del permute_idx[sequence_idx]
        arr = arr.permute(sequence_idx, *permute_idx).contiguous()
    return arr

def get_seq_range(datas, metas, seq_index=0):
    if isinstance(datas, list):
        return [get_seq_range(datas[i], metas[i], seq_index=seq_index) for i in range(len(datas))]
    current_idx = metas['idx'][seq_index]
    if isinstance(current_idx, slice):
        seq_range = torch.arange(current_idx.start or 0,
                                 current_idx.stop or datas.shape[seq_index])
    else:
        seq_range = torch.LongTensor([current_idx])
    return seq_range

def collate_out(data, pack=False):
    if isinstance(data[0], (list, tuple)):
        return [collate_out(list(d)) for d in zip(*data)]
    lengths = [0 if len(d.shape) == 0 else d.shape[0] for d in data]

    if isinstance(data[0], np.ndarray):
        data = [torch.from_numpy(d) for d in data]
    elif not hasattr(data[0], "__iter__"):
        data = [torch.Tensor([d]) for d in data]
    if len(set(lengths)) == 1:
        data = torch.stack(data)
    else:
        if pack:
            data = rnn.pack_padded_sequence(rnn.pad_sequence(data, batch_first=True), lengths,
                                            batch_first=True, enforce_sorted=False)
        else:
            data = rnn.pad_sequence(data, batch_first=True)
    return data


class AudioDataset(Dataset):
    """Returns a dataset suitable for machine learning processing in PyTorch.
    It allows to run the main quality-check routines.
    """
    types = ['.aif', '.wav', '.aiff', '.mp3']
    scale_amount = 10

    def _settransform(self, transforms):
        assert isinstance(transforms, AudioTransform)
        self._transforms = transforms
        if self._transforms.needs_scaling and len(self.data) > 0:
            self.scale_transform(self.scale_amount)
    def _gettransform(self):
        return self._transforms
    def _deltransform(self):
        self._transforms = AudioTransform()
    transforms = property(_gettransform, _settransform, _deltransform)

    def _setactivetasks(self, active_tasks):
        active_tasks = checklist(active_tasks)
        for a in active_tasks:
            assert a in self.metadata.keys()
        self._active_tasks = active_tasks
    def _getactivetasks(self):
        return self._active_tasks
    def _delactivetasks(self):
        self._active_tasks = []
    active_tasks = property(_getactivetasks, _setactivetasks, _delactivetasks)

    def __init__(self, root, sr=None, transforms=AudioTransform(), target_length=None, target_transforms=None, drop_time=False, active_tasks = [], **kwargs):
        """
        Args:
            root (string): root folder
            transforms (MidiTransform, optional): Midi transformation to be applied. Defaults to MidiTransform().
            chord_transforms (TargetTransform, optional): Target transformation. Defaults to None.
            window_length (int, optional): Defaults to 2.
            hop_length (int, optional): Defaults to 2.
            quantization (list, optional): Defaults to [4,3].
            parser (func): parser function (see importation.midi_import)
        """
        if not os.path.isdir(root):
            raise FileNotFoundError(root)
        self.root_directory = root
        self.target_length = target_length
        self.data = []
        self.metadata = {}
        self.classes = {}
        self.sr = sr or 44100
        self.partitions = {}
        self.partition_files = None
        self.parse_files()

        # data attributes
        self._pre_transform = AudioTransform()
        self.transforms = transforms
        self._drop_time = False
        self.scale_amount = kwargs.get('scale_amount', self.scale_amount)
        # metadata attributes
        self._active_tasks = active_tasks
        self.target_transforms = target_transforms
        # sequence attributes
        self._sequence_mode = None
        self._sequence_length = None
        self._sequence_idx = -2
        self._dtype = torch.get_default_dtype()

    @property
    def available_transforms(self):
        transform_dir = f"{self.root_directory}/transforms"
        if not os.path.isdir(transform_dir):
            return []
        transforms = os.listdir(transform_dir)
        available_transforms = []
        for d in transforms:
            current_dir = f"{transform_dir}/{d}"
            if d != "raw":
                if not os.path.isdir(current_dir):
                    continue
                if not os.path.isfile(current_dir+"/parsing.ldn"):
                    continue
            available_transforms.append(d)
        return available_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item, **kwargs):
        # retrieve data
        if isinstance(item, slice):
            item = list(range(len(self)))[item]

        if hasattr(item, "__iter__"):
            data = [self._get_item(i, **kwargs) for i in item]
            data, seq_idx = [d[0] for d in data], [d[1] for d in data]
        else:
            data, seq_idx = self._get_item(item, **kwargs)

        metadata = self._get_item_metadata(item, seq=seq_idx)

        if self._transforms is not None:
            data = self._transforms(data, time=metadata.get('time'))
            if metadata.get('time') is not None:
                data, time = data

        if isinstance(data, list):
            try:
                data = collate_out(data)
            except RuntimeError:
                print('[Warning] Data could not be stacked with indexes %s'%item)

        return data, metadata


    def _get_item(self, item: int, **kwargs):
        sequence_mode = kwargs.get('sequence_mode', self._sequence_mode)
        sequence_length = kwargs.get('sequence_length', self._sequence_length)
        sequence_idx = kwargs.get('sequence_idx', self._sequence_idx or 0)
        # importing data
        seq_idx = torch.tensor(0, dtype=torch.long)
        if isinstance(self.data, list):
            idx = [slice(None)] * len(self.data[item].shape)
            is_asynchronous = isinstance(self.data[item], lardon.OfflineDataList)
            if sequence_length is not None:
                if sequence_mode == "random":
                    if is_asynchronous:
                        idx[sequence_idx] = lardon.randslice(sequence_length)
                    else:
                        seq_idx = random.randrange(0, self.data[item].shape[sequence_idx] - sequence_length)
                        idx[sequence_idx] = slice(seq_idx, seq_idx + sequence_length)
                else:
                    idx[sequence_idx] = slice(0, sequence_length)
            if is_asynchronous:
                data, meta = self.data[item].__getitem__(tuple(idx), return_metadata=True, return_indices=True)
                seq_idx = meta['idx']
            else:
                data = self.data[item].__getitem__(tuple(idx))
        else:
            ndim = None
            if self.data.ndim is None:
                ndim = len(self.data.entries[item].shape)
            else:
                ndim = self.data.ndim - 1
            idx = [item] + [slice(None)] * (ndim)
            is_asynchronous = isinstance(self.data, lardon.OfflineDataList)
            if sequence_length is not None:
                if sequence_mode == "random":
                    if is_asynchronous:
                        idx[sequence_idx] = lardon.randslice(sequence_length)
                    else:
                        seq_idx = random.randrange(0, self.data[item].shape[sequence_idx] - sequence_length)
                        idx[sequence_idx] = slice(seq_idx, seq_idx + sequence_length)
            if is_asynchronous:
                data, meta = self.data.__getitem__(tuple(idx), return_metadata=True, return_indices=True)
                seq_idx = meta['idx'][sequence_idx]
            else:
                data = self.data.__getitem__(idx)
        return data, seq_idx

    def _get_item_metadata(self, item: int, seq=None):
        if hasattr(item, "__iter__"):
            metadatas = [self._get_item_metadata(item[i], seq[i]) for i in range(len(item))]
            compiled_metadata = {}
            for k in metadatas[0].keys():
                try:
                    compiled_metadata[k] = torch.stack([m[k] for m in metadatas])
                except:
                    compiled_metadata[k] = [m[k] for m in metadatas]
            return compiled_metadata
                
        seq = seq if seq is not None else slice(None)
        if hasattr(seq, "__iter__"):
            metadata = {'time': torch.Tensor(self.metadata['time'][item][seq])}
        else:
            metadata = {'time': torch.Tensor([self.metadata['time'][item][seq]])}
        for k in self._active_tasks+['sr']:
            current_meta = self.metadata[k][item]
            if isinstance(current_meta, ContinuousList):
                current_meta = current_meta[metadata['time']]
            elif hasattr(current_meta, '__iter__'):
                current_meta = current_meta[seq]
            if isinstance(current_meta, int):
                metadata[k] = torch.LongTensor([current_meta])
            elif isinstance(current_meta, float):
                metadata[k] = torch.FloatTensor([current_meta])
            else:
                metadata[k] = torch.Tensor(current_meta)
        return metadata

    def make_partitions(self, names, balance, from_files=True):
        """
        Builds partitions from the data
        Args:
            names (list[str]) : list of partition names
            balance (list[float]) : list of partition balances (must sum to 1)
        """
        partition_files = {}
        if from_files:
            files = list(self.hash.keys())
            permutation = np.random.permutation(len(files))
            cum_ids = np.cumsum([0] + [int(n * len(files)) for n in balance])
            partitions = {}
            partition_files = {}
            for i, n in enumerate(names):
                partition_files[n] = [files[f] for f in permutation[cum_ids[i]:cum_ids[i + 1]]]
                partitions[n] = sum([checklist(self.hash[f]) for f in partition_files[n]], [])
        else:
            permutation = np.random.permutation(len(self.data))
            cum_ids = np.cumsum([0]+[int(n*len(self.data)) for n in balance])
            partitions = {}
            for i, n in enumerate(names):
                partitions[n] = permutation[cum_ids[i]:cum_ids[i+1]]
                partition_files[n] = [self.files[n] for n in permutation[cum_ids[i]:cum_ids[i+1]]]
        self.partitions = partitions
        self.partition_files = partition_files

    @property
    def has_sequences(self):
        return isinstance(self.data[0], list)

    def drop_sequences(self, sequence_length, idx=None, mode="random"):
        if sequence_length is None:
            self._sequence_mode = None
            self._sequence_length = None
            self._sequence_idx = idx
        else:
            # assert self.has_sequences, "Dataset %s does not contain sequences"%self
            self._sequence_length = sequence_length
            self._sequence_mode = mode
            self._sequence_idx = 0 if idx is None else idx

    def retrieve(self, item):
        """
        Create a sub-dataset containing target items
        Args:
            item (iter[int] or str) : target data ids / partition

        Returns:
            subdataset (MidiDataset) : obtained sub-dataset
        """
        if isinstance(item, list):
            item = np.array(item)
        elif isinstance(item, torch.LongTensor):
            item = item.detach().numpy()
        elif isinstance(item, int):
            item = np.array([item])
        elif isinstance(item, str):
            item = self.partitions[item]

        dataset = type(self)(self.root_directory, sr = self.sr, target_length=self.target_length, transforms=self._transforms)

        if isinstance(self.data, lardon.OfflineDataList):
            dataset.data = lardon.OfflineDataList([self.data.entries[d] for d in item])
        else:
            dataset.data = [self.data[d] for d in item]
        dataset.metadata = {k: (np.array(v)[item]).tolist() for k, v in self.metadata.items()}
        dataset.files = [self.files[f] for f in item]
        dataset.hash = {}
        for i, f in enumerate(dataset.files):
            dataset.hash[f] = dataset.hash.get(f, []) + [i]
        dataset._active_tasks = self._active_tasks
        dataset._sequence_length = self._sequence_length
        dataset._sequence_mode = self._sequence_mode
        dataset._sequence_idx = self._sequence_idx
        dataset._pre_transform = self._pre_transform
        dataset._dtype = self._dtype
        return dataset

    def retrieve_from_class(self, task, labels):
        valid_classes = [self.classes[task][i] for i in labels]
        ids = np.where(np.in1d(self.metadata[task], valid_classes))[0]
        return self.retrieve(ids)

    def __delitem__(self, key):
        del self.data.entries[key]
        del self.files[key]
        for k in self.metadata.keys():
            del self.metadata[k][key]

    def parse_files(self):
        """
        parses root directory
        """
        files = []
        for r, d, f in os.walk(self.root_directory+"/data"):
            for f_tmp in f:
                if os.path.splitext(f_tmp)[1] in self.types:
                    f_tmp = re.sub(os.path.abspath(self.root_directory+'/data'), '', os.path.abspath(r+'/'+f_tmp))[1:]
                    files.append(f_tmp)
        self.files = files
        self.hash = {self.files[i]:i for i in range(len(self.files))}

    def import_data(self, flatten=False, scale=None, write_transforms=False, save_transform_as=None):
        """
        Imports data from root directory (must be parsed beforehand)
        Args:
            flatten (bool): if False, created nested arrays for each file. If True, creates a flattened array
            scale (bool): performs transform scale after import
        """
        data = []
        metadata = {}
        files = []
        hash = {}
        running_id = 0
        for i, f in tqdm(enumerate(self.files), total=len(self.files), desc="Importing audio files..."):
            current_data, current_metadata = parse_audio_file(os.path.abspath(self.root_directory+'/'+f), sr=self.sr, len=self.target_length)
            if len(metadata.keys()) == 0:
                metadata = {k: [] for k in current_metadata.keys()}
            if flatten:
                data.extend(current_data)
                files.extend([self.files[i]]*len(current_data))
                for k, v in current_metadata.items():
                    metadata[k].extend(checklist(v))
                hash[f] = list(range(running_id, running_id+len(current_data)))
                running_id +=  len(current_data)
            else:
                data.append(current_data)
                hash[f] = running_id
                for k, v in current_metadata.items():
                    metadata[k].append(v)
                running_id += 1
                files.append(self.files[i])
                hash[self.files[i]] = i

        self.data = data
        self.hash = hash
        self.files = files

        self.metadata = {**self.import_metadata(), **metadata}
        if write_transforms:
            self.write_transforms(save_as=save_transform_as)

    def scale_transform(self, scale):
        if scale and self._transforms is not None:
            if scale is True:
                self._transforms.scale(self.data[:])
            elif isinstance(scale, int):
                idx = torch.randperm(len(self.data))[:scale]
                scale_data = [self.data[i.item()] for i in idx]
                self._transforms.scale(scale_data)

    def import_transform(self, transform):
        assert transform in self.available_transforms
        target_directory = f"{self.root_directory}/transforms/{transform}"
        if len(self.files) > 0:
            files = [os.path.splitext(f)[0] + lardon.lardon_ext for f in self.files]
        else:
            files = None
        self.data, self.metadata = lardon.parse_folder(target_directory, drop_metadata=True, files=files)
        with open(target_directory+'/transforms.ct', 'rb') as f:
            original_transform = dill.load(f)
        with open(target_directory + '/dataset.ct', 'rb') as f:
            save_dict = dill.load(f)
            self.load_dict(save_dict)
        self._pre_transform = original_transform
        self.metadata = {**self.metadata, **self.import_metadata()}
        return original_transform

    def import_metadata(self):
        metadata_directory = f"{self.root_directory}/metadata"
        if not os.path.isdir(metadata_directory):
            return {}

        task_test = lambda x: os.path.isdir(f"{metadata_directory}/{x}") \
                              and os.path.isfile(f"{metadata_directory}/{x}/metadata.txt")
        tasks = list(filter(task_test, os.listdir(metadata_directory)))

        metadata = {}
        for t in tasks:
            if os.path.isfile(f"{metadata_directory}/{t}/callback.txt"):
                with open(f"{metadata_directory}/{t}/callback.txt", 'r') as f:
                    callback = re.sub('\n', '', f.read())
                callback = metadata_hash.get(callback)
            else:
                callback = metadata_hash.get(t)
            if callback is None:
                print('[Warning] could not find callback for task %s'%t)
                continue
            if os.path.isfile(f"{metadata_directory}/{t}/classes.txt"):
                self.classes[t] = {}
                with open(f"{metadata_directory}/{t}/classes.txt", 'r') as f:
                    for l in f.readlines():
                        class_id, name = re.sub('\n', '',l).split('\t')
                        self.classes[t][int(class_id)] = name
            current_hash = {}
            current_directory = f"{metadata_directory}/{t}"
            metadata[t] = [None] * len(self.files)
            with open(f"{current_directory}/metadata.txt") as metafile:
                for line in metafile.readlines():
                    file, raw_metadata = line.replace('\n', '').split('\t')
                    current_hash[file] = raw_metadata
            for file, idx in self.hash.items():
                if file in current_hash.keys():
                    current_metadata = callback(current_hash[file], current_path=current_directory)
                    metadata[t][idx] = current_metadata
                else:
                    print('warning : file %s not found in metadata'%(file, ))
        return metadata

    def write_transforms(self, save_as=None, force=False):
        """
        Write transforms in place, parsing the files to the target transform using lardon.
        Note : dataset should not be flattened, providing degenerated lardon pickling.
        :param name: transform name
        :param selector:
        :return:
        """
        transformed_meta = []
        target_directory = f"{self.root_directory}/transforms/{save_as}"
        if save_as:
            with lardon.LardonParser(self.root_directory+'/data', target_directory, force=force) as parser:
                for i, d in enumerate(tqdm(self.data, desc="exporting transforms...", total=len(self.data))):
                    time = torch.tensor(0.) if not "time" in self.metadata['time'] else self.metadata['time'][i]
                    new_data, new_time = self._transforms(d, time=time, sr=self.metadata['sr'][i])
                    new_data = checknumpy(new_data)
                    transformed_meta.append({'time':new_time.numpy(), 'sr':self.metadata['sr'][i]})
                    if save_as is not None:
                        parser.register(new_data, transformed_meta[-1], filename=(f"{self.root_directory}/{self.files[i]}"))
                    # transformed_data.append(new_data.numpy()); transformed_meta.append({'time': new_time.numpy()})
                    # transformed_data.append(selector(new_data)); transformed_time.append(selector(new_time))
            transformed_data = lardon.parse_folder(target_directory)
            transformed_meta = {k: [t[k] for t in transformed_meta] for k in transformed_meta[0].keys()}
            with open(f"{target_directory}/transforms.ct", 'wb') as f:
                dill.dump(self._transforms, f)
            save_dict = self.get_attributes()
            with open(f"{target_directory}/dataset.ct", "wb") as f:
                dill.dump(save_dict, f)
        else:
            transformed_data = []
            for i, d in enumerate(tqdm(self.data, desc="replacing transforms...", total=len(self.data))):
                time = torch.tensor(0.) if not "time" in self.metadata['time'] else self.metadata['time'][i]
                new_data, new_time = self._transforms(d, time=time, sr=self.metadata['sr'][i])
                transformed_data.append(new_data)
                transformed_meta.append({'time': new_time, 'sr': self.metadata['sr'][i]})
            transformed_meta = {k: [t[k] for t in transformed_meta] for k in transformed_meta[0].keys()}
        self._transforms = AudioTransform()
        self.data = transformed_data
        self.metadata = {**self.metadata, **transformed_meta}


    def flatten_data(self, axis=0):
        data = []
        metadata = {k: [] for k in self.metadata.keys()}
        files = []
        hash = {}
        running_id = 0
        for i, d in tqdm(enumerate(self.data), desc="Flattening data...", total=len(self.data)):
            if isinstance(self.data, lardon.OfflineDataList):
                current_data = self.data.entries[i].scatter(axis)
            else:
                current_data = self.data[i].split(1, dim=axis)
            data.extend(current_data)
            if self.metadata.get('time'):
                time = self.metadata['time'][i]
                metadata['time'].extend(np.zeros_like(time))
            for k in self.metadata.keys():
                if k == "time":
                    continue
                if self.metadata[k][i] is None:
                    continue
                elif isinstance(self.metadata[k][i], ContinuousList):
                    metadata[k].extend(self.metadata[k][i][time])
                elif isinstance(self.metadata[k][i], (np.ndarray)) or torch.is_tensor(self.metadata[k][i]):
                    if self.metadata[k][i].shape[0] == len(current_data):
                        metadata[k].append(self.metadata[k][i])
                    else:
                        metadata[k].append(self.metadata[k][i].repeat(len(current_data)))
                else:
                    metadata[k].extend([self.metadata[k][i]]*len(current_data))
            hash[self.files[i]] = list(range(running_id, running_id + len(current_data)))
            files.extend([self.files[i]] * len(current_data))
            running_id += len(current_data)

        if isinstance(data[0], lardon.OfflineEntry):
            self.data = lardon.OfflineDataList(data)
        else:
            self.data = data
        self.metadata = metadata
        if self.metadata.get('time'):
            self.metadata['time'] = np.array(self.metadata['time'])[:, np.newaxis]
        self.hash = hash
        self.files = files
        self._sequence_idx = None
        self._sequence_length = None


    def get_attributes(self):
        """
        Get a dict version of the dataset (conveniant for pickling)
        Returns:
            save_dict (dict) : dictionary
        """
        return {'files':self.files,
                'hash':self.hash,
                'transforms':self._transforms,
                'root_directory':self.root_directory,
                'partitions':self.partitions,
                'partition_files': self.partition_files,
                'target_length': self.target_length,
                'pre_transform': self._pre_transform,
                'sequence_mode': self._sequence_mode,
                'sequence_length': self._sequence_length,
                'sequence_idx':self._sequence_idx}

    def save(self, name, write_transforms=False, **kwargs):
        """Manage folders and save data to disk

        Args:
            name (string): dataset instance name
            write_transforms (bool, optional): If True saves the current instance as a dict. Defaults to True.
        """
        if not os.path.isdir(self.root_directory+'/pickles'):
            os.makedirs(self.root_directory+'/pickles')
        if write_transforms:
            self.write_transforms(name, flatten=False)
        save_dict = {**self.get_attributes(), **kwargs}
        save_dict['data'] = self.data
        save_dict['metadata'] = self.metadata
        filename = f"{self.root_directory}/pickles/{name}.ctp"
        with open(filename, 'wb') as savefile:
            dill.dump(save_dict, savefile)

    def load_dict(self, save_dict):
        files = []; hash = {}
        for f in save_dict['files']:
            new_file = re.sub(save_dict['root_directory'], self.root_directory, f)
            files.append(new_file)
            hash[new_file] = save_dict['hash'][f]
        self.files = files
        self.hash = hash
        if save_dict.get('data'):
            self.data = save_dict.get('data')
        if save_dict.get('metadata'):
            self.metadata = save_dict.get('metadata')
        self.target_length = save_dict.get('target_length')
        self.partitions = save_dict.get('partitions', {})
        self.partition_files = save_dict.get('partition_files')
        self._active_tasks = save_dict.get('active_tasks', [])
        self._pre_transform =save_dict.get('pre_transform')
        self._sequence_mode = save_dict.get('sequence_mode')
        self._sequence_length = save_dict.get('sequence_length')
        self._sequence_idx = save_dict.get('sequence_idx')

    @classmethod
    def load(cls, filename, root_prefix, drop_dict=False):
        """Load a MidiDataset instance from save dict

        Args:
            filename (string): filename
            root_prefix (string): root folder

        Returns:
            MidiDataset: a MidiDataset instance
        """
        filename = f"{root_prefix}/pickles/{filename}"
        with open(filename, 'rb') as savefile:
            save_dict = dill.load(savefile)
        dataset = AudioDataset(root_prefix,
                              transforms=save_dict['transforms'],
                              target_length=save_dict['target_length'],
                              tasks=save_dict['tasks'])
        dataset.load_dict(save_dict)
        if drop_dict:
            return dataset, save_dict
        else:
            return dataset

    def transform_file(self, file):
        data, meta = parse_audio_file(file, self.sr, len=self.target_length)
        if self._pre_transform is not None:
            data, meta['time'] = self._pre_transform(data, time=meta['time'])
        data, meta['time'] = self._transforms(data, time=meta['time'])
        return data, meta

    def invert_data(self, data):
        if self._transforms.invertible:
            inv_data = self._transforms.invert(data)
        else:
            raise NotInvertibleError
        if self._pre_transform is not None:
            if self._pre_transform.invertible:
                inv_data = self._pre_transform.invert(data)
            else:
                raise NotInvertibleError
        return inv_data


