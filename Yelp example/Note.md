## Example : Classifying Sentiment of Restaurant Reviews

### Data Vectorization Classes

#### The Vocabulary

Class to process text and extract vocabulary for mapping

```python
class Vocabulary(object):
```

##### `__int__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>")`

- `token_to_idx`：一个可选的字典，表示已有的词汇到索引的映射。
- `add_unk`：一个布尔值，指示是否添加未登录词（unknown token）。
- `unk_token`：表示未登录词的字符串。

```python
def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
    if token_to_idx is None:
        token_to_idx = {}
    self._token_to_idx = token_to_idx

    self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}
    
    self._add_unk = add_unk
    self._unk_token = unk_token
    
    self.unk_index = -1
    if add_unk:
        self.unk_index = self.add_token(unk_token) 
```

- 如果 `token_to_idx` 为 `None`，则将其初始化为空字典。
- `self._token_to_idx` 存储词汇到索引的映射。
- `self._idx_to_token` 是索引到词汇的映射，通过反转 `self._token_to_idx` 创建。
- `self.unk_index` 初始化为 -1，表示未登录词的索引。
- 如果 `add_unk` 为 `True`，则调用 `self.add_token(unk_token)` 方法将未登录词添加到词汇表，并存储其索引。

##### `add_token(self, token)`

- `token (str)`：要添加到词汇表中的词汇。
- `index (int)`：与词汇对应的整数索引。

```python
def add_token(self, token):
    if token in self._token_to_idx:
        index = self._token_to_idx[token]
    else:
        index = len(self._token_to_idx)
        self._token_to_idx[token] = index
        self._idx_to_token[index] = token
    return index
```

- 首先检查 `token` 是否已经存在于 `_token_to_idx` 映射字典中。如果存在，直接获取该词汇的索引 `index`。
- 如果 `token` 不存在于 `_token_to_idx` 中, 计算新的索引值 `index`，它等于当前 `_token_to_idx` 的长度（即已有词汇的数量）。
- 将 `token` 和 `index` 添加到 `_token_to_idx` 字典中。同时，在 `_idx_to_token` 字典中，将 `index` 和 `token` 添加进去，保持两个字典的同步。

##### `add_many(self.tokens)`

将一个字符串列表（即多个词汇）添加到词汇表中。它会调用 `add_token` 方法为每个词汇添加索引，并返回这些词汇对应的索引列表。

```python
def add_many(self, tokens):
    return [self.add_token(token) for token in tokens]
```

##### `to_serializable(self)`

用于将 `Vocabulary` 对象的状态转换为一个可以序列化的字典。这对于将对象保存到文件或通过网络传输非常有用。

```python
def to_serializable(self):
    return {'token_to_idx': self._token_to_idx, 
            'add_unk': self._add_unk, 
            'unk_token': self._unk_token}
```

##### `from_serializable(cls,contents)`

是一个类方法，用于从一个可序列化的字典实例化 `Vocabulary` 对象。这对于从文件或其他数据源加载保存的 `Vocabulary` 对象非常有用。

```python
@classmethod
def from_serializable(cls, contents):
    return cls(**contents)
```

- 使用 `@classmethod` 装饰器将 `from_serializable` 方法定义为类方法。
- 类方法的第一个参数是类本身（通常命名为 `cls`），而不是实例。
- `contents` 是一个字典，包含 `Vocabulary` 对象的状态。
- 使用 `cls(**contents)` 创建并返回一个新的 `Vocabulary` 对象。`**contents` 是解包字典的语法，将字典中的键值对作为关键字参数传递给类的构造函数。

##### `lookup_token(self,token)`

```python
def lookup_token(self, token):
    if self.unk_index >= 0:
        return self._token_to_idx.get(token, self.unk_index)
    else:
        return self._token_to_idx[token]
```

- 如果 `unk_index` 大于等于 0，说明已添加 UNK 词汇。（只有当`add_UNK=False`的时候才会等于-1＜0）。使用 `get` 方法从 `_token_to_idx` 字典中查找 `token` 对应的索引。如果 `token` 不存在(不属于`Vocabulary`范围内的词)，则返回 `unk_index`。
- 如果未添加 UNK 词汇（即 `unk_index` 小于 0也就是`add_UNK=False`），直接从 `_token_to_idx` 字典中查找 `token` 对应的索引。如果 `token` 不存在，则会引发 `KeyError` 异常。

##### `lookup_index(self.index)`

用于检索与给定索引相关联的词汇。如果索引不存在于词汇表中，则引发 `KeyError` 异常。

```python
def lookup_index(self, index):
    if index not in self._idx_to_token:
        raise KeyError("the index (%d) is not in the Vocabulary" % index)
    return self._idx_to_token[index]
```

##### `__str__(self)`

用于返回 `Vocabulary` 对象的字符串表示 `Vocabulary` 对象的大小。这通常用于打印对象时的输出。

```python
def __str__(self):
    return "<Vocabulary(size=%d)>" % len(self)
```

##### `__len__(self)`

用于返回 `Vocabulary` 对象中词汇的数量。

```python
def __len__(self):
    return len(self._token_to_idx)
```



#### The Vectorizer

```python
from collections import Counter
import string
import numpy as np
class ReviewVectorizer(object):
```

`ReviewVectorizer` 类用于协调两个 `Vocabulary` 对象：一个用于映射单词到整数（`review_vocab`），另一个用于映射类别标签到整数（`rating_vocab`）。这个类主要是将词汇表和标签表结合起来使用。

##### `__init__(self,review_vocab,rating_vocab)`

```python
def __init__(self, review_vocab, rating_vocab):
    self.review_vocab = review_vocab
    self.rating_vocab = rating_vocab
```

- `review_vocab (Vocabulary)`：用于映射单词到整数的词汇表对象。
- `rating_vocab (Vocabulary)`：用于映射类别标签到整数的词汇表对象。

##### `vectorize(self,review)`

用于将给定的评论（`review`）转换为一个折叠的一热向量表示。这个向量表示评论中单词的存在性。

```python
def vectorize(self, review):
    one_hot = np.zeros(len(self.review_vocab), dtype=np.float32)
    
    for token in review.split(" "):
        if token not in string.punctuation:
            one_hot[self.review_vocab.lookup_token(token)] = 1

    return one_hot
```

- 创建一个长度等于词汇表大小的零向量，数据类型为 `np.float32`。
- 将评论字符串按空格拆分为单词列表。对每个单词，检查其是否为标点符号。如果不是标点符号，则查找单词对应的索引，并将一热向量中对应位置的值设为 `1`

##### `from_dataframe(cls,review_df,cutoff=25)`

是一个类方法，用于从数据集的 DataFrame 实例化 `ReviewVectorizer` 对象。它根据词频阈值（`cutoff` 参数）筛选并添加词汇到词汇表中，并将类别标签添加到另一个词汇表中。

```python
@classmethod
def from_dataframe(cls, review_df, cutoff=25):
    review_vocab = Vocabulary(add_unk=True)
    rating_vocab = Vocabulary(add_unk=False)
    
    # Add ratings
    for rating in sorted(set(review_df.rating)):
        rating_vocab.add_token(rating)

    # Add top words if count > provided count
    word_counts = Counter()
    for review in review_df.review:
        for word in review.split(" "):
            if word not in string.punctuation:
                word_counts[word] += 1

    for word, count in word_counts.items():
        if count > cutoff:
            review_vocab.add_token(word)

    return cls(review_vocab, rating_vocab)

```

- 遍历评论数据集中所有唯一的类别标签，并将其添加到 `rating_vocab` 中。(别的数据集中不一定有`rating`列)
- 使用 `Counter` 统计评论数据集中每个单词的频次。`string.punctuation`是标点符号。
- 遍历词频统计结果，只有词频超过 `cutoff` 值的词汇才添加到 `review_vocab` 中。
- 使用 `review_vocab` 和 `rating_vocab` 初始化 `ReviewVectorizer` 对象并返回

##### `from_serializable(cls,contents)`

是一个类方法，用于从一个可序列化的字典实例化 `ReviewVectorizer` 对象。该方法利用字典中的数据恢复 `ReviewVectorizer` 对象的状态。

```python
@classmethod
def from_serializable(cls, contents):

    review_vocab = Vocabulary.from_serializable(contents['review_vocab'])
    rating_vocab =  Vocabulary.from_serializable(contents['rating_vocab'])

    return cls(review_vocab=review_vocab, rating_vocab=rating_vocab)
```

- `contents (dict)`：包含 `ReviewVectorizer` 对象状态的可序列化字典。(两个键值对，值也是字典包含`Vocabulary`类的状态(3个参数))
- 使用 `Vocabulary` 类的 `from_serializable` 方法，从字典中的数据恢复 `review_vocab` 和 `rating_vocab` 对象。
- 使用恢复的 `review_vocab` 和 `rating_vocab` 初始化 `ReviewVectorizer` 对象并返回。

##### `to_serializable(self)`

用于创建一个可序列化的字典，以便缓存 `ReviewVectorizer` 对象的状态。这个方法将 `ReviewVectorizer` 对象中的词汇表对象转换为可序列化的字典。

```python
def to_serializable(self):
    return {'review_vocab': self.review_vocab.to_serializable(),
            'rating_vocab': self.rating_vocab.to_serializable()}
```

- 调用 `review_vocab` 和 `rating_vocab` 的 `to_serializable` 方法，将它们转换为可序列化的字典。 创建并返回包含这些字典的字典。

#### The Dataset

```python
from torch.utils.data import Dataset, DataLoader
class ReviewDataset(Dataset):
```

`ReviewDataset` 类继承自 `Dataset` 类，负责处理评论数据集。该类将数据集划分为训练集、验证集和测试集，并根据数据集分片选择相应的子集。

在 PyTorch 中，`torch.utils.data.Dataset` 是一个抽象基类，用于表示数据集。自定义数据集类需要继承这个基类并实现以下方法：

1. **`__len__` 方法**：返回数据集的大小。
2. **`__getitem__` 方法**：支持通过索引获取数据集中的元素。

##### `__init__(self,review_df,vectorizer)`

用于初始化 `ReviewDataset` 对象，接收数据集 DataFrame 和 `ReviewVectorizer` 对象作为参数，并将数据集分为训练集、验证集和测试集。

```python
def __init__(self, review_df, vectorizer):
    self.review_df = review_df
    self._vectorizer = vectorizer

    self.train_df = self.review_df[self.review_df.split == 'train']
    self.train_size = len(self.train_df)

    self.val_df = self.review_df[self.review_df.split == 'val']
    self.validation_size = len(self.val_df)

    self.test_df = self.review_df[self.review_df.split == 'test']
    self.test_size = len(self.test_df)

    self._lookup_dict = {
        'train': (self train_df, self.train_size),
        'val': (self.val_df, self.validation_size),
        'test': (self.test_df, self.test_size)
    }

    self.set_split('train') 
```

- 根据 `split` 列将数据集划分为训练集、验证集和测试集，并计算每个子集的大小。
- 创建一个字典`lookup_dict`，将每个分片名称映射到相应的 DataFrame 和大小。
- 调用 `set_split` 方法，将默认分片设置为训练集。

##### `set_split(self,split="train")`

```python
def set_split(self, split="train"):
    self._target_split = split
    self._target_df, self._target_size = self._lookup_dict[split]
```

- 使用 `split` 参数值查找 `_lookup_dict` 字典中对应的数据子集和大小，并分别存储到 `self._target_df` 和 `self._target_size` 属性中。

##### `load_dataset_and_make_vectorizer(cls,review_csv)`

是一个类方法，用于加载数据集并从头开始创建一个新的向量化器。该方法从指定的 CSV 文件加载数据集，并基于训练集数据创建 `ReviewVectorizer` 对象，最终返回一个 `ReviewDataset` 实例。

```python
@classmethod
def load_dataset_and_make_vectorizer(cls, review_csv):
    review_df = pd.read_csv(review_csv)
    train_review_df = review_df[review_df.split == 'train']
    return cls(review_df, ReviewVectorizer.from_dataframe(train_review_df))
```

- 使用 pandas 读取 CSV 文件，加载数据集到 `review_df` DataFrame。
- 使用训练集 DataFrame 创建 `ReviewVectorizer` 对象，并使用原始数据集 DataFrame 和创建的向量化器对象初始化 `ReviewDataset` 对象，最后返回该实例。

##### `load_dataset_and_load_vectorizer(cls, review_csv, vectorizer_filepath)`

一个类方法，用于加载数据集及其对应的向量化器。在向量化器已经缓存并保存到文件中以供重用的情况下使用该方法。

```python
@classmethod
def load_dataset_and_load_vectorizer(cls, review_csv, vectorizer_filepath):
    review_df = pd.read_csv(review_csv)
    vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
    return cls(review_df, vectorizer)

```

- 使用 pandas 读取 CSV 文件，加载数据集到 `review_df` DataFrame。
- 调用 `load_vectorizer_only` 方法，从指定路径加载保存的向量化器。
- 使用加载的数据集 DataFrame 和向量化器对象初始化 `ReviewDataset` 对象，最后返回该实例。

##### `load_vectornizer_only(vectorizer_filepaht)`

是一个静态方法，用于从文件中加载序列化的向量化器。该方法从指定文件路径读取向量化器的序列化数据，并返回一个 `ReviewVectorizer` 实例。

```python
@staticmethod
def load_vectorizer_only(vectorizer_filepath):
    with open(vectorizer_filepath) as fp:
        return ReviewVectorizer.from_serializable(json.load(fp))
```

- 使用 `open` 函数打开指定路径的文件。
- 使用 `json.load` 函数读取文件内容，并将其作为参数传递给 `ReviewVectorizer.from_serializable` 方法，返回一个 `ReviewVectorizer` 实例。

##### `save_vectorizer(self, vectorizer_filepath)`

用于将向量化器保存到磁盘上的文件中。该方法使用 JSON 格式将向量化器序列化并保存到指定的文件路径。

```python
def save_vectorizer(self, vectorizer_filepath):
    with open(vectorizer_filepath, "w") as fp:
        json.dump(self._vectorizer.to_serializable(), fp)
```

- 使用 `open` 函数以写模式打开指定路径的文件。
- 使用 `json.dump` 函数将向量化器的序列化表示保存到文件中。

##### `get_vectorizer(self)`

```python
def get_vectorizer(self):
    return self._vectorizer
```

##### `__len__(self)`

用于返回数据集当前选择分片的大小。它是 Python 内置的特殊方法，使得类的实例可以使用内置函数 `len()` 来获取对象的大小。

```python
def __len__(self):
    return self._target_size
```

- `self._target_size` 是由 `set_split` 方法设置的，表示当前选择的分片（例如，训练集、验证集或测试集）的大小。
- 在调用 `set_split` 方法后，`self._target_size` 会被更新为相应分片的数据集大小，因此 `__len__` 方法返回的值取决于最近一次调用 `set_split` 方法时选择的分片。

##### `__gititem__(self,index)`

是 `ReviewDataset` 类的一个核心方法，使其能够兼容 PyTorch 数据集的使用方式比如切片`[]`和。该方法根据索引值返回数据集中对应的数据点，包括特征向量和标签。

```python
def __getitem__(self, index):
    row = self._target_df.iloc[index]

    review_vector = self._vectorizer.vectorize(row.review)

    rating_index = self._vectorizer.rating_vocab.lookup_token(row.rating)

    return {'x_data': review_vector, 'y_target': rating_index}
```

- 使用 `vectorize` 方法将评论文本转换为特征向量。
- 使用 `lookup_token` 方法将标签转换为对应的索引值。
- 返回一个包含特征向量和标签索引的字典。

##### `get_num_batches(self, batch_size)`

用于根据给定的批量大小返回数据集中的批次数量。该方法计算在指定的批量大小下，可以从数据集中提取的完整批次数量。

```python
def get_num_batches(self, batch_size):
    return len(self) // batch_size
```

##### 类外函数`generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu")`

```python
def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict
```

`DatalLoader`是 PyTorch 中的数据加载器，用于处理和加载数据集。在深度学习训练过程中，数据加载器通过提供批量数据的方式大大简化了数据的读取和处理过程。`DataLoader` 可以处理大多数常见的数据操作，如数据集的随机打乱、批量化、并行加载等。

**关键参数**

- `dataset`：需要加载的数据集，必须是 `torch.utils.data.Dataset` 的子类。
- `batch_size`：每个批次的数据量大小。
- `shuffle`：是否在每个 epoch 开始时打乱数据。
- `num_workers`：用于数据加载的子进程数量。默认为 0，表示数据将在主进程中加载。
- `drop_last`：如果数据集大小不能被批次大小整除，是否丢弃最后一个不完整的批次。

**示例代码：**

```python
import torch
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return {'data': sample, 'label': label}

# 示例数据
data = torch.randn(100, 3, 32, 32)  # 100个样本，每个样本是3x32x32的图像
labels = torch.randint(0, 10, (100,))  # 100个标签，每个标签是0-9的整数

# 创建数据集对象
dataset = CustomDataset(data, labels)

# 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2, drop_last=True)

# 遍历 DataLoader
for batch in dataloader:
    print(batch['data'].shape, batch['label'].shape)
    
Output:
  torch.Size([10, 3, 32, 32]) torch.Size([10])
  torch.Size([10, 3, 32, 32]) torch.Size([10])
  torch.Size([10, 3, 32, 32]) torch.Size([10])
  torch.Size([10, 3, 32, 32]) torch.Size([10])
  torch.Size([10, 3, 32, 32]) torch.Size([10])
  torch.Size([10, 3, 32, 32]) torch.Size([10])
  torch.Size([10, 3, 32, 32]) torch.Size([10])
  torch.Size([10, 3, 32, 32]) torch.Size([10])
  torch.Size([10, 3, 32, 32]) torch.Size([10])
  torch.Size([10, 3, 32, 32]) torch.Size([10])
```



### The Model:`ReviewClassifier`

```python
class ReviewClassifier(nn.Module):

    def __init__(self, num_features):
        super(ReviewClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=num_features, 
                             out_features=1)

    def forward(self, x_in, apply_sigmoid=False):
        y_out = self.fc1(x_in).squeeze()
        # 使用 squeeze 函数去掉输出张量中的单维度，以确保输出形状为 (batch_size,)
        if apply_sigmoid:
            y_out = torch.sigmoid(y_out)
        return y_out
```



### Training Routine

#### Helper Functions

##### `make_train_state(args)`

用于初始化和返回一个字典，用于在训练过程中跟踪和管理训练状态。该字典包含关于训练过程的各种信息，如早停状态、学习率、当前 epoch 索引、训练和验证损失、训练和验证准确性等。

```python
def make_train_state(args):
    return {'stop_early': False,# 指示是否提前停止训练
            'early_stopping_step': 0, # 记录早停计数器的步数,大于某个值时停止训练
            'early_stopping_best_val': 1e8, # 用于早停的最佳验证损失
            'learning_rate': args.learning_rate, # 学习率
            'epoch_index': 0, # 当前 epoch 的索引
            'train_loss': [], # 记录每个 epoch 的训练损失
            'train_acc': [], # 记录每个 epoch 的训练准确性
            'val_loss': [], # 记录每个 epoch 的验证损失
            'val_acc': [], # 记录每个 epoch 的验证准确性
            'test_loss': -1, # 测试损失
            'test_acc': -1, # 测试准确率
            'model_filename': args.model_state_file} # 模型保存文件名
```

##### `update_train_state(args, model, train_state)`

用于管理和更新训练状态，特别是实现**早停机制和保存模型检查点**。如果模型在验证集上的性能有所提升，函数会保存模型的状态；如果验证损失不再降低，函数会更新早停计数器，决定是否提前停止训练。

- `args`：包含训练相关参数的对象。
- `model`：当前训练的模型。
- `train_state`：表示训练状态的字典

```python
def update_train_state(args, model, train_state):
    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]

        # If loss worsened
        if loss_t >= train_state['early_stopping_best_val']:
            # Update step
            train_state['early_stopping_step'] += 1
        # Loss decreased
        else:
            # Save the best model
            if loss_t < train_state['early_stopping_best_val']:
           # model.state_dict() 方法返回包含模型所有参数和持久缓冲区的字典。
                torch.save(model.state_dict(), train_state['model_filename']) 
         	

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state
```

- 在第一个 epoch，保存模型状态，并将 `stop_early` 设置为 `False`。
- 在每个 epoch 结束时，检查验证损失的变化。
- 如果验证损失增加，更新早停计数器。
- 如果验证损失减少，保存模型状态并重置早停计数器。
- 检查早停计数器是否达到预设的早停标准，如果达到则设置 `stop_early` 为 `True`。

##### `compute_accuracy(y_pred, y_target)`

```python
def compute_accuracy(y_pred, y_target):
    y_target = y_target.cpu()
    y_pred_indices = (torch.sigmoid(y_pred)>0.5).cpu().long()#.max(dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100
```

##### `set_seed_everywhere(seed,cuda)`

```python
def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
```

用于设置全局随机种子，以确保代码的可重复性。它设置 NumPy 和 PyTorch 的随机种子，如果使用 CUDA 进行计算，还会设置所有 GPU 设备的随机种子

##### `handle_dirs`

```python
def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
```

用于检查指定的目录路径是否存在。如果目录不存在，则创建该目录。这对于确保在文件操作之前目标目录已经存在非常有用。

#### Setting and some prep work

##### `args=Namespace()`

```python
args = Namespace(
    # Data and Path information
    frequency_cutoff=25, # 用于词汇构建的频率截止值，低于该频率的词不会包含在词汇表中。
    model_state_file='model.pth', # 保存模型状态的文件名。
    review_csv='data/yelp/reviews_with_splits_lite.csv',
    # review_csv='data/yelp/reviews_with_splits_full.csv',
    save_dir='model_storage/ch3/yelp/', # 模型保存目录。
    vectorizer_file='vectorizer.json', # 保存向量化器的文件名。
    # No Model hyper parameters
    # Training hyper parameters
    batch_size=128, # 每个批次的样本数量。
    early_stopping_criteria=5, # 早停标准，如果在指定次数的 epoch 内验证损失没有下降，则停止训练。
    learning_rate=0.001, # 学习率。
    num_epochs=100, # 训练的最大 epoch 数。
    seed=1337, # 随机种子，用于确保结果的可重复性。
    # Runtime options
    catch_keyboard_interrupt=True, # 是否捕获键盘中断，以便在训练过程中可以安全地中断训练。
    cuda=True, # 是否使用 CUDA（GPU）
    expand_filepaths_to_save_dir=True, # 是否将文件路径扩展到保存目录。
    reload_from_files=False, # 是否从文件重新加载模型和向量化器。
)
```

##### `if args.expand_filepaths_to_save_dir:`

将文件路径扩展为包含完整目录的路径，并打印扩展后的文件路径。

```python
if args.expand_filepaths_to_save_dir:
    args.vectorizer_file = os.path.join(args.save_dir,
                                        args.vectorizer_file)

    args.model_state_file = os.path.join(args.save_dir,
                                         args.model_state_file)
    
    print("Expanded filepaths: ")
    print("\t{}".format(args.vectorizer_file))
    print("\t{}".format(args.model_state_file))
```

##### Check CUDA

检查 CUDA 是否可用，并根据检查结果配置设备（CPU 或 GPU）进行计算。

```python
if not torch.cuda.is_available():
    args.cuda = False

print("Using CUDA: {}".format(args.cuda))

args.device = torch.device("cuda" if args.cuda else "cpu")
```

##### Seet seed

```python
set_seed_everywhere(args.seed, args.cuda)
```

##### Handle dirs

```python
handle_dirs(args.save_dir)
```



#### Initializations

加载或创建数据集和向量化器，并基于向量化器的词汇表大小初始化分类器模型。

```python
if args.reload_from_files:
    # training from a checkpoint
    print("Loading dataset and vectorizer")
    dataset = ReviewDataset.load_dataset_and_load_vectorizer(args.review_csv,
                                                             args.vectorizer_file)
else:
    print("Loading dataset and creating vectorizer")
    # create dataset and vectorizer
    dataset = ReviewDataset.load_dataset_and_make_vectorizer(args.review_csv)
    dataset.save_vectorizer(args.vectorizer_file)    
vectorizer = dataset.get_vectorizer()

classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))

```



#### Training loop

##### Initial Setting

```python
classifier = classifier.to(args.device)
loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(classifier.parameters(), lr = args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer,mode = 'min', factor = 0.5,patience = 1)
# - `optimizer=optimizer` 指定要调整的优化器。
# - `mode='min'` 表示学习率在监测的指标停止下降时减少。
# - `factor=0.5` 表示每次减少学习率时，新的学习率是旧学习率的 `0.5` 倍。
# - `patience=1` 表示在监测的指标不改进的次数达到 1 次时减少学习率。
```

- 使用 `to` 方法将 `classifier` 模型移动到指定的设备（`args.device`）。
- 创建一个二元交叉熵损失函数（`BCEWithLogitsLoss`）。`BCEWithLogitsLoss` 结合了一个 Sigmoid 层和二元交叉熵损失，可以直接用于未经过 Sigmoid 激活的输出。
- 使用 `Adam` 优化器优化模型参数。
- 创建一个 `ReduceLROnPlateau` 学习率调度器，当监测的指标不再改进时，自动减少学习率。
- `optimizer=optimizer` 指定要调整的优化器。
- `mode='min'` 表示学习率在监测的指标停止下降时减少。
- `factor=0.5` 表示每次减少学习率时，新的学习率是旧学习率的 `0.5` 倍。
- `patience=1` 表示在监测的指标不改进的次数达到 1 次时减少学习率。

##### Setting bars

```python
train_state = make_train_state(args)

epoch_bar = tqdm_notebook(desc='training routine', total=args.num_epochs,position=0)

dataset.set_split('train')
train_bar = tqdm_notebook(desc='split=train',total=dataset.get_num_batches(args.batch_size), position=1, leave=True)

dataset.set_split('val')
val_bar = tqdm_notebook(desc='split=val',total=dataset.get_num_batches(args.batch_size), position=1, leave=True)
```

初始化训练状态，并使用 `tqdm_notebook` 创建进度条来跟踪训练和验证过程。

`tqdm_notebook`: 是 `tqdm` 库的一部分，专门用于在 Jupyter Notebook 环境中显示进度条。`tqdm` 是一个用于显示循环进度的 Python 库，它可以用于跟踪任务的执行进度，尤其是在长时间运行的任务中。`tqdm_notebook` 提供了一个适合在 Jupyter Notebook 中使用的进度条界面。

- **`iterable`**：任何可以迭代的对象，例如列表、元组、字典或生成器。
- **`desc`**：字符串，进度条前面的描述文本。
- **`total`**：进度条的总步数，如果可以从 `iterable` 推断出来则可选。
- **`position`**：进度条的位置，用于在多个进度条的情况下控制它们的相对位置。
- **`leave`**：布尔值，当迭代完成时是否保留进度条（默认为 `True`）。
- **`ncols`**：进度条的总宽度。
- **`mininterval`**：进度条更新之间的最小间隔时间（以秒为单位）。
- **`maxinterval`**：进度条更新之间的最大间隔时间（以秒为单位）。
- **`unit`**：进度条的单位，例如 `it` 表示迭代次数，`s` 表示秒等。
- **`unit_scale`**：布尔值或数字，用于自动缩放单位。
- **`dynamic_ncols`**：布尔值，是否动态调整进度条的宽度。

##### Main Loop

`try:`

**第一个循环：`epoch`循环**

```python
# (前面还有个 try :)
for epoch_index in range(args.num_epochs):
        train_state['epoch_index'] = epoch_index
        dataset.set_split('train')
        batch_generator = generate_batches(dataset, 
                                           batch_size=args.batch_size, 
                                           device=args.device)
        running_loss = 0.0
        running_acc = 0.0
        classifier.train()
```

- 设置当前 epoch 索引。 

- 将数据集分割设置为训练集。

- 生成批次数据。

- 初始化累积损失和准确性。

- 设置模型为训练模式。

- **第二个循环：`batch`训练循环**

  - ```python
    for batch_index, batch_dict in enumerate(batch_generator):
        optimizer.zero_grad()
        y_pred = classifier(x_in=batch_dict['x_data'].float())
        loss = loss_func(y_pred, batch_dict['y_target'].float())
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)
        loss.backward()
        optimizer.step()
        acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
        running_acc += (acc_t - running_acc) / (batch_index + 1)
        train_bar.set_postfix(loss=running_loss, acc=running_acc, epoch=epoch_index)
        train_bar.update()
    ```

  - 将梯度归零。

  - 计算模型输出。

  - 计算损失并更新累积损失。

  - 反向传播计算梯度。

  - 使用优化器更新模型参数。

  - 计算准确性并更新累积准确性。

  - 更新训练进度条。

- 记录训练结果

  - ```python
    train_state['train_loss'].append(running_loss)
    train_state['train_acc'].append(running_acc)
    ```

- 验证前准备

  - ```python
    dataset.set_split('val')
    batch_generator = generate_batches(dataset,batch_size=args.batch_size,device=args.device)
    running_loss = 0.
    running_acc = 0.
    classifier.eval()
    ```

- 第三个循环：`batch`验证循环

  - ```python
    for batch_index, batch_dict in enumerate(batch_generator):
        y_pred = classifier(x_in=batch_dict['x_data'].float())
        loss = loss_func(y_pred, batch_dict['y_target'].float())
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)
        acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
        running_acc += (acc_t - running_acc) / (batch_index + 1)
        val_bar.set_postfix(loss=running_loss, acc=running_acc, epoch=epoch_index)
        val_bar.update()
    ```

  - (同训练循环)对每个批次进行验证，计算输出、损失和准确性，并更新累积值和进度条。

- 记录验证结果

  - ```python
    train_state['val_loss'].append(running_loss)
    train_state['val_acc'].append(running_acc)
    ```

- 更新训练状态和学习率

  - ```python
    train_state = update_train_state(args=args,model=classifier,train_state=train_state) # 会保存最佳模型，更新早停条件
    scheduler.step(train_state['val_loss'][-1])
    ```

- 更新进度条

  - ```python
    train_bar.n = 0
    val_bar.n = 0
    epoch_bar.update()
    ```

- 检查早停条件

  - ```python
    if train_state['stop_early']:
       break
    ```

`expect`

```python
except KeyboardInterrupt:
    print("Exiting loop")
```

##### Testing

```python
classifier.load_state_dict(torch.load(train_state['model_filename']))
classifier = classifier.to(args.device)

dataset.set_split('test')
batch_generator = generate_batches(dataset, 
                                   batch_size=args.batch_size, 
                                   device=args.device)
running_loss = 0.
running_acc = 0.
classifier.eval()

for batch_index, batch_dict in enumerate(batch_generator):
    # compute the output
    y_pred = classifier(x_in=batch_dict['x_data'].float())

    # compute the loss
    loss = loss_func(y_pred, batch_dict['y_target'].float())
    loss_t = loss.item()
    running_loss += (loss_t - running_loss) / (batch_index + 1)

    # compute the accuracy
    acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
    running_acc += (acc_t - running_acc) / (batch_index + 1)

train_state['test_loss'] = running_loss
train_state['test_acc'] = running_acc
```

在测试集上使用最优模型计算损失和准确性。具体步骤包括加载最佳模型状态，将数据集分割设置为测试集，计算每个批次的输出、损失和准确性，并累积结果。

```python
print("Test loss: {:.3f}".format(train_state['test_loss']))
print("Test Accuracy: {:.2f}".format(train_state['test_acc']))
```

#### Inference

```python
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text
```

```python
def predict_rating(review, classifier, vectorizer, decision_threshold=0.5):
    review = preprocess_text(review)
    
    vectorized_review = torch.tensor(vectorizer.vectorize(review))
    result = classifier(vectorized_review.view(1, -1))
    
    probability_value = F.sigmoid(result).item()
    index = 1
    if probability_value < decision_threshold:
        index = 0

    return vectorizer.rating_vocab.lookup_index(index)
```

```python
test_review = "this is a pretty awesome book"

classifier = classifier.cpu()
prediction = predict_rating(test_review, classifier, vectorizer, decision_threshold=0.5)
print("{} -> {}".format(test_review, prediction))
```

#### Interpretability

```python
# Sort weights
fc1_weights = classifier.fc1.weight.detach()[0]
_, indices = torch.sort(fc1_weights, dim=0, descending=True)
indices = indices.numpy().tolist()

# Top 20 words
print("Influential words in Positive Reviews:")
print("--------------------------------------")
for i in range(20):
    print(vectorizer.review_vocab.lookup_index(indices[i]))
    
print("====\n\n\n")

# Top 20 negative words
print("Influential words in Negative Reviews:")
print("--------------------------------------")
indices.reverse()
for i in range(20):
    print(vectorizer.review_vocab.lookup_index(indices[i]))
```

- 使用 `detach()` 方法从计算图中分离权重张量，以确保不计算梯度。
- 获取第一个输出神经元的权重（因为输出是一个标量）。
- 使用 `torch.sort` 按降序排序权重，并获取排序后的索引。
- 将索引转换为 NumPy 数组，再转换为列表。
- 显示对正面和负面评论影响最大的前20个词汇
