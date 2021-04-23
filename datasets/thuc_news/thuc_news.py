# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""THUNews"""

import csv
import ctypes
import os

import datasets

csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))

_CITATION = """\
@misc{xujianhua,
    title={page xxx},
    author={Xiang Zhang and Junbo Zhao and Yann LeCun},
    year={2015},
    eprint={1509.01626},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
"""

_DESCRIPTION = """\
THUCTC(THU Chinese Text Classification)是由清华大学自然语言处理实验室推出的中文文本分类工具包，能够自动高效地实现用户自定义的文本分类语料的训练、\
评测、分类功能。文本分类通常包括特征选取、特征降维、分类模型学习三个步骤。如何选取合适的文本特征并进行降维，是中文文本分类的挑战性问题。、
我组根据多年在中文文本分类的研究经验，在THUCTC中选取二字串bigram作为特征单元，特征降维方法为Chi-square，权重计算方法为tfidf，、
分类模型使用的是LibSVM或LibLinear。THUCTC对于开放领域的长文本具有良好的普适性，不依赖于任何中文分词工具的性能，具有准确率高、测试速度快的优点。
"""

_DATA_URL = "http://127.0.0.1/thuc_news.zip"

_CLS = ['体育', '娱乐', '家居', '彩票', '房产', '教育', '时尚', '时政', '星座', '游戏', '社会', '科技', '股票', '财经']

class THUC_News(datasets.GeneratorBasedBuilder):
    """Sogou News dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "content": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(
                        names=_CLS
                    ),
                }
            ),
            # No default supervised_keys (as we have to pass both premise
            # and hypothesis as input).
            supervised_keys=None,
            homepage="",  # didn't find a real homepage
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_DATA_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(dl_dir, "thuc_news", "test.txt")}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join(dl_dir, "thuc_news", "train.txt")}
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        with open(filepath, encoding="utf-8") as txt_file:
            data = txt_file.readlines()
            for id_, row in enumerate(data):
                row = row.split('\t')
                yield id_, {"content": row[1], "label": _CLS.index(row[0])}
