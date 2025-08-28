import json
import datasets
from datasets import DownloadManager, DatasetInfo


class CMRC2018TRIAL(datasets.GeneratorBasedBuilder):

    def _info(self):
        """Define the dataset info by columns information."""
        return datasets.DatasetInfo(
            description="CMRC 2018 trial",
            features=datasets.Features({
                'id' : datasets.Value('string'),
                'context' : datasets.Value('string'),
                'question' : datasets.Value('string'),
                'answers' : datasets.features.Sequence({
                    'text' : datasets.Value('string'),
                    'answer_start' : datasets.Value('int32'),
                })
            })
        )

    def _split_generators(self, dl_manager: DownloadManager):
        """Returns datasets.SplitGenerator for spliting the dataset."""
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                        gen_kwargs={'filepath' : './data/cmrc2018_trial.json'})]

    def _generate_examples(self, filepath):
        """Generates specific examples from the dataset."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for example in data['data']:
                for paragraph in example['paragraphs']:
                    context = paragraph['context'].strip()
                    for qa in paragraph['qas']:
                        question = qa['question'].strip()
                        id_ = qa['id']

                        answer_starts = [answer['answer_start'] for answer in qa['answers']]
                        answers = [answer['text'].strip() for answer in qa['answers']]

                        yield id_, {
                            "context": context,
                            "question": question,
                            "id": id_,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            }
                        }
