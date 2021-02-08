import unittest
from load_dataset import load_zeshel_data, load_mentions, load_entities
from dataloader import EntitySet, MentionSet
from transformers import BertTokenizer
import torch


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.docs = {
            "a": {
                "title": "Barbars",
                "text": 'Baibars Baibars or Baybars ( , " al - Malik al - Zahir Rukn al - Din Baibars al - Bunduqdari " ) , nicknamed Abu l - Futuh ( Arabic : أبو الفتوح ) ( 1223 – 1 July 1277 ) was the fourth Sultan of Egypt from the Mamluk Bahri dynasty . He was one of the commanders of the Egyptian forces that inflicted a devastating defeat on the Seventh Crusade of King Louis IX of France . He also led the vanguard of the Egyptian army at the Battle of Ain Jalut in 1260 , which marked the first substantial defeat of the Mongol army , and is considered a turning point in history .',
                "document_id": "a",
            }
        }
        self.mentions = {
            "text": "the fourth Sultan of Egypt",
            "end_index": 73,
            "context_document_id": "a",
            "label_document_id": "a",
            "start_index": 69,
        }
        self.max_len = 25
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def test_entityset(self):
        entityset = EntitySet(self.docs, self.max_len, self.tokenizer)
        ids, masks = entityset[0]
        self.assertTrue(
            torch.equal(
                ids,
                torch.tensor(
                    [
                        101,
                        3347,
                        8237,
                        2015,
                        3,
                        21790,
                        8237,
                        2015,
                        21790,
                        8237,
                        2015,
                        2030,
                        3016,
                        8237,
                        2015,
                        1006,
                        1010,
                        1000,
                        2632,
                        1011,
                        14360,
                        2632,
                        1011,
                        23564,
                        102,
                    ]
                ),
            ),
            "Entity encoding is wrong",
        )
        self.assertTrue(
            torch.equal(
                masks,
                torch.tensor(
                    [
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                    ]
                ),
            ),
            "Entity encoding is wrong",
        )

    def test_mentionset(self):
        mentionset = MentionSet(self.mentions, self.docs, self.tokenizer, self.max_len)
        window = mentionset.get_mention_window(self.mentions)
        self.assertEqual(
            window,
            [
                "the",
                "egyptian",
                "forces",
                "that",
                "inflicted",
                "a",
                "devastating",
                "defeat",
                "[unused0]",
                "the",
                "fourth",
                "sultan",
                "of",
                "egypt",
                "[unused1]",
                "king",
                "louis",
                "ix",
                "of",
                "france",
                ".",
                "he",
                "also",
            ],
            "Window Extraction is wrong",
        )


if __name__ == "__main__":
    unittest.main()
