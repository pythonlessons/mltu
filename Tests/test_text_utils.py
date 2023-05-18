import unittest

from mltu.utils.text_utils import edit_distance, get_cer, get_wer 

class TestTextUtils(unittest.TestCase):

    def test_edit_distance(self):
        """ This unit test includes several test cases to cover different scenarios, including no errors, 
        substitution errors, insertion errors, deletion errors, and a more complex case with multiple 
        errors. It also includes a test case for empty input.
        """
        # Test simple case with no errors
        prediction_tokens = ["A", "B", "C"]
        reference_tokens = ["A", "B", "C"]
        self.assertEqual(edit_distance(prediction_tokens, reference_tokens), 0)
        
        # Test simple case with one substitution error
        prediction_tokens = ["A", "B", "D"]
        reference_tokens = ["A", "B", "C"]
        self.assertEqual(edit_distance(prediction_tokens, reference_tokens), 1)
        
        # Test simple case with one insertion error
        prediction_tokens = ["A", "B", "C"]
        reference_tokens = ["A", "B", "C", "D"]
        self.assertEqual(edit_distance(prediction_tokens, reference_tokens), 1)
        
        # Test simple case with one deletion error
        prediction_tokens = ["A", "B"]
        reference_tokens = ["A", "B", "C"]
        self.assertEqual(edit_distance(prediction_tokens, reference_tokens), 1)
        
        # Test more complex case with multiple errors
        prediction_tokens = ["A", "B", "C", "D", "E"]
        reference_tokens = ["A", "C", "B", "F", "E"]
        self.assertEqual(edit_distance(prediction_tokens, reference_tokens), 3)
        
        # Test empty input
        prediction_tokens = []
        reference_tokens = []
        self.assertEqual(edit_distance(prediction_tokens, reference_tokens), 0)

    def test_get_cer(self):
        # Test simple case with no errors
        preds = ["A B C"]
        target = ["A B C"]
        self.assertEqual(get_cer(preds, target), 0)
        
        # Test simple case with one character error
        preds = ["A B C"]
        target = ["A B D"]
        self.assertEqual(get_cer(preds, target), 1/5)
        
        # Test simple case with multiple character errors
        preds = ["A B C"]
        target = ["D E F"]
        self.assertEqual(get_cer(preds, target), 3/5)
        
        # Test empty input
        preds = []
        target = []
        self.assertEqual(get_cer(preds, target), 0)

        # Test simple case with different word lengths
        preds = ["ABC"]
        target = ["ABCDEFG"]
        self.assertEqual(get_cer(preds, target), 4/7)

    def test_get_wer(self):
        # Test simple case with no errors
        preds = "A B C"
        target = "A B C"
        self.assertEqual(get_wer(preds, target), 0)
        
        # Test simple case with one word error
        preds = "A B C"
        target = "A B D"
        self.assertEqual(get_wer(preds, target), 1/3)
        
        # Test simple case with multiple word errors
        preds = "A B C"
        target = "D E F"
        self.assertEqual(get_wer(preds, target), 1)
        
        # Test empty input
        preds = ""
        target = ""
        self.assertEqual(get_wer(preds, target), 0)

        # Test simple case with different sentence lengths
        preds = ["ABC"]
        target = ["ABC DEF"]
        self.assertEqual(get_wer(preds, target), 1)


if __name__ == "__main__":
    unittest.main()
