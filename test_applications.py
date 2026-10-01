"""Checks for conditional continuation scoring and interactive applications."""

import io
import math
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from ngram_lm import NGramLanguageModel, interactive_session


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        self.model = NGramLanguageModel(2).fit_tokenized([
            ['the', 'cat', 'eats', 'fish'],
            ['the', 'cat', 'likes', 'fish'],
            ['the', 'dog', 'eats', 'meat'],
        ])

    def test_conditional_score_excludes_context_and_extends_history(self):
        # P(eats fish | the cat) = P(eats | cat) P(fish | eats) = 1/4.
        score = self.model.continuation_log_probability('the cat', 'eats fish')
        self.assertAlmostEqual(math.exp(score), 1 / 4)
        self.assertEqual(self.model.continuation_log_probability('unseen cat', 'eats fish'), score)
        self.assertEqual(self.model.continuation_log_probability('the cat', ''), 0)
        self.assertEqual(self.model.continuation_log_probability('the cat', 'fish'), -math.inf)

    def test_ranking_and_length_normalization(self):
        rows = self.model.rank_continuations('the cat', ['fish', 'eats fish', 'likes fish'])
        self.assertEqual([row['candidate'] for row in rows], ['likes fish', 'eats fish', 'fish'])
        normalized = self.model.rank_continuations('the cat', ['eats fish'], True)[0]
        self.assertAlmostEqual(normalized['score'], math.log(1 / 4) / 2)
        with self.assertRaises(ValueError):
            self.model.rank_continuations('cat', ['!!!'])

    def test_unigram_and_trigram_contexts(self):
        sentences = [['a', 'b', 'c'], ['x', 'b', 'd']]
        tri = NGramLanguageModel(3).fit_tokenized(sentences)
        self.assertEqual(tri.continuation_log_probability('a b', 'c'), 0)
        self.assertEqual(tri.continuation_log_probability('x b', 'c'), -math.inf)
        uni = NGramLanguageModel(1).fit_tokenized(sentences)
        self.assertAlmostEqual(uni.continuation_log_probability('anything', 'b c'), math.log(1 / 18))

    def test_interactive_prediction_and_ranking(self):
        answers = ['1']
        for context in ['the cat', 'the dog', 'eats', 'likes', 'the']:
            answers.extend([context, 'fish'])
        answers.extend(['', '2', 'the cat', 'eats fish', 'likes fish', '', '1', '3', '0'])
        with patch('builtins.input', side_effect=answers), redirect_stdout(io.StringIO()) as output:
            trials = interactive_session(self.model, top_k=2)
        self.assertEqual(len(trials), 7)
        self.assertEqual(trials[0]['prediction'], 'eats')
        self.assertEqual(trials[5]['prediction'], 'likes fish')
        self.assertIn('Da thu 5 contexts', output.getvalue())

    def test_eof_exit(self):
        with patch('builtins.input', side_effect=EOFError), redirect_stdout(io.StringIO()):
            self.assertEqual(interactive_session(self.model), [])


if __name__ == '__main__':
    unittest.main()
