import unittest
import wiser
from .notebook_runner import run_notebook
import os


class TestNotebook(unittest.TestCase):

    def test_runner(self):
        nb, errors = run_notebook(
            os.getcwd()+'/tutorials/introduction/Intro Tutorial 1 - Tagging and Linking Rules.ipynb')
        self.assertEqual(errors, [])


if __name__ == '__main__':
    unittest.main()
