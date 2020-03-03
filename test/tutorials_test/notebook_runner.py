from nbconvert.preprocessors import Preprocessor
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat
import os
import nbformat
from traitlets.config import Config


def run_notebook(notebook_path, notebook_path_two):
    nb_name, _ = os.path.splitext(os.path.basename(notebook_path))
    nb2_name, _ = os.path.splitext(os.path.basename(notebook_path_two))

    dirname = os.path.dirname(notebook_path)
    workdingdir = os.getcwd()
    print(workdingdir)

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

# read second notebook
    with open(notebook_path_two) as f2:
        nb2 = nbformat.read(f2, as_version=4)

    nb['cells'][1]['source'] = nb['cells'][1]['source'] + \
        "\nimport os\nos.chdir({0})\n".format(workdingdir)
    print(nb['cells'][1]['source'])
    nb['cells'][2]['source'] = nb['cells'][2]['source'].replace(
        "data/wikipedia", workdingdir+"/test/data/wikipedia")
    print(nb['cells'][2]['source'])
    nb['cells'][3]['source'] = nb['cells'][3]['source'].replace("750", "10")
    nb['cells'][3]['source'] = nb['cells'][3]['source'].replace("653", "10")
    nb['cells'][-2]['source'] = nb['cells'][-2]['source'].replace(
        "output/tmp/", workdingdir+"/test/output/tmp/")
    proc = ExecutePreprocessor(timeout=600, kernel_name='python')
    proc.allow_errors = True
    proc.preprocess(
        nb, {'metadata': {'path': '/'}})
    output_path = os.path.join(dirname, '{}_all_output.ipynb'.format(nb_name))
    with open(output_path, mode='wt') as f:
        nbformat.write(nb, f)
    errors = []
    for cell in nb.cells:
        if 'outputs' in cell:
            for output in cell['outputs']:
                if output.output_type == 'error':
                    errors.append(output)

    proc = ExecutePreprocessor(timeout=600, kernel_name='python')
    proc.allow_errors = True
    proc.preprocess(
        nb2, {'metadata': {'path': '/'}})
    output_path = os.path.join(dirname, '{}_all_output.ipynb'.format(nb2_name))

    with open(output_path, mode='wt') as f:
        nbformat.write(nb2, f)

    for cell in nb.cells:
        if 'outputs' in cell:
            for output in cell['outputs']:
                if output.output_type == 'error':
                    errors.append(output)

    return nb2, errors


# if __name__ == '__main__':
#     nb, errors = run_notebook('Testing.ipynb')
#     print(errors)
