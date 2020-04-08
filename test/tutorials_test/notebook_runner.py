from nbconvert.preprocessors import Preprocessor
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat
import os
import nbformat
from traitlets.config import Config


def run_notebook(notebook_path, notebook_path_two, notebook_path_three):
    nb_name, _ = os.path.splitext(os.path.basename(notebook_path))
    nb2_name, _ = os.path.splitext(os.path.basename(notebook_path_two))
    nb3_name, _ = os.path.splitext(os.path.basename(notebook_path_three))

    dirname = os.path.dirname(notebook_path)
    workdingdir = os.getcwd()
    print(workdingdir)

    # read notebooks
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    with open(notebook_path_two) as f2:
        nb2 = nbformat.read(f2, as_version=4)

    with open(notebook_path_three) as f3:
        nb3 = nbformat.read(f3, as_version=4)

    nb['cells'][1]['source'] = nb['cells'][1]['source'] + \
        "\nimport os\nos.chdir({0})\n".format(workdingdir)
    print(nb['cells'][1]['source'])
    nb['cells'][2]['source'] = nb['cells'][2]['source'].replace(
        "data/wikipedia", workdingdir+"/data/wikipedia")
    print(nb['cells'][2]['source'])

    nb['cells'][3]['source'] = nb['cells'][3]['source'].replace("750", "10")
    nb['cells'][3]['source'] = nb['cells'][3]['source'].replace("653", "10")
    nb['cells'][-2]['source'] = nb['cells'][-2]['source'].replace(
        "output/tmp/", workdingdir + "/output/tmp/")

    nb2['cells'][2]['source'] = nb2['cells'][2]['source'].replace(
        "output/tmp/", workdingdir + "/output/tmp/")

    nb2['cells'][-1]['source'] = nb2['cells'][-1]['source'].replace(
        "output/generative/link_hmm", workdingdir + "/output/generative/link_hmm")

    print(nb2['cells'][3]['source'])
    # replace the jsonnet file with the correct dir.
    nb3['cells'][5]['source'] = nb3['cells'][5]['source'].replace(
        "training_config/tutorial.jsonnet", workdingdir + "/tutorials/introduction/training_config/tutorial.jsonnet")

    # Read in the file
    with open(workdingdir + "/tutorials/introduction/training_config/tutorial.jsonnet", 'r') as file:
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace("\"num_epochs\": 50", "\"num_epochs\": 1")
    filedata = filedata.replace(
        "output/generative/link_hmm/", workdingdir+"/output/generative/link_hmm/")
    # Write the file out again
    with open(workdingdir + "/tutorials/introduction/training_config/tutorial.jsonnet", 'w') as file:
        file.write(filedata)

    nb3['cells'][5]['source'] = nb3['cells'][5]['source'].replace(
        "output/discriminative/link_hmm", workdingdir + "/output/discriminative/link_hmm")

    nb3['cells'][-5]['source'] = nb3['cells'][-5]['source'].replace(
        "output/discriminative/link_hmm", workdingdir + "/output/discriminative/link_hmm")
    print(nb3['cells'][5]['source'])

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

    onlyfiles = [f for f in os.listdir(
        workdingdir + "/output/tmp/") if os.isfile(os.join(workdingdir + "/output/tmp/", f))]

    defaultFiles = [f for f in os.listdir(
        workdingdir + "/output/")]

    print("Listing all files", onlyfiles)
    print("Listing default files", defaultFiles)

    proc = ExecutePreprocessor(timeout=600, kernel_name='python')
    proc.allow_errors = True
    proc.preprocess(
        nb2, {'metadata': {'path': '/'}})
    output_path = os.path.join(dirname, '{}_all_output.ipynb'.format(nb2_name))

    with open(output_path, mode='wt') as f:
        nbformat.write(nb2, f)

    proc = ExecutePreprocessor(timeout=600, kernel_name='python')
    proc.allow_errors = True
    proc.preprocess(
        nb3, {'metadata': {'path': '/'}})
    output_path = os.path.join(dirname, '{}_all_output.ipynb'.format(nb3_name))

    with open(output_path, mode='wt') as f:
        nbformat.write(nb3, f)

    for cell in nb.cells:
        if 'outputs' in cell:
            for output in cell['outputs']:
                if output.output_type == 'error':
                    errors.append(output)

    for cell in nb2.cells:
        if 'outputs' in cell:
            for output in cell['outputs']:
                if output.output_type == 'error':
                    errors.append(output)

    for cell in nb3.cells:
        if 'outputs' in cell:
            for output in cell['outputs']:
                if output.output_type == 'error':
                    errors.append(output)

    # change the jsonnet back to original
    # Read in the file
    with open(workdingdir + "/tutorials/introduction/training_config/tutorial.jsonnet", 'r') as file:
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace("\"num_epochs\": 1", "\"num_epochs\": 50")
    filedata = filedata.replace(
        workdingdir+"/output/generative/link_hmm/", "output/generative/link_hmm/")
    # Write the file out again
    with open(workdingdir + "/tutorials/introduction/training_config/tutorial.jsonnet", 'w') as file:
        file.write(filedata)

    return nb3, errors


# if __name__ == '__main__':
#     nb, errors = run_notebook('Testing.ipynb')
#     print(errors)
