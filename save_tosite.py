#%% 
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor, ExtractOutputPreprocessor
from traitlets.config import Config
import os
import shutil

def convert_notebook(notebook_path, output_html_path, additional_files=None):
    """
    Converts a Jupyter notebook to an HTML file suitable for GitHub pages.

    Args:
    notebook_path (str): Path to the Jupyter notebook file.
    output_html_path (str): Path where the HTML file will be saved.
    additional_files (list of str, optional): List of paths to additional files required for the notebook.
    """
    # Copy additional files to the current working directory, if provided
    if additional_files:
        for file_path in additional_files:
            shutil.copy(file_path, '.')

    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Execute the notebook
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': '.'}})

    # Convert the notebook to HTML
    html_exporter = HTMLExporter()
    html_exporter.template_name = 'classic'

    # Configuration for GitHub Pages compatibility
    c = Config()
    c.HTMLExporter.preprocessors = [ExtractOutputPreprocessor()]

    # Export the notebook to HTML
    (body, resources) = html_exporter.from_notebook_node(nb, resources=c)
    
    # Write the HTML to a file
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(body)

    return output_html_path

# Example usage of the function
notebook_path = 'report_generator.ipynb'
output_html_path = 'docs/html_test.html'
# additional_files = ['path_to_additional_file1', 'path_to_additional_file2'] # Optional

# Convert the notebook
converted_html_path = convert_notebook(notebook_path, output_html_path, None)
print(f"Notebook converted to HTML: {converted_html_path}")

#%%
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor, ExtractOutputPreprocessor
from traitlets.config import Config
import os
import shutil

def filter_code_cells(cell):
    """
    Filter function to exclude code cells.
    """
    return cell.cell_type == 'code'

def convert_notebook_hide_code(notebook_path, output_html_path, additional_files=None):
    """
    Converts a Jupyter notebook to an HTML file suitable for GitHub pages, hiding code cells.

    Args:
    notebook_path (str): Path to the Jupyter notebook file.
    output_html_path (str): Path where the HTML file will be saved.
    additional_files (list of str, optional): List of paths to additional files required for the notebook.
    """
    # Copy additional files to the current working directory, if provided
    if additional_files:
        for file_path in additional_files:
            shutil.copy(file_path, '.')

    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Execute the notebook
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': '.'}})

    # Configure HTMLExporter
    c = Config()
    c.HTMLExporter.preprocessors = [ExtractOutputPreprocessor()]
    c.TemplateExporter.exclude_input_cells = True  # Exclude code cells

    # Convert the notebook to HTML
    html_exporter = HTMLExporter(config=c)
    (body, resources) = html_exporter.from_notebook_node(nb, resources=c)
    
    # Write the HTML to a file
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(body)

    return output_html_path

# Example usage of the function
notebook_path = 'report_generator.ipynb'
output_html_path = 'docs/html_test3.html'
# additional_files = ['path_to_additional_file1', 'path_to_additional_file2'] # Optional

# Convert the notebook
converted_html_path = convert_notebook_hide_code(notebook_path, output_html_path, None)
print(f"Notebook converted to HTML: {converted_html_path}")

# %%