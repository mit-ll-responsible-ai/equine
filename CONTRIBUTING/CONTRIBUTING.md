

## Developer Installation 
1. Clone the git repository and navigate to that directory
    
2. Install virtual environment (the below example assumes conda)

    ```shell
    conda create --name equine python>=3.10
    conda activate equine
    ```
3. Install the code with the extra `tests` dependencies

    ```shell
    pip install -e .[tests]
    ```
    Note that if you are using zsh, you may need to escape the brackets with single quotes:
    ```shell
    pip install -e '.[tests]'
    ```

4. Activate the pre-commit hooks
    ```console
    pre-commit install
    pre-commit run
    ```

## Building the documentation
We use [mkdocs](https://www.mkdocs.org) to build our documentation and autodocumented API. 
Most of these files can be found in the `docs` folder, and the dependencies to generate
your own version of the documentation can be installed via:
    ```shell
    pip install -e .[docs]
    ```

You can build the documentation by issuing `mkdocs build` from the root directory. Once
that has built, you can check out the new documentation via `mkdocs serve` and connecting
your browser to localhost:8000.
