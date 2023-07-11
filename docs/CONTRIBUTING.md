

## Developer Installation 
1. Clone the [git repository](https://github.com/mit-ll-responsible-ai/equine) and navigate to that directory
    
2. Install virtual environment (the below example assumes conda)

    ```shell
    conda create --name equine python>=3.10
    conda activate equine
    ```

3. Install the code with the extra `tests` dependencies

    ```shell
    pip install -e .'[tests]'
    ```

4. Activate the pre-commit hooks
    ```console
    pre-commit install
    pre-commit run
    ```


We prefer that any contributed code be outfitted with contracts from `icontract` and tested with `hypothesis`. 
This combination frequently means that the tests require few (if any) actual post-checks -- if the contracts are
well-written, then `hypothesis` can generate reasonable tests that will explore the bounds of the contracts
for each method.  Assuming that tests pass, then make a pull request. 



## Documentation
In the [MIT-LL Responsible AI GitHub organization](https://github.com/mit-ll-responsible-ai), we use the numpy/scipy format for docstrings.

We use [mkdocs](https://www.mkdocs.org) to build our documentation and autodocumented API. 
Most of these files can be found in the `docs` folder, and the dependencies to generate
your own version of the documentation can be installed via:
    ```shell
    pip install -e .'[docs]'
    ```

You can test new documentation by issuing `mkdocs serve` from the root directory. Once
it has built, you can check out the new documentation by connecting your browser to localhost:8000.
