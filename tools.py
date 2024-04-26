from langchain_core.tools import tool
from langchain_experimental.utilities import pythonrepl
from typing import annotated



@tool
def python_repl(
    code: annotated[str, "the python code to execute to generate your chart."]
):
    """use this to execute python code. if you want to see the output of a value,
    you should print it out with `print(...)`. this is visible to the user."""

    repl = pythonrepl()
    try:
        result = repl.run(code)
    except baseexception as e:
        return f"failed to execute. error: {repr(e)}"
    return f"succesfully executed:\n```python\n{code}\n```\nstdout: {result}"
