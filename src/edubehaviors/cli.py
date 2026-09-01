"""Console script for edubehaviors."""

import typer
from rich.console import Console

from edubehaviors import utils

app = typer.Typer()
console = Console()


@app.command()
def main() -> None:
    """Console script for edubehaviors."""
    console.print("Replace this message by putting your code into "
               "edubehaviors.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()


if __name__ == "__main__":
    app()
