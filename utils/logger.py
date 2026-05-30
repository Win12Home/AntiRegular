from utils.variables import ANTI_REGULAR__VERBOSE
from rich import print
from datetime import datetime


class Logger:  # Emmm, nobody cannot realize
    @staticmethod
    def debug(message: str) -> None:
        if ANTI_REGULAR__VERBOSE:
            print(
                f"[red]Anti[/red][blue]Regular[/blue] [{datetime.now().strftime('%H:%M:%S')}] [green]DEBUG:[/green]   {message}"
            )

    @staticmethod
    def info(message: str) -> None:
        print(
            f"[red]Anti[/red][blue]Regular[/blue] [{datetime.now().strftime('%H:%M:%S')}] [blue]INFO:[/blue]    {message}"
        )

    @staticmethod
    def warning(message: str) -> None:
        print(
            f"[red]Anti[/red][blue]Regular[/blue] [{datetime.now().strftime('%H:%M:%S')}] [yellow]WARNING:[/yellow]  {message}"
        )

    @staticmethod
    def error(message: str) -> None:
        print(
            f"[red]Anti[/red][blue]Regular[/blue] [{datetime.now().strftime('%H:%M:%S')}] [red]ERROR:[/red]   {message}"
        )
