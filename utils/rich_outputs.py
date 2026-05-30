from rich import print
from rich import traceback


def install_traceback() -> None:
    traceback.install(
        extra_lines=5,  # I like code
        max_frames=999,  # Wow my console has been bombed yeah!!! I am proud!!! My computer bombed!!!
        code_width=120,  # SMALL
    )


def show_anti_regular_ascii() -> None:
    ascii_art = """
[red] █████╗ ███╗   ██╗████████╗██╗                              
██╔══██╗████╗  ██║╚══██╔══╝██║                            
███████║██╔██╗ ██║   ██║   ██║                            
██╔══██║██║╚██╗██║   ██║   ██║                            
██║  ██║██║ ╚████║   ██║   ██║                            
╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝                            [/red]
                                                          
[blue]██████╗ ███████╗ ██████╗ ██╗   ██╗██╗      █████╗ ██████╗ 
██╔══██╗██╔════╝██╔════╝ ██║   ██║██║     ██╔══██╗██╔══██╗
██████╔╝█████╗  ██║  ███╗██║   ██║██║     ███████║██████╔╝
██╔══██╗██╔══╝  ██║   ██║██║   ██║██║     ██╔══██║██╔══██╗
██║  ██║███████╗╚██████╔╝╚██████╔╝███████╗██║  ██║██║  ██║
╚═╝  ╚═╝╚══════╝ ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝[/blue]
    """
    print(ascii_art)
