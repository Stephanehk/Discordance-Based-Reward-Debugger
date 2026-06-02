"""CLI shim so `python -m web_ui.cli <subcommand>` works."""
from web_ui.cli.main import main

if __name__ == "__main__":
    main()
