"""Load the installation-matched PLAXIS client only when a connection is requested."""
from importlib import import_module


def load_new_server():
    """Delegate to the vendor easy API without a bundled or version-guessing fallback.

    Core package imports and fake-client tests do not need PLAXIS. A missing
    optional client gives installation guidance; errors inside an installed client
    propagate unchanged, so an incompatible environment cannot look like success.
    """
    try:
        easy = import_module('plxscripting.easy')
    except ModuleNotFoundError as exc:
        if exc.name != 'plxscripting':
            raise
        raise ImportError(
            'Install the matching scripting client from your PLAXIS installation '
            '(V22: use its bundled client); the plaxis2023 extra is for 2023.2 only.'
        ) from exc
    return easy.new_server


def new_server(*args, **kwargs):
    """Connect using the explicitly installed client after loading it on demand."""
    return load_new_server()(*args, **kwargs)
