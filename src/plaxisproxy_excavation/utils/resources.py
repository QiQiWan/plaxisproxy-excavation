from importlib.resources import files

def get_default_template_path(name: str) -> str:
    return str(files("excavation_sim.data").joinpath(name))